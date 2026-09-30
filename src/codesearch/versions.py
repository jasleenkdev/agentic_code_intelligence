"""Versioned snippet libraries (P1) — indexes keyed by version, rebuilt incrementally.

A *library* (e.g. `gladys`, `apps`) has any number of *versions* (a git commit / tag for a
codebase, a free label for a snippet set). Each version is stored under
`index/versions/<library>/<version>/`:

    manifest.json        counts, timings (list / chunk / encode / write), models, source
    snippets.jsonl       one snippet per line (id, text, path, symbol, lines, lineage key, hash)
    emb__<model>.npy     L2-normalised embeddings aligned with snippets.jsonl

Incrementality comes from two content-addressed caches:
  * chunking: a file's snippets are cached by its git blob SHA (already a content hash), so only
    files that changed between commits are re-parsed;
  * encoding: `encoders.EmbeddingCache` keys vectors by sha1(model, text), so only snippets whose
    text changed are re-encoded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src import config
from src.codesearch.encoders import Encoder, slug

logger = logging.getLogger(__name__)

VERSIONS_DIR = Path(os.environ.get("CODESEARCH_VERSIONS_DIR", "") or config.INDEX_DIR / "versions")
CHUNK_CACHE_DIR = Path(
    os.environ.get("CODESEARCH_CHUNK_CACHE", "") or config.INDEX_DIR / "chunk_cache"
)
MAX_LINES = 120
OVERLAP = 10


@dataclass
class Snippet:
    """One retrievable unit of one version."""

    id: str
    text: str
    version: str = ""
    path: str = ""
    symbol: str = ""
    start_line: int = 0
    end_line: int = 0
    lineage_key: str = ""  # stable identity across versions (path::symbol#window, or given)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha1(self.text.encode()).hexdigest()

    def embed_text(self) -> str:
        """Text given to the encoder: a short location header for codebase snippets."""
        if self.path:
            head = f"# file: {self.path}\n"
            if self.symbol:
                head += f"# symbol: {self.symbol}\n"
            return head + self.text
        return self.text


# ---------------------------------------------------------------- codebase (git) source


def _git(git_dir: Path, *args: str, input_: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", f"--git-dir={git_dir}", *args], check=True, capture_output=True, input=input_
    ).stdout


def resolve_commit(git_dir: Path, ref: str) -> str:
    return _git(git_dir, "rev-parse", f"{ref}^{{commit}}").decode().strip()


def _chunk_js(path: str, source: bytes) -> list[dict]:
    """Function/method-level chunks of one JS file (whole-file windows if it has none)."""
    from src.indexer.extractor import extract

    try:
        facts, _errors = extract(source)
    except Exception as e:  # parse failure: log and fall back to line windows
        logger.debug("parse failed for %s: %s", path, e)
        facts = None
    lines = source.decode("utf-8", errors="replace").split("\n")
    out: list[dict] = []
    if facts is not None:
        kinds = {s["local_id"]: s["kind"] for s in facts.symbols}
        for s in facts.symbols:
            if s["kind"] not in {"function", "arrow", "method"}:
                continue
            parent = kinds.get(s["parent_local"])
            if parent not in {"module", "class"} and s["kind"] != "method":
                continue
            if s["end_line"] - s["start_line"] < 2:
                continue
            for w, (a, b) in enumerate(_windows(s["start_line"], s["end_line"])):
                out.append(
                    {
                        "symbol": s["qualname"],
                        "start_line": a,
                        "end_line": b,
                        "text": "\n".join(lines[a - 1 : b]),
                        "lineage_key": f"{path}::{s['qualname']}#{w}",
                    }
                )
    if not out and any(ln.strip() for ln in lines):
        for w, (a, b) in enumerate(_windows(1, len(lines))):
            out.append(
                {
                    "symbol": "<module>",
                    "start_line": a,
                    "end_line": b,
                    "text": "\n".join(lines[a - 1 : b]),
                    "lineage_key": f"{path}::<module>#{w}",
                }
            )
    return out


def _windows(start: int, end: int) -> list[tuple[int, int]]:
    if end - start + 1 <= MAX_LINES:
        return [(start, end)]
    spans, a = [], start
    while a <= end:
        b = min(a + MAX_LINES - 1, end)
        spans.append((a, b))
        if b == end:
            break
        a = b - OVERLAP + 1
    return spans


def snippets_from_git(
    git_dir: Path,
    ref: str,
    include: tuple[str, ...] = config.INCLUDE_DIRS,
    exclude_dirs: frozenset[str] = config.EXCLUDE_DIR_NAMES,
    stats: dict | None = None,
) -> list[Snippet]:
    """All snippets of a codebase at `ref`, re-parsing only blobs not seen before."""
    stats = stats if stats is not None else {}
    commit = resolve_commit(git_dir, ref)
    t = time.perf_counter()
    listing = _git(git_dir, "ls-tree", "-r", "-l", commit, "--", *include).decode()
    files = []
    for line in listing.splitlines():
        meta, path = line.split("\t", 1)
        _mode, kind, sha, size = meta.split()
        parts = path.split("/")
        if kind != "blob" or Path(path).suffix not in config.JS_EXTENSIONS:
            continue
        if any(p in exclude_dirs for p in parts[:-1]) or path.endswith(".min.js"):
            continue
        if size != "-" and int(size) > config.MAX_FILE_BYTES:
            continue
        files.append((path, sha))
    stats["list_s"] = time.perf_counter() - t

    t = time.perf_counter()
    CHUNK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CHUNK_CACHE_DIR / f"{slug(str(git_dir))[-60:]}.pkl"
    cache: dict[str, list[dict]] = {}
    if cache_file.exists():
        try:
            cache = pickle.loads(cache_file.read_bytes())
        except Exception as e:  # corrupt cache: rebuild it
            logger.warning("chunk cache unreadable (%s); rebuilding", e)
    todo = [(p, s) for p, s in files if f"{p}@{s}" not in cache]
    if todo:
        blobs = _read_blobs(git_dir, [s for _, s in todo])
        for (path, sha), data in zip(todo, blobs, strict=True):
            if sum(len(x) for x in data.split(b"\n")) / max(data.count(b"\n"), 1) > 300:
                cache[f"{path}@{sha}"] = []  # minified
                continue
            cache[f"{path}@{sha}"] = _chunk_js(path, data)
        cache_file.write_bytes(pickle.dumps(cache))
    stats.update(
        {
            "files": len(files),
            "files_parsed": len(todo),
            "files_reused": len(files) - len(todo),
            "chunk_s": time.perf_counter() - t,
            "commit": commit,
        }
    )
    out = []
    for path, sha in files:
        for c in cache[f"{path}@{sha}"]:
            out.append(
                Snippet(
                    id=f"{path}:{c['start_line']}-{c['end_line']}",
                    text=c["text"],
                    path=path,
                    symbol=c["symbol"],
                    start_line=c["start_line"],
                    end_line=c["end_line"],
                    lineage_key=c["lineage_key"],
                )
            )
    return out


def _read_blobs(git_dir: Path, shas: list[str]) -> list[bytes]:
    """Batch-read blobs (fetches missing ones once for a blobless clone)."""
    raw = _git(git_dir, "cat-file", "--batch", input_=("\n".join(shas) + "\n").encode())
    out, i = [], 0
    for _ in shas:
        nl = raw.index(b"\n", i)
        header = raw[i:nl].split()
        size = int(header[2])
        out.append(raw[nl + 1 : nl + 1 + size])
        i = nl + 1 + size + 1
    return out


# ---------------------------------------------------------------- snippet-set source


def snippets_from_jsonl(path: Path) -> list[Snippet]:
    """A snippet set: one JSON object per line with `id`, `text` and optional `lineage_key`."""
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(
                Snippet(id=str(d["id"]), text=d["text"], lineage_key=d.get("lineage_key", d["id"]))
            )
    return out


# ---------------------------------------------------------------- versioned index


def version_dir(library: str, version: str) -> Path:
    return VERSIONS_DIR / slug(library) / re.sub(r"[^A-Za-z0-9._-]+", "_", version)


def index_version(
    library: str,
    version: str,
    snippets: list[Snippet],
    models: list[str],
    source: dict | None = None,
    extra_stats: dict | None = None,
    snippet_view_name: str = "raw",
) -> dict:
    """Encode (incrementally) and persist one version. Returns its manifest.

    `snippet_view_name` is applied to snippet-set entries (no path) so a library is embedded
    exactly like the submitted pipeline embeds the APPS corpus (shared cache, same ranking);
    codebase snippets keep their location header and raw code.
    """
    from src.codesearch.preprocess import snippet_view

    t_all = time.perf_counter()
    d = version_dir(library, version)
    d.mkdir(parents=True, exist_ok=True)
    for s in snippets:
        s.version = version
    enc_stats = {}
    for key in models:
        enc = Encoder(key)
        t = time.perf_counter()
        texts = [
            s.embed_text() if s.path else snippet_view(s.text, snippet_view_name) for s in snippets
        ]
        emb = enc.encode(texts, "doc")
        np.save(d / f"emb__{key}.npy", emb.astype(np.float32))
        enc_stats[key] = {**enc.last_stats, "seconds": time.perf_counter() - t}
    t = time.perf_counter()
    with (d / "snippets.jsonl").open("w") as f:
        for s in snippets:
            f.write(json.dumps(asdict(s)) + "\n")
    write_s = time.perf_counter() - t
    manifest = {
        "library": library,
        "version": version,
        "snippets": len(snippets),
        "models": models,
        "snippet_view": snippet_view_name,
        "source": source or {},
        "encode": enc_stats,
        "write_s": write_s,
        "timings": extra_stats or {},
        "total_s": time.perf_counter()
        - t_all
        + sum(v for k, v in (extra_stats or {}).items() if k.endswith("_s")),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        "indexed %s@%s: %d snippets, encoded %s in %.1f s",
        library,
        version,
        len(snippets),
        {k: v["encoded"] for k, v in enc_stats.items()},
        manifest["total_s"],
    )
    return manifest


def list_versions(library: str) -> list[dict]:
    root = VERSIONS_DIR / slug(library)
    out = []
    for m in sorted(root.glob("*/manifest.json")) if root.exists() else []:
        out.append(json.loads(m.read_text()))
    return sorted(out, key=lambda m: m.get("source", {}).get("order", m["built_at"]))


def list_libraries() -> list[str]:
    return (
        sorted(p.name for p in VERSIONS_DIR.iterdir() if p.is_dir())
        if VERSIONS_DIR.exists()
        else []
    )


def load_version(library: str, version: str, model: str) -> tuple[list[Snippet], np.ndarray]:
    d = version_dir(library, version)
    snippets = [Snippet(**json.loads(x)) for x in (d / "snippets.jsonl").read_text().splitlines()]
    return snippets, np.load(d / f"emb__{model}.npy")
