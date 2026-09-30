"""Index build orchestration: walk → parse/extract → store → resolve → catalog → chunks → BM25 →
embeddings, with a manifest recording counts, skips, timings, peak memory and on-disk sizes."""

from __future__ import annotations

import hashlib
import json
import logging
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.indexer import store
from src.indexer.extractor import extract, make_parser
from src.indexer.walker import walk_repo

logger = logging.getLogger(__name__)

STAGES = ("structure", "catalog", "chunks", "bm25", "vectors")


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def _repo_commit(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_structure(conn, repo_root: Path) -> dict:
    """Parse every file and store symbols/calls/strings/imports/exports/bindings."""
    walk = walk_repo(repo_root)
    parser = make_parser()
    skipped = list(walk.skipped)
    indexed = 0
    parse_error_files = 0
    for i, path in enumerate(walk.files, 1):
        rel = path.relative_to(repo_root).as_posix()
        try:
            src = path.read_bytes()
        except OSError as exc:
            logger.warning("skip %s: unreadable (%s)", rel, exc)
            skipped.append({"path": rel, "reason": f"unreadable: {exc}"})
            continue
        try:
            facts, errors = extract(src, parser)
        except Exception as exc:  # noqa: BLE001 — a bad file must never stop the run
            logger.warning("skip %s: extraction failed (%s)", rel, exc)
            skipped.append({"path": rel, "reason": f"extraction failed: {exc!r}"[:200]})
            continue
        if errors:
            parse_error_files += 1
            logger.debug("%s: %d parse error nodes (partial facts kept)", rel, errors)
        store.insert_file(conn, rel, facts, errors, hashlib.sha1(src).hexdigest())
        indexed += 1
        if i % 500 == 0:
            logger.info("structure: %d/%d files", i, len(walk.files))
    conn.commit()
    return {
        "files_indexed": indexed,
        "files_with_parse_errors": parse_error_files,
        "files_skipped": skipped,
    }


def _dir_sizes() -> dict[str, int]:
    sizes = {}
    for p in sorted(config.INDEX_DIR.glob("*")):
        if not p.is_file() or p.name.endswith((".tmp", "-wal", "-shm")):
            continue
        # Only the active embedding variant counts; other vectors__* files are experiments.
        if p.name.startswith("vector") and p.name not in {
            config.VECTORS_PATH.name,
            config.VECTOR_IDS_PATH.name,
        }:
            continue
        sizes[p.name] = p.stat().st_size
    return sizes


def build_index(stages: tuple[str, ...] = STAGES, repo_root: Path = config.REPO_ROOT) -> dict:
    """Build the requested stages. 'structure' rebuilds the DB from scratch (idempotent)."""
    if not repo_root.is_dir():
        raise FileNotFoundError(f"Target repo not found: {repo_root}")
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    manifest: dict = {}
    if config.MANIFEST_PATH.exists() and "structure" not in stages:
        manifest = json.loads(config.MANIFEST_PATH.read_text())
        timings = manifest.get("timings_sec", {})

    t_all = time.perf_counter()
    if "structure" in stages:
        from src.graph.resolver import resolve_calls

        tmp = config.DB_PATH.with_suffix(".db.tmp")
        tmp.unlink(missing_ok=True)
        conn = store.connect(tmp)
        store.create_schema(conn)
        t0 = time.perf_counter()
        struct = build_structure(conn, repo_root)
        timings["parse_extract"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        store.create_indexes(conn)
        resolve_stats = resolve_calls(conn)
        conn.commit()
        timings["resolve_calls"] = round(time.perf_counter() - t0, 2)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        shutil.move(tmp, config.DB_PATH)
        for suffix in ("-wal", "-shm"):
            Path(str(tmp) + suffix).unlink(missing_ok=True)
        manifest.update(
            {
                "repo_commit": _repo_commit(repo_root),
                "repo_root": str(repo_root),
                "include_dirs": list(config.INCLUDE_DIRS),
                "files_indexed": struct["files_indexed"],
                "files_with_parse_errors": struct["files_with_parse_errors"],
                "files_skipped": struct["files_skipped"],
                "resolution": resolve_stats,
            }
        )

    conn = store.connect(config.DB_PATH)
    if "catalog" in stages:
        from src.catalog.builder import build_catalog

        t0 = time.perf_counter()
        manifest["catalog"] = build_catalog(conn, repo_root)
        conn.commit()
        timings["catalog"] = round(time.perf_counter() - t0, 2)
    if "chunks" in stages:
        from src.retrieval.chunker import build_chunks

        t0 = time.perf_counter()
        manifest["chunk_stats"] = build_chunks(conn, repo_root)
        conn.commit()
        timings["chunks"] = round(time.perf_counter() - t0, 2)
    if "bm25" in stages:
        from src.retrieval.bm25 import build_bm25

        t0 = time.perf_counter()
        build_bm25(conn)
        timings["bm25"] = round(time.perf_counter() - t0, 2)
    if "vectors" in stages:
        from src.retrieval.embeddings import build_vectors

        t0 = time.perf_counter()
        manifest["embedding"] = build_vectors(conn)
        timings["vectors"] = round(time.perf_counter() - t0, 2)

    counts = {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608 — fixed names
        for t in (
            "files",
            "symbols",
            "calls",
            "edges",
            "strings",
            "imports",
            "exports",
            "const_refs",
            "bindings",
            "chunks",
            "catalog",
        )
    }
    conn.close()
    timings["total_this_run"] = round(time.perf_counter() - t_all, 2)
    manifest.update(
        {
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stages_run": list(stages),
            "counts": counts,
            "embedding_model": config.EMBED_MODEL,
            "timings_sec": timings,
            "peak_rss_mb_this_run": round(peak_rss_mb(), 1),
        }
    )
    manifest["index_bytes"] = _dir_sizes()
    manifest["index_bytes_total"] = sum(manifest["index_bytes"].values())
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest
