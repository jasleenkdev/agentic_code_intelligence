"""Standalone search engine for the demo, CLI and API (R-D-1).

Loads versioned libraries from `index/versions/` (built by `scripts/index_version.py`), runs the
same multi-pass pipeline as the official MTEB run, and returns ranked snippets together with
everything the UI shows: query category, pre-processing, per-pass timings, latency, versions and
lineages. Retrieval only — no generation.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, replace

import numpy as np

from src.codesearch import versions as vstore
from src.codesearch.lineage import (
    collapse,
    infer_lineages,
    members_by_lineage,
    parse_version_hint,
)
from src.codesearch.pipeline import (
    Library,
    PassTrace,
    PipelineConfig,
    library_from_embeddings,
    rank,
)
from src.codesearch.preprocess import categorise_query, categorise_snippet, query_view

logger = logging.getLogger(__name__)

ALL = "all"


@dataclass
class Loaded:
    library: str
    version: str
    lib: Library
    snippets: list[vstore.Snippet]
    cfg: PipelineConfig
    lineages: list[int] | None = None
    members: dict[int, list[int]] | None = None
    version_rank: np.ndarray | None = None
    versions: list[str] | None = None
    load_ms: float = 0.0


def library_config(library: str) -> PipelineConfig:
    """The pipeline for a library: the submitted APPS pipeline, restricted to the models the
    library was indexed with (codebases may use a lighter model, see versions manifest)."""
    from src.codesearch.submission import FINAL_CONFIG

    mans = vstore.list_versions(library)
    models = mans[0]["models"] if mans else [k for k, _ in FINAL_CONFIG.models]
    weights = dict(FINAL_CONFIG.models)
    chosen = tuple((m, weights.get(m, 1.0)) for m in models)
    cfg = replace(FINAL_CONFIG, models=chosen)
    if is_codebase(library):
        # Program-style categories (stdin script / function / class Solution) describe
        # competitive-programming Python; they carry no signal for a JavaScript codebase.
        cfg = replace(cfg, w_style=0.0)
    return cfg


def is_codebase(library: str) -> bool:
    mans = vstore.list_versions(library)
    return bool(mans) and mans[-1].get("source", {}).get("kind") == "codebase"


class SearchEngine:
    def __init__(self) -> None:
        self._loaded: dict[tuple[str, str], Loaded] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- catalogue
    def libraries(self) -> list[dict]:
        out = []
        for name in vstore.list_libraries():
            mans = vstore.list_versions(name)
            if not mans:
                continue
            out.append(
                {
                    "name": name,
                    "kind": mans[-1].get("source", {}).get("kind", "snippets"),
                    "description": mans[-1].get("source", {}).get("description", ""),
                    "models": mans[-1]["models"],
                    "versions": [
                        {
                            "version": m["version"],
                            "snippets": m["snippets"],
                            "built_at": m["built_at"],
                            "encoded": {k: v["encoded"] for k, v in m["encode"].items()},
                            "total_s": round(m["total_s"], 2),
                            "commit": m.get("timings", {}).get("commit", ""),
                        }
                        for m in mans
                    ],
                    "has_lineages": (vstore.VERSIONS_DIR / name / "lineages.json").exists(),
                }
            )
        return out

    # ---------------------------------------------------------------- loading
    def _get(self, library: str, version: str) -> Loaded:
        key = (library, version)
        with self._lock:
            if key not in self._loaded:
                self._loaded[key] = (
                    self._load_all(library) if version == ALL else self._load(library, version)
                )
            return self._loaded[key]

    def _load(self, library: str, version: str) -> Loaded:
        t = time.perf_counter()
        cfg = library_config(library)
        emb, snippets = {}, None
        for key, _w in cfg.models:
            snippets, emb[key] = vstore.load_version(library, version, key)
        lib = library_from_embeddings(
            [s.id for s in snippets], [s.text for s in snippets], emb, cfg
        )
        return Loaded(
            library, version, lib, snippets, cfg, load_ms=(time.perf_counter() - t) * 1000
        )

    def _load_all(self, library: str) -> Loaded:
        t = time.perf_counter()
        cfg = library_config(library)
        order = [m["version"] for m in vstore.list_versions(library)]
        snippets: list[vstore.Snippet] = []
        emb: dict[str, list[np.ndarray]] = {k: [] for k, _ in cfg.models}
        for v in order:
            for key, _w in cfg.models:
                sn, e = vstore.load_version(library, v, key)
                emb[key].append(e)
            snippets.extend(sn)
        stacked = {k: np.concatenate(v) for k, v in emb.items()}
        ids = [f"{s.id}@{s.version}" for s in snippets]
        lib = library_from_embeddings(ids, [s.text for s in snippets], stacked, cfg)
        lineages = load_or_build_lineages(library, snippets, stacked[cfg.models[0][0]])
        return Loaded(
            library,
            ALL,
            lib,
            snippets,
            cfg,
            lineages=lineages,
            members=members_by_lineage(lineages),
            version_rank=np.array([order.index(s.version) for s in snippets]),
            versions=order,
            load_ms=(time.perf_counter() - t) * 1000,
        )

    def libraries_map(self) -> dict[str, list[str]]:
        return {n: [m["version"] for m in vstore.list_versions(n)] for n in vstore.list_libraries()}

    def warm(self, library: str, version: str) -> None:
        """Load the index and the query encoders so the first live query is not slowed down."""
        from src.codesearch.pipeline import encode_queries

        loaded = self._get(library, version)
        # Realistic lengths matter: the first forward pass at a new sequence length pays a
        # one-off kernel set-up cost (~10 s measured), so warm with short, medium and long texts.
        base = "You are given n integers a_1, ..., a_n. Print the maximum sum of a subarray. "
        texts = [f"warm-up {i}: " + base * reps for i, reps in enumerate((1, 6, 14))]
        for key, _w in loaded.cfg.models:
            encode_queries(texts, key, loaded.cfg)

    # ---------------------------------------------------------------- search
    def search(self, query: str, library: str, version: str | None = None, k: int = 10) -> dict:
        t_all = time.perf_counter()
        mans = vstore.list_versions(library)
        if not mans:
            raise KeyError(f"library {library!r} is not indexed")
        labels = [m["version"] for m in mans]
        version = version or labels[-1]
        passes: list[PassTrace] = []
        t = time.perf_counter()
        loaded = self._get(library, version)
        passes.append(
            PassTrace(
                "load index",
                f"{library}@{version}: {len(loaded.snippets)} snippets "
                f"({'cached in memory' if (time.perf_counter() - t) < 0.05 else 'loaded'})",
                (time.perf_counter() - t) * 1000,
            )
        )
        hint = parse_version_hint(query, labels)
        text = hint.cleaned_query if version == ALL else query
        info = categorise_query(text)
        pool = 200 if version == ALL else max(k, 10)
        cfg = replace(loaded.cfg, top_k=max(pool, k))
        idx, sc = rank([text], loaded.lib, cfg, trace=passes)
        idx, sc = idx[0], sc[0]
        results = []
        if version == ALL:
            t = time.perf_counter()
            full = np.full(len(loaded.snippets), -np.inf, dtype=np.float32)
            full[idx] = sc
            hits = collapse(
                full,
                loaded.lineages,
                loaded.version_rank,
                hint,
                [s.version for s in loaded.snippets],
                top_k=k,
                members_of=loaded.members,
            )
            passes.append(
                PassTrace(
                    "lineage collapse",
                    f"{len(set(loaded.lineages[i] for i in idx))} lineages among top {len(idx)}"
                    f" → {len(hits)}; version choice: "
                    + (f"hint '{hint.phrase}'" if hint.kind else "best match, ties → newest"),
                    (time.perf_counter() - t) * 1000,
                )
            )
            for r, h in enumerate(hits, 1):
                res = self._result(loaded, h.index, h.score, r, query)
                res["lineage"] = {
                    "id": h.lineage,
                    "versions": [
                        {
                            "version": loaded.snippets[m].version,
                            "id": loaded.snippets[m].id,
                            "score": None if not np.isfinite(s) else round(float(s), 4),
                            "same_text": loaded.snippets[m].content_hash
                            == loaded.snippets[h.index].content_hash,
                        }
                        for m, s in h.members
                    ],
                }
                results.append(res)
        else:
            for r, (i, s) in enumerate(zip(idx[:k], sc[:k], strict=True), 1):
                results.append(self._result(loaded, int(i), float(s), r, query))
        latency = (time.perf_counter() - t_all) * 1000
        return {
            "query": query,
            "library": library,
            "version": version,
            "versions": labels,
            "library_size": len(loaded.snippets),
            "category": (
                {**info.to_dict(), "style": "code description"}
                if is_codebase(library)
                else info.to_dict()
            ),
            "preprocessing": {
                "views": [v for v, _ in loaded.cfg.query_views],
                "cleaned": query_view(info, loaded.cfg.query_views[0][0])[:600],
                "raw_chars": len(query),
                "hint": asdict(hint) if hint.kind else None,
            },
            "pipeline": loaded.cfg.to_dict(),
            "passes": [asdict(p) for p in passes],
            "results": results,
            "latency_ms": round(latency, 1),
        }

    def _result(self, loaded: Loaded, i: int, score: float, rank_: int, query: str) -> dict:
        s = loaded.snippets[i]
        out = {
            "rank": rank_,
            "id": s.id,
            "version": s.version,
            "score": round(score, 4),
            "text": s.text,
            "path": s.path,
            "symbol": s.symbol,
            "start_line": s.start_line,
            "end_line": s.end_line,
            "style": "function" if s.path else categorise_snippet(s.text).style,
        }
        if loaded.lib.literal is not None:
            out["matched_literals"] = loaded.lib.literal.matched(query, i)[:8]
        return out


def load_or_build_lineages(
    library: str, snippets: list[vstore.Snippet], emb: np.ndarray
) -> list[int]:
    """Lineage id per entry of the all-versions library (cached in lineages.json)."""
    path = vstore.VERSIONS_DIR / library / "lineages.json"
    keys = [f"{s.id}@{s.version}" for s in snippets]
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("keys") == keys:
            return saved["lineages"]
    t = time.perf_counter()
    lin = infer_lineages(
        [s.text for s in snippets],
        [s.version for s in snippets],
        emb,
        keys=[s.lineage_key for s in snippets],
    )
    path.write_text(json.dumps({"keys": keys, "lineages": lin}))
    logger.info("lineages for %s: %d in %.1f s", library, len(set(lin)), time.perf_counter() - t)
    return lin


_ENGINE: SearchEngine | None = None


def get_engine() -> SearchEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SearchEngine()
    return _ENGINE
