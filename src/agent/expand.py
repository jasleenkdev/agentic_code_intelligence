"""Grounded refinement round 1: graph/catalog expansion from the best partial hits (D-028).

`read_neighbours` looks up the immediate call-graph neighbours (callers' call sites, resolved
callees' definitions) of the top candidates. `expand_and_rescore` adds those neighbours and the
top domain-catalog entry's locations to the candidate pool and re-ranks with one-hop relevance
propagation:

    score'(c) = own(c) + λ · Σ_{seed s adjacent to c} score(s)

where `own(c)` is the chunk's fused retrieval score for the query (neighbours that were not
retrieved are scored on the same scale by `Retriever.score_chunks`) and seeds are the top-k hits.
Everything is deterministic; no LLM.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

RESOLVED = ("static", "heuristic")


@dataclass(frozen=True)
class ExpandConfig:
    enabled: bool = True
    seeds: int = 5  # top hits whose neighbours are added
    lam: float = 0.25  # propagation weight λ
    per_seed: int = 8  # max neighbours per seed and relation
    use_catalog: bool = True


def read_neighbours(retriever, hits: list, seeds: int, per_seed: int = 8) -> dict[int, list]:
    """chunk_id of each seed → [(neighbour_chunk_id, relation, label)], relation ∈ caller/callee."""
    conn: sqlite3.Connection = retriever.conn
    out: dict[int, list] = {}
    for h in hits[:seeds]:
        sid = h.symbol_id
        found: list[tuple[int, str, str]] = []
        if sid is None or h.kind == "module":
            out[h.chunk_id] = found
            continue
        for r in conn.execute(
            "SELECT f.path, c.line, s.qualname FROM calls c JOIN files f ON f.id = c.file_id"
            " JOIN symbols s ON s.id = c.caller_id WHERE c.callee_symbol_id = ? AND"
            " c.resolution IN ('static','heuristic') ORDER BY f.path, c.line LIMIT ?",
            (sid, per_seed),
        ):
            cid = retriever.chunk_for_line(r["path"], r["line"])
            if cid is not None and cid != h.chunk_id:
                found.append((cid, "caller", r["qualname"]))
        for r in conn.execute(
            "SELECT DISTINCT ch.id, t.qualname FROM calls c JOIN chunks ch ON ch.symbol_id ="
            " c.callee_symbol_id JOIN symbols t ON t.id = c.callee_symbol_id WHERE"
            " c.caller_id = ? AND c.resolution IN ('static','heuristic') LIMIT ?",
            (sid, per_seed),
        ):
            if r["id"] != h.chunk_id:
                found.append((r["id"], "callee", r["qualname"]))
        out[h.chunk_id] = list(dict.fromkeys(found))
    return out


def catalog_chunks(retriever, query: str, limit: int = 6) -> tuple[list[int], str | None]:
    """Chunks holding the definition/usages of the top catalog entry for the query (score ≥ 1)."""
    from src.catalog.search import search_catalog

    hits = [h for h in search_catalog(retriever.conn, query, limit=1) if h.score >= 1.0]
    if not hits:
        return [], None
    ids = []
    for row in (hits[0].definitions + hits[0].usages)[:limit]:
        cid = retriever.chunk_for_line(row["file_path"], row["line"])
        if cid is not None:
            ids.append(cid)
    return list(dict.fromkeys(ids)), f"{hits[0].kind} {hits[0].name}"


def expand_and_rescore(
    retriever, query: str, hits: list, cfg: ExpandConfig, k: int = 20
) -> tuple[list, dict]:
    """Return (re-ranked hits, info). `hits` must come from `retriever.search(query, …)`."""
    if not hits:
        return hits, {"added": 0}
    top = max(h.score for h in hits) or 1.0
    base = {h.chunk_id: h.score / top for h in hits}
    neigh = read_neighbours(retriever, hits, cfg.seeds, cfg.per_seed)
    support: dict[int, float] = defaultdict(float)
    edges = 0
    for seed_cid, lst in neigh.items():
        for cid, _rel, _label in lst:
            support[cid] += cfg.lam * base.get(seed_cid, 0.0)
            edges += 1
    cat_ids, cat_label = catalog_chunks(retriever, query) if cfg.use_catalog else ([], None)
    for cid in cat_ids:
        support[cid] += cfg.lam  # catalog entries are exact domain matches: full seed weight
    new_ids = [c for c in support if c not in base]
    own = dict(base)
    if new_ids:
        raw = retriever.score_chunks(query, new_ids)
        for c in new_ids:
            own[c] = raw.get(c, 0.0) / top
    final = {c: own.get(c, 0.0) + support.get(c, 0.0) for c in set(own) | set(support)}
    order = sorted(final, key=lambda c: -final[c])[:k]
    by_id = {h.chunk_id: h for h in hits}
    meta = retriever._meta([c for c in order if c not in by_id])
    out = []
    for c in order:
        h = by_id.get(c) or meta.get(c)
        if h is None:
            continue
        h.score = round(final[c], 5)
        if c not in by_id:
            h.ranks = {"expansion": 1}
        out.append(h)
    info = {
        "seeds": min(cfg.seeds, len(hits)),
        "edges": edges,
        "added": len(new_ids),
        "catalog": cat_label,
        "moved_to_top": out[0].chunk_id != hits[0].chunk_id,
    }
    return out, info
