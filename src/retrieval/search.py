"""Hybrid retrieval: BM25 + vectors fused by Reciprocal Rank Fusion, an optional structural list
(exact symbol-name matches + domain-catalog hits), and an optional CPU cross-encoder rerank.

Every ranking knob comes from `RankingConfig` (FR-6) so evaluation ablations are just configs.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from src import config
from src.config import RankingConfig
from src.indexer import store

logger = logging.getLogger(__name__)

IDENT_RE = re.compile(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*(?:\(\))?")


@dataclass
class Hit:
    chunk_id: int
    file_path: str
    start_line: int
    end_line: int
    symbol: str
    kind: str
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)
    symbol_id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def identifier_like(query: str) -> list[str]:
    """Tokens that look like code identifiers: camelCase, snake_case, dotted, or `name()`."""
    out = []
    for tok in IDENT_RE.findall(query):
        bare = tok.removesuffix("()")
        if (
            tok.endswith("()")
            or "_" in bare
            or "." in bare
            or re.search(r"[a-z][A-Z]", bare)
            or (
                bare[:1].isupper()
                and re.search(r"[a-z]", bare)
                and len(bare) > 3
                and re.search(r"[A-Z].*[A-Z]", bare)
            )
        ):
            out.append(bare.split(".")[-1])
    return list(dict.fromkeys(out))


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    from src.retrieval.embeddings import detach_from_mmap

    try:
        model = CrossEncoder(config.RERANK_MODEL, device="cpu", local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached yet → download once
        model = CrossEncoder(config.RERANK_MODEL, device="cpu")
    detach_from_mmap(model.model)
    return model


class Retriever:
    """Thread-safe-enough read-only retriever (SQLite opened with check_same_thread=False)."""

    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self.conn = conn or store.connect(config.DB_PATH, readonly=True)
        self._bm25 = None
        self._vectors = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ backends
    @property
    def bm25(self):
        if self._bm25 is None:
            from src.retrieval.bm25 import load_bm25

            self._bm25 = load_bm25()
        return self._bm25

    @property
    def vectors(self):
        if self._vectors is None:
            from src.retrieval.embeddings import VectorIndex

            self._vectors = VectorIndex()
        return self._vectors

    def vectors_available(self) -> bool:
        return config.VECTORS_PATH.exists()

    # ------------------------------------------------------------------ helpers
    def _meta(self, chunk_ids: list[int]) -> dict[int, Hit]:
        if not chunk_ids:
            return {}
        out: dict[int, Hit] = {}
        for i in range(0, len(chunk_ids), 500):
            part = chunk_ids[i : i + 500]
            q = (
                "SELECT c.id, c.start_line, c.end_line, c.symbol_id, f.path, s.qualname, s.kind"
                " FROM chunks c JOIN files f ON f.id = c.file_id LEFT JOIN symbols s ON"
                f" s.id = c.symbol_id WHERE c.id IN ({','.join('?' * len(part))})"
            )
            for r in self.conn.execute(q, part):
                out[r["id"]] = Hit(
                    chunk_id=r["id"],
                    file_path=r["path"],
                    start_line=r["start_line"],
                    end_line=r["end_line"],
                    symbol=r["qualname"] or "<module>",
                    kind=r["kind"] or "module",
                    symbol_id=r["symbol_id"],
                )
        return out

    def chunk_for_line(self, file_path: str, line: int) -> int | None:
        row = self.conn.execute(
            "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path = ? AND"
            " c.start_line <= ? AND c.end_line >= ? ORDER BY (c.end_line - c.start_line) LIMIT 1",
            (file_path, line, line),
        ).fetchone()
        return row["id"] if row else None

    def chunk_text(self, chunk_id: int) -> tuple[str, str]:
        r = self.conn.execute(
            "SELECT header, text FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return (r["header"], r["text"]) if r else ("", "")

    def _filter(self, ranked: list[tuple[int, float]], path_prefix: str | None) -> list[int]:
        ids = [cid for cid, _ in ranked]
        if not path_prefix:
            return ids
        meta = self._meta(ids)
        return [cid for cid in ids if cid in meta and meta[cid].file_path.startswith(path_prefix)]

    # ------------------------------------------------------------------ retrievers
    def bm25_scored(
        self, query: str, n: int, path_prefix: str | None = None
    ) -> list[tuple[int, float]]:
        with self._lock:
            ranked = self.bm25.search(query, n * (8 if path_prefix else 1))
        keep = set(self._filter(ranked, path_prefix))
        return [(c, s) for c, s in ranked if c in keep][:n]

    def vector_scored(
        self, query: str, n: int, path_prefix: str | None = None
    ) -> list[tuple[int, float]]:
        with self._lock:
            ranked = self.vectors.search(query, n * (8 if path_prefix else 1))
        keep = set(self._filter(ranked, path_prefix))
        return [(c, s) for c, s in ranked if c in keep][:n]

    def bm25_ranked(self, query: str, n: int, path_prefix: str | None = None) -> list[int]:
        return [c for c, _ in self.bm25_scored(query, n, path_prefix)]

    def vector_ranked(self, query: str, n: int, path_prefix: str | None = None) -> list[int]:
        return [c for c, _ in self.vector_scored(query, n, path_prefix)]

    def structural_ranked(self, query: str, n: int, path_prefix: str | None = None) -> list[int]:
        """Graph-derived candidates: for each unambiguous identifier in the query (≤ 3
        definitions), its definition chunk, then the chunks of its call sites (callers) and of
        its resolved callees; then chunks of the top domain-catalog hit."""
        from src.catalog.search import search_catalog

        ids: list[int] = []
        neighbours: list[int] = []
        for name in identifier_like(query):
            syms = self.conn.execute(
                "SELECT id FROM symbols WHERE name = ? AND kind != 'module'", (name,)
            ).fetchall()
            if not syms or len(syms) > 3:
                continue
            sids = [r["id"] for r in syms]
            marks = ",".join("?" * len(sids))
            for r in self.conn.execute(
                f"SELECT id FROM chunks WHERE symbol_id IN ({marks}) ORDER BY start_line", sids
            ):
                ids.append(r["id"])
            for r in self.conn.execute(
                "SELECT f.path, c.line FROM calls c JOIN files f ON f.id = c.file_id WHERE"
                f" c.callee_symbol_id IN ({marks}) ORDER BY f.path, c.line LIMIT 15",
                sids,
            ):
                cid = self.chunk_for_line(r["path"], r["line"])
                if cid is not None:
                    neighbours.append(cid)
            for r in self.conn.execute(
                "SELECT DISTINCT ch.id FROM calls c JOIN chunks ch ON ch.symbol_id ="
                f" c.callee_symbol_id WHERE c.caller_id IN ({marks}) AND c.resolution IN"
                " ('static','heuristic') LIMIT 15",
                sids,
            ):
                neighbours.append(r["id"])
        ids.extend(neighbours)
        hits = [h for h in search_catalog(self.conn, query, limit=2) if h.score >= 1.0]
        for hit in hits[:1]:
            for row in (hit.definitions + hit.usages)[:15]:
                cid = self.chunk_for_line(row["file_path"], row["line"])
                if cid is not None:
                    ids.append(cid)
        ids = list(dict.fromkeys(ids))
        return self._filter([(i, 0.0) for i in ids], path_prefix)[:n]

    # ------------------------------------------------------------------ fusion
    def search(
        self,
        query: str,
        k: int = 10,
        cfg: RankingConfig = config.RANKING,
        path_prefix: str | None = None,
    ) -> list[Hit]:
        n = cfg.candidates_per_retriever
        scored: dict[str, list[tuple[int, float]]] = {}
        if cfg.use_bm25:
            scored["bm25"] = self.bm25_scored(query, n, path_prefix)
        if cfg.use_vectors and self.vectors_available():
            try:
                scored["vector"] = self.vector_scored(query, n, path_prefix)
            except Exception as exc:  # noqa: BLE001 — degrade to lexical-only
                logger.warning("vector search unavailable: %s", exc)
        if cfg.use_graph_boost:
            scored["structural"] = [(c, 0.0) for c in self.structural_ranked(query, n, path_prefix)]
        w_vector, w_bm25 = cfg.w_vector, cfg.w_bm25
        has_ident = bool(identifier_like(query))
        if cfg.ident_w_vector is not None and has_ident:
            w_vector = cfg.ident_w_vector
        if cfg.nl_w_bm25 is not None and not has_ident:
            w_bm25 = cfg.nl_w_bm25
        weights = {"bm25": w_bm25, "vector": w_vector, "structural": 1.0 + cfg.graph_boost * 10}
        fused: dict[int, float] = {}
        ranks: dict[int, dict[str, int]] = {}
        # Normalisation ranges + weights of this query, so score_chunks() can place chunks that
        # were not retrieved (graph neighbours) on the same scale (D-028).
        self.last_norm = {"query": query, "weights": weights, "ranges": {}}
        for name, items in scored.items():
            w = weights[name]
            if cfg.fusion == "score" and name != "structural" and items:
                vals = [sc for _, sc in items]
                lo, hi = min(vals), max(vals)
                self.last_norm["ranges"][name] = (lo, hi)
                for rank, (cid, sc) in enumerate(items, 1):
                    norm = (sc - lo) / (hi - lo) if hi > lo else 1.0
                    fused[cid] = fused.get(cid, 0.0) + w * norm
                    ranks.setdefault(cid, {})[name] = rank
            else:
                # RRF (also used for the rank-only structural list in score mode, rescaled so
                # rank 1 contributes w like a top normalised score does).
                scale = (cfg.rrf_k + 1) if cfg.fusion == "score" else 1.0
                for rank, (cid, _) in enumerate(items, 1):
                    fused[cid] = fused.get(cid, 0.0) + scale * w / (cfg.rrf_k + rank)
                    ranks.setdefault(cid, {})[name] = rank
        order = sorted(fused, key=lambda c: -fused[c])
        if cfg.use_rerank and order:
            order = self._rerank(query, order, cfg.rerank_top_n)
        top = order[: max(k, 1)]
        meta = self._meta(top)
        hits = []
        for cid in top:
            if cid not in meta:
                continue
            h = meta[cid]
            h.score = round(fused[cid], 5)
            h.ranks = ranks.get(cid, {})
            hits.append(h)
        return hits

    def score_chunks(self, query: str, chunk_ids: list[int]) -> dict[int, float]:
        """Fused score of arbitrary chunks for `query`, using the normalisation ranges and
        weights of the last `search()` for the same query (scores clipped to [0, 1] per list)."""
        import numpy as np

        norm = getattr(self, "last_norm", None)
        if not chunk_ids or not norm or norm["query"] != query:
            return {}
        out = {c: 0.0 for c in chunk_ids}
        ranges, weights = norm["ranges"], norm["weights"]
        if "bm25" in ranges:
            from src.retrieval.bm25 import tokenize_query

            idx = self.bm25.index_of
            pairs = [(c, idx[c]) for c in chunk_ids if c in idx]
            toks = tokenize_query(query)
            if pairs and toks:
                raw = self.bm25.bm25.get_batch_scores(toks, [i for _, i in pairs])
                lo, hi = ranges["bm25"]
                for (c, _), sc in zip(pairs, raw, strict=True):
                    out[c] += weights["bm25"] * min(
                        1.0, max(0.0, (sc - lo) / (hi - lo) if hi > lo else 0.0)
                    )
        if "vector" in ranges and self.vectors_available():
            from src.retrieval.embeddings import embed_query

            v = self.vectors
            pairs = [(c, v.index_of[c]) for c in chunk_ids if c in v.index_of]
            if pairs:
                sims = v.matrix[np.array([i for _, i in pairs])] @ embed_query(query)
                lo, hi = ranges["vector"]
                for (c, _), sc in zip(pairs, sims, strict=True):
                    out[c] += weights["vector"] * min(
                        1.0, max(0.0, (float(sc) - lo) / (hi - lo) if hi > lo else 0.0)
                    )
        return out

    def _rerank(self, query: str, order: list[int], top_n: int) -> list[int]:
        head = order[:top_n]
        pairs = []
        for cid in head:
            header, text = self.chunk_text(cid)
            pairs.append((query, (header + text)[:1500]))
        scores = _get_reranker().predict(pairs, batch_size=16)
        reranked = [cid for _, cid in sorted(zip(scores, head, strict=True), key=lambda t: -t[0])]
        return reranked + order[top_n:]


_DEFAULT: Retriever | None = None


def get_retriever() -> Retriever:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Retriever()
    return _DEFAULT
