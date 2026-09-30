"""Multi-pass ranking pipeline shared by the MTEB run, the evaluation harness and the demo.

    pass 0  pre-process   parse + categorise + clean each query into views; normalise snippets
    pass 1  dense recall  weighted sum of cosine scores over (model × query view) → top-K
    pass 2  rerank        on the top-K only: + literal overlap + BM25 + program-style agreement
                          − hubness penalty; per-query standardised signals, fixed weights
    output  ranked snippet ids with the final fused score

A weighted sum of cosines over several models equals one cosine over the concatenation of
weighted, L2-normalised embeddings, so pass 1 is an "ensemble via concatenated embeddings".
All weights live in `PipelineConfig` and were chosen on dev-tune only (D-035).
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from src.codesearch.encoders import Encoder
from src.codesearch.lexical import BM25Matrix, LiteralMatcher
from src.codesearch.preprocess import (
    categorise_query,
    categorise_snippet,
    query_view,
    snippet_view,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Every ranking knob. `models` / `query_views` are (name, weight) pairs."""

    models: tuple[tuple[str, float], ...] = (("jina-code-0.5b", 1.0),)
    query_views: tuple[tuple[str, float], ...] = (("full", 1.0),)
    snippet_view: str = "raw"
    first_pass_k: int = 100
    w_literal: float = 0.0
    w_bm25: float = 0.0
    w_style: float = 0.0
    w_hub: float = 0.0
    hub_k: int = 10
    top_k: int = 100

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PassTrace:
    name: str
    detail: str
    ms: float


@dataclass
class Library:
    """An indexed snippet library: ids, texts, embeddings per model, lexical indexes, styles."""

    ids: list[str]
    texts: list[str]
    embeddings: dict[str, np.ndarray]
    styles: np.ndarray  # int codes, see STYLE_CODES
    literal: LiteralMatcher | None = None
    bm25: BM25Matrix | None = None
    hub: dict[str, np.ndarray] = field(default_factory=dict)  # per model: hubness per snippet
    build_stats: dict = field(default_factory=dict)


STYLE_CODES = {"stdin": 0, "function": 1, "class": 2}


def get_encoder(key: str, _cache: dict = {}) -> Encoder:  # noqa: B006 — process-wide cache
    if key not in _cache:
        _cache[key] = Encoder(key)
    return _cache[key]


def build_library(ids: list[str], texts: list[str], cfg: PipelineConfig) -> Library:
    """Encode (cache-aware) and index a snippet library for `cfg`.

    Hubness is estimated from the library alone (query-free, D-037): a snippet's mean cosine to
    its `hub_k` nearest other snippets. Reference *queries* are deliberately not used — any APPS
    query set has its answers inside the corpus, so a query-based penalty would demote known
    answers, which is the leakage D-034 rules out.
    """
    t0 = time.perf_counter()
    views = [snippet_view(t, cfg.snippet_view) for t in texts]
    emb, stats = {}, {}
    for key, _w in cfg.models:
        enc = get_encoder(key)
        emb[key] = enc.encode(views, "doc")
        stats[key] = dict(enc.last_stats)
    lib = library_from_embeddings(ids, texts, emb, cfg)
    lib.build_stats["encode"] = stats
    lib.build_stats["seconds"] = time.perf_counter() - t0
    return lib


def library_from_embeddings(
    ids: list[str], texts: list[str], emb: dict[str, np.ndarray], cfg: PipelineConfig
) -> Library:
    """Index a library whose snippet embeddings already exist (e.g. a stored version)."""
    styles = np.array([STYLE_CODES[categorise_snippet(t).style] for t in texts], dtype=np.int8)
    lib = Library(ids, texts, emb, styles)
    if cfg.w_literal:
        lib.literal = LiteralMatcher(texts)
    if cfg.w_bm25:
        lib.bm25 = BM25Matrix(texts)
    if cfg.w_hub:
        for key, _w in cfg.models:
            lib.hub[key] = snippet_hubness(emb[key], cfg.hub_k)
    return lib


def snippet_hubness(emb: np.ndarray, k: int, block: int = 2048) -> np.ndarray:
    """Mean cosine of each snippet to its k nearest other snippets."""
    out = np.zeros(len(emb), dtype=np.float32)
    k = min(k, max(len(emb) - 1, 1))
    for s in range(0, len(emb), block):
        sims = emb[s : s + block] @ emb.T
        for r in range(sims.shape[0]):
            sims[r, s + r] = -1.0
        out[s : s + block] = (-np.partition(-sims, k - 1, axis=1)[:, :k]).mean(axis=1)
    return out


def encode_queries(queries: list[str], key: str, cfg: PipelineConfig) -> np.ndarray:
    """Weighted, re-normalised sum of the query views' embeddings for one model."""
    infos = [categorise_query(q) for q in queries]
    enc = get_encoder(key)
    total = None
    for view, w in cfg.query_views:
        v = enc.encode([query_view(i, view) for i in infos], "query") * w
        total = v if total is None else total + v
    return total / np.linalg.norm(total, axis=1, keepdims=True).clip(1e-9)


def _zscore(m: np.ndarray) -> np.ndarray:
    mu = m.mean(axis=1, keepdims=True)
    sd = m.std(axis=1, keepdims=True)
    return (m - mu) / np.where(sd > 1e-9, sd, 1.0)


def dense_scores(queries: list[str], lib: Library, cfg: PipelineConfig) -> np.ndarray:
    total = np.zeros((len(queries), len(lib.ids)), dtype=np.float32)
    for key, w in cfg.models:
        s = encode_queries(queries, key, cfg) @ lib.embeddings[key].T
        if cfg.w_hub and key in lib.hub:
            s = s - cfg.w_hub * lib.hub[key][None, :]
        total += w * s
    return total


def rank(
    queries: list[str], lib: Library, cfg: PipelineConfig, trace: list[PassTrace] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices [nq, top_k], scores [nq, top_k]) into `lib`, best first."""
    t = time.perf_counter()
    infos = [categorise_query(q) for q in queries]
    if trace is not None:
        trace.append(
            PassTrace(
                "pre-process",
                f"categorised {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}; "
                f"views {[v for v, _ in cfg.query_views]}",
                (time.perf_counter() - t) * 1000,
            )
        )
    t = time.perf_counter()
    dense = dense_scores(queries, lib, cfg)
    k1 = min(cfg.first_pass_k, dense.shape[1])
    cand = np.argpartition(-dense, k1 - 1, axis=1)[:, :k1]
    if trace is not None:
        trace.append(
            PassTrace(
                "dense recall",
                f"{len(cfg.models)} model(s) × {len(cfg.query_views)} view(s) over "
                f"{len(lib.ids)} snippets → top {k1}"
                + (
                    " (query embeddings from cache)"
                    if all(get_encoder(k).last_stats.get("encoded", 1) == 0 for k, _ in cfg.models)
                    else " (query encoded on CPU)"
                ),
                (time.perf_counter() - t) * 1000,
            )
        )
    t = time.perf_counter()
    feats = _zscore(dense)
    final = np.take_along_axis(feats, cand, axis=1)
    notes = []
    if cfg.w_literal and lib.literal is not None:
        lit = lib.literal.score(queries)
        final = final + cfg.w_literal * np.take_along_axis(np.log1p(lit), cand, axis=1)
        notes.append("literal overlap")
    if cfg.w_bm25 and lib.bm25 is not None:
        bm = _zscore(lib.bm25.score(queries))
        final = final + cfg.w_bm25 * np.take_along_axis(bm, cand, axis=1)
        notes.append("BM25")
    if cfg.w_style:
        qstyle = np.array([STYLE_CODES[i.style] for i in infos])
        agree = (lib.styles[cand] == qstyle[:, None]).astype(np.float32)
        final = final + cfg.w_style * agree
        notes.append("program-style agreement")
    order = np.argsort(-final, axis=1)[:, : cfg.top_k]
    idx = np.take_along_axis(cand, order, axis=1)
    sc = np.take_along_axis(final, order, axis=1)
    if trace is not None:
        trace.append(
            PassTrace(
                "rerank",
                f"top {k1} re-scored with {', '.join(notes) or 'dense score only'}",
                (time.perf_counter() - t) * 1000,
            )
        )
    return idx, sc


def run_dict(queries: dict[str, str], lib: Library, cfg: PipelineConfig) -> dict[str, list[str]]:
    qids = list(queries)
    idx, _ = rank([queries[q] for q in qids], lib, cfg)
    return {q: [lib.ids[j] for j in idx[i]] for i, q in enumerate(qids)}
