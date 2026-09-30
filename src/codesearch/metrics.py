"""Ranking metrics computed the way MTEB reports them (pytrec_eval semantics, binary qrels)."""

from __future__ import annotations

import math

import numpy as np


def rank_of_relevant(ranked: list[str], rel: dict[str, int]) -> int | None:
    """1-based rank of the first relevant document, or None if absent."""
    for i, d in enumerate(ranked, 1):
        if rel.get(d, 0) > 0:
            return i
    return None


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1))
    ideal = sorted((v for v in rel.values() if v > 0), reverse=True)[:k]
    idcg = sum((2**v - 1) / math.log2(i + 1) for i, v in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def evaluate_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """NDCG@10, MRR@10 (MTEB's `mrr_at_10`), MRR@100, Recall@{1,10,100} over the qrels queries."""
    nd, mrr10, mrr100, r1, r10, r100 = [], [], [], [], [], []
    for q, rel in qrels.items():
        ranked = run.get(q, [])
        r = rank_of_relevant(ranked, rel)
        nd.append(ndcg_at_k(ranked, rel, 10))
        mrr10.append(1 / r if r and r <= 10 else 0.0)
        mrr100.append(1 / r if r and r <= 100 else 0.0)
        r1.append(1.0 if r == 1 else 0.0)
        r10.append(1.0 if r and r <= 10 else 0.0)
        r100.append(1.0 if r and r <= 100 else 0.0)
    return {
        "ndcg@10": float(np.mean(nd)),
        "mrr@10": float(np.mean(mrr10)),
        "mrr@100": float(np.mean(mrr100)),
        "recall@1": float(np.mean(r1)),
        "recall@10": float(np.mean(r10)),
        "recall@100": float(np.mean(r100)),
        "n": len(qrels),
    }


def run_from_scores(
    scores: np.ndarray, qids: list[str], dids: list[str], k: int = 100
) -> dict[str, list[str]]:
    """Top-k document ids per query from a [n_queries, n_docs] score matrix."""
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    out = {}
    for i, q in enumerate(qids):
        idx = top[i][np.argsort(-scores[i, top[i]])]
        out[q] = [dids[j] for j in idx]
    return out
