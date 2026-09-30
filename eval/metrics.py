"""Retrieval metrics over location ranges.

A returned location (file, start, end) is *relevant* if it is in the same file as a gold location
and the line ranges overlap. Definitions (documented in docs/approach.md §A.5 and D-020):
- precision@k : relevant results among the returned top-k ÷ min(k, number returned)
                (0 if nothing is returned). Snippet-returning systems often return fewer than k
                results; dividing by k would reward padding.
- recall@k    : gold locations overlapped by at least one of the top-k ÷ number of gold locations.
- MRR         : 1 / rank of the first relevant result (0 if none), averaged over queries.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence


def overlaps(a: dict, b: dict) -> bool:
    return (
        a["file_path"] == b["file_path"]
        and a["start_line"] <= b["end_line"]
        and b["start_line"] <= a["end_line"]
    )


def relevance(results: Sequence[dict], gold: Sequence[dict]) -> list[bool]:
    return [any(overlaps(r, g) for g in gold) for r in results]


def precision_at(results: Sequence[dict], gold: Sequence[dict], k: int) -> float:
    top = list(results)[:k]
    if not top:
        return 0.0
    return sum(relevance(top, gold)) / len(top)


def recall_at(results: Sequence[dict], gold: Sequence[dict], k: int) -> float:
    if not gold:
        return 0.0
    top = list(results)[:k]
    covered = sum(1 for g in gold if any(overlaps(r, g) for r in top))
    return covered / len(gold)


def reciprocal_rank(results: Sequence[dict], gold: Sequence[dict]) -> float:
    for i, rel in enumerate(relevance(results, gold), 1):
        if rel:
            return 1.0 / i
    return 0.0


def query_metrics(results: Sequence[dict], gold: Sequence[dict]) -> dict:
    return {
        "p@1": precision_at(results, gold, 1),
        "p@5": precision_at(results, gold, 5),
        "p@10": precision_at(results, gold, 10),
        "r@5": recall_at(results, gold, 5),
        "r@10": recall_at(results, gold, 10),
        "r@20": recall_at(results, gold, 20),
        "mrr": reciprocal_rank(results, gold),
    }


def percentile(values: Iterable[float], pct: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100
    lo, hi = int(pos), min(int(pos) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def aggregate(rows: Sequence[dict]) -> dict:
    """Mean of per-query metrics plus latency percentiles."""
    if not rows:
        return {}
    keys = ["p@1", "p@5", "p@10", "r@5", "r@10", "r@20", "mrr"]
    out = {k: round(statistics.fmean(r["metrics"][k] for r in rows), 3) for k in keys}
    lat = [r["latency_ms"] for r in rows]
    out["latency_p50_ms"] = round(percentile(lat, 50), 1)
    out["latency_p95_ms"] = round(percentile(lat, 95), 1)
    out["n"] = len(rows)
    return out
