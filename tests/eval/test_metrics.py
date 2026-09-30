from eval.metrics import aggregate, percentile, precision_at, query_metrics, recall_at

G = [
    {"file_path": "a.js", "start_line": 10, "end_line": 20},
    {"file_path": "b.js", "start_line": 5, "end_line": 5},
]


def R(path, s, e):
    return {"file_path": path, "start_line": s, "end_line": e}


def test_overlap_based_precision_recall_mrr():
    results = [R("c.js", 1, 9), R("a.js", 18, 30), R("b.js", 1, 4), R("b.js", 5, 9)]
    m = query_metrics(results, G)
    assert m["p@1"] == 0.0
    assert m["mrr"] == 0.5
    assert precision_at(results, G, 5) == 0.5  # 2 relevant of 4 returned
    assert recall_at(results, G, 2) == 0.5 and recall_at(results, G, 10) == 1.0


def test_empty_results_score_zero():
    assert query_metrics([], G) == {
        k: 0.0 for k in ("p@1", "p@5", "p@10", "r@5", "r@10", "r@20", "mrr")
    }


def test_percentile_and_aggregate():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    rows = [
        {"metrics": query_metrics([R("a.js", 10, 10)], G), "latency_ms": 10},
        {"metrics": query_metrics([], G), "latency_ms": 30},
    ]
    agg = aggregate(rows)
    assert agg["mrr"] == 0.5 and agg["latency_p50_ms"] == 20 and agg["n"] == 2
