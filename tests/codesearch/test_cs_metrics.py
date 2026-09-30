import math

import numpy as np

from src.codesearch.metrics import evaluate_run, ndcg_at_k, run_from_scores


def test_ndcg_and_mrr_single_relevant():
    qrels = {"q1": {"d2": 1}, "q2": {"d9": 1}}
    run = {"q1": ["d1", "d2", "d3"], "q2": ["d1"]}
    m = evaluate_run(run, qrels)
    assert math.isclose(m["ndcg@10"], (1 / math.log2(3)) / 2)
    assert math.isclose(m["mrr@10"], 0.25)
    assert m["recall@1"] == 0.0 and m["recall@10"] == 0.5


def test_graded_ndcg():
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}) == 1.0
    assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}) < 1.0


def test_run_from_scores_orders():
    s = np.array([[0.1, 0.9, 0.5]])
    assert run_from_scores(s, ["q"], ["a", "b", "c"], k=3) == {"q": ["b", "c", "a"]}
