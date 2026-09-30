"""Benchmark embedding models on the APPS tuning splits (never the test split, D-035).

    python -m eval.apps_bench --models bge-small,coderankembed [--sets dev-tune,dev-holdout]

Writes results/apps_model_benchmark.{json,md}: NDCG@10, MRR@10, Recall@k and encoding
throughput per model. Embeddings land in the content-hash cache, so ablations reuse them.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from src import config
from src.codesearch.data import dev_splits
from src.codesearch.encoders import PROFILES, Encoder
from src.codesearch.metrics import evaluate_run, run_from_scores

OUT_JSON = config.RESULTS_DIR / "apps_model_benchmark.json"
OUT_MD = config.RESULTS_DIR / "apps_model_benchmark.md"


def bench_model(key: str, sets: list[str], max_seq: int | None = None) -> dict:
    splits = dev_splits()
    corpus = splits["dev-tune"].corpus
    dids = list(corpus)
    enc = Encoder(key, max_seq=max_seq)
    t0 = time.perf_counter()
    D = enc.encode([corpus[d] for d in dids], "doc")
    doc_stats = dict(enc.last_stats)
    row = {
        "model": key,
        "model_id": enc.profile.model_id,
        "params_m": enc.profile.params_m,
        "licence": enc.profile.licence,
        "max_seq": enc.max_seq,
        "docs": len(dids),
        "doc_encode_s": round(doc_stats["seconds"], 1),
        "doc_encoded_now": doc_stats["encoded"],
        "sets": {},
    }
    for name in sets:
        s = splits[name]
        qids = list(s.queries)
        Q = enc.encode([s.queries[q] for q in qids], "query")
        qs = dict(enc.last_stats)
        run = run_from_scores(Q @ D.T, qids, dids, k=100)
        m = evaluate_run(run, s.qrels)
        m["query_encode_s"] = round(qs["seconds"], 1)
        m["query_encoded_now"] = qs["encoded"]
        row["sets"][name] = m
    row["total_s"] = round(time.perf_counter() - t0, 1)
    return row


# Published APPS **test** NDCG@10 (for comparison only; see docs/approach.md §2.2 for sources).
PUBLISHED_TEST = {
    "bge-small": 0.056,
    "gte-modernbert": 0.564,
    "qwen3-emb-0.6b": 0.752,
    "jina-code-0.5b": 0.842,
    "sfr-code-400m": 0.486,
}
# Models whose authors report training on the AppsRetrieval train split: their dev scores are
# contaminated (the dev sets are built from train problems).
TRAINED_ON_APPS_TRAIN = {"jina-code-0.5b"}


def write(rows: list[dict]) -> None:
    OUT_JSON.write_text(json.dumps(rows, indent=2))
    lines = [
        "# APPS embedding-model benchmark (train-derived tuning splits)",
        "",
        "Corpus = the 5 000 APPS **train** snippets; queries = dev-tune / dev-holdout (600",
        "stdin-style train problems each, seed 13, D-035). The test split is not used here. CPU",
        "only, Apple M1 Pro. Encode seconds are wall-clock for texts not already in the embedding",
        "cache (0 = everything was cached from an earlier run).",
        "",
        "¹ jina-code-embeddings was trained on the AppsRetrieval **train** split (per its paper's",
        "training-data list), so its dev scores are contaminated (0.96 vs 0.84 published on test);",
        "for every other model the dev score tracks the published test score closely (D-040).",
        "",
        "| Model | Params | Max seq | dev-tune NDCG@10 | dev-tune MRR@10 | R@100 | "
        "dev-holdout NDCG@10 | dev-holdout MRR@10 | Published test NDCG@10 | "
        "Doc encode s (5k) | Query encode s (1.2k) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: -r["sets"].get("dev-tune", {}).get("ndcg@10", 0)):
        t = r["sets"].get("dev-tune", {})
        h = r["sets"].get("dev-holdout", {})
        qs = sum(v.get("query_encode_s", 0) for v in r["sets"].values())
        mark = "¹" if r["model"] in TRAINED_ON_APPS_TRAIN else ""
        pub = PUBLISHED_TEST.get(r["model"])
        lines.append(
            f"| {r['model']}{mark} (`{r['model_id']}`) | {r['params_m']}M | {r['max_seq']} | "
            f"{t.get('ndcg@10', 0):.4f} | {t.get('mrr@10', 0):.4f} | "
            f"{t.get('recall@100', 0):.3f} | {h.get('ndcg@10', 0):.4f} | "
            f"{h.get('mrr@10', 0):.4f} | {pub if pub is not None else '—'} | "
            f"{r['doc_encode_s']} | {qs:.0f} |"
        )
    lines += [
        "",
        "Not benchmarked: `jina-embeddings-v2-base-code` (its remote code fails under",
        "transformers 5: removed `find_pruneable_heads_and_indices`, missing `is_decoder` config",
        "defaults), `SFR-Embedding-Code-400M_R` (stopped to save CPU time; published test 0.486,",
        "far below the leaders).",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(PROFILES))
    ap.add_argument("--sets", default="dev-tune,dev-holdout")
    ap.add_argument("--max-seq", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    rows = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else []
    for key in args.models.split(","):
        try:
            row = bench_model(key, args.sets.split(","), args.max_seq)
        except Exception as e:  # one broken model must not stop the sweep
            logging.exception("model %s failed: %s", key, e)
            continue
        rows = [r for r in rows if not (r["model"] == key and r["max_seq"] == row["max_seq"])]
        rows.append(row)
        write(rows)
        print(
            json.dumps({k: row[k] for k in ("model", "max_seq", "doc_encode_s")}),
            {n: round(m["ndcg@10"], 4) for n, m in row["sets"].items()},
            flush=True,
        )


if __name__ == "__main__":
    main()
