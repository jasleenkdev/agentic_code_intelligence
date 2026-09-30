"""Coordinate-ascent sweep of the pipeline's knobs on dev-tune (never the test split, D-035).

    python -m eval.apps_sweep --models jina-code-0.5b[:1.0][,qwen3-emb-0.6b:0.5]

Starts from the given models with default settings and, knob by knob, keeps the value with the
best dev-tune NDCG@10 (ties → the simpler / smaller value). dev-holdout is evaluated for the
chosen setting only. Writes results/apps_sweep.{json,md}. All embeddings come from the cache
after the first pass, so a full sweep takes minutes.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace

from src import config
from src.codesearch.data import dev_splits
from src.codesearch.metrics import evaluate_run
from src.codesearch.pipeline import PipelineConfig, build_library, run_dict

OUT_JSON = config.RESULTS_DIR / "apps_sweep.json"
OUT_MD = config.RESULTS_DIR / "apps_sweep.md"

GRID: list[tuple[str, list]] = [
    (
        "query_views",
        [
            (("full", 1.0),),
            (("raw", 1.0),),
            (("no_examples", 1.0),),
            (("full", 1.0), ("no_examples", 1.0)),
            (("full", 1.0), ("statement", 0.5)),
        ],
    ),
    ("snippet_view", ["raw", "normalised"]),
    ("w_style", [0.0, 0.25, 0.5, 1.0]),
    ("w_hub", [0.0, 0.25, 0.5, 1.0]),
    ("w_literal", [0.0, 0.05, 0.1, 0.2]),
    ("w_bm25", [0.0, 0.05, 0.1]),
]


def parse_models(spec: str) -> tuple[tuple[str, float], ...]:
    out = []
    for part in spec.split(","):
        key, _, w = part.partition(":")
        out.append((key, float(w or 1.0)))
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--weights", default="", help="second-model weights to try, e.g. 0.3|0.5|1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    splits = dev_splits()
    tune, hold = splits["dev-tune"], splits["dev-holdout"]
    corpus = tune.corpus
    libs: dict = {}

    def score(cfg: PipelineConfig, s=tune) -> dict:
        key = (cfg.snippet_view, cfg.models)
        if key not in libs:
            full = replace(cfg, w_literal=1.0, w_bm25=1.0, w_hub=1.0)  # build every index once
            libs[key] = build_library(list(corpus), list(corpus.values()), full)
        return evaluate_run(run_dict(s.queries, libs[key], cfg), s.qrels)

    grid = list(GRID)
    models = parse_models(args.models)
    if len(models) == 2 and args.weights:  # ensemble weight of the second model
        combos = [(models[0], (models[1][0], float(x))) for x in args.weights.split("|")]
        grid.insert(0, ("models", [(models[0],), *combos]))
    cfg = PipelineConfig(models=models)
    log = []
    best = score(cfg)
    log.append({"knob": "start", "value": repr(cfg.models), "ndcg@10": best["ndcg@10"]})
    for knob, values in grid:
        results = []
        for v in values:
            c = replace(cfg, **{knob: v})
            m = score(c)
            results.append((m["ndcg@10"], v, m))
            log.append(
                {"knob": knob, "value": repr(v), "ndcg@10": m["ndcg@10"], "mrr@10": m["mrr@10"]}
            )
            print(f"{knob}={v!r}: ndcg@10 {m['ndcg@10']:.4f} mrr@10 {m['mrr@10']:.4f}", flush=True)
        top = max(r[0] for r in results)
        chosen = next(r for r in results if r[0] >= top - 1e-9)  # first = simplest on ties
        cfg = replace(cfg, **{knob: chosen[1]})
        best = chosen[2]
    held = score(cfg, hold)
    out = {"chosen": cfg.to_dict(), "dev-tune": best, "dev-holdout": held, "log": log}
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    lines = [
        "# Pipeline sweep (coordinate ascent on dev-tune; dev-holdout only for the chosen setting)",
        "",
        f"Chosen: `{cfg.to_dict()}`",
        "",
        f"dev-tune NDCG@10 **{best['ndcg@10']:.4f}**, MRR@10 {best['mrr@10']:.4f}; "
        f"dev-holdout NDCG@10 **{held['ndcg@10']:.4f}**, MRR@10 {held['mrr@10']:.4f}",
        "",
        "| Knob | Value | dev-tune NDCG@10 | dev-tune MRR@10 |",
        "|---|---|---|---|",
    ]
    for r in log:
        lines.append(
            f"| {r['knob']} | `{r['value']}` | {r['ndcg@10']:.4f} | {r.get('mrr@10', 0):.4f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(
        json.dumps({"chosen": cfg.to_dict(), "tune": best["ndcg@10"], "holdout": held["ndcg@10"]})
    )


if __name__ == "__main__":
    main()
