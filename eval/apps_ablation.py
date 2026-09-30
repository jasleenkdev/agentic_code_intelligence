"""Ablation table for the APPS pipeline on the tuning splits (never the test split, D-035).

    python -m eval.apps_ablation                # every variant in VARIANTS
    python -m eval.apps_ablation --only final   # variants whose name contains "final"

Each variant is a `PipelineConfig`; libraries are built once per (snippet view, models) and the
embedding cache makes repeated runs cheap. Writes results/apps_ablation.{json,md}.
Selection rule: a component stays in the final pipeline only if it raises dev-tune NDCG@10;
dev-holdout is reported to show the choice generalises.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from src import config
from src.codesearch.data import dev_splits
from src.codesearch.metrics import evaluate_run
from src.codesearch.pipeline import PipelineConfig, build_library, run_dict
from src.codesearch.submission import FINAL_CONFIG, VARIANTS

OUT_JSON = config.RESULTS_DIR / "apps_ablation.json"
OUT_MD = config.RESULTS_DIR / "apps_ablation.md"


def evaluate_variant(cfg: PipelineConfig, sets: list[str], libs: dict) -> dict:
    splits = dev_splits()
    corpus = splits["dev-tune"].corpus
    key = (
        cfg.snippet_view,
        cfg.models,
        cfg.w_literal > 0,
        cfg.w_bm25 > 0,
        cfg.w_hub > 0,
        cfg.hub_k,
    )
    if key not in libs:
        libs[key] = build_library(list(corpus), list(corpus.values()), cfg)
    lib = libs[key]
    out = {}
    for name in sets:
        s = splits[name]
        t = time.perf_counter()
        run = run_dict(s.queries, lib, cfg)
        m = evaluate_run(run, s.qrels)
        m["rank_ms_per_query"] = (time.perf_counter() - t) * 1000 / len(s.queries)
        out[name] = m
    return out


def write(rows: list[dict]) -> None:
    OUT_JSON.write_text(json.dumps(rows, indent=2))
    lines = [
        "# APPS pipeline ablations (train-derived tuning splits, D-035)",
        "",
        "Queries: dev-tune / dev-holdout = 600 + 600 stdin-style APPS **train** problems; corpus =",
        "the 5 000 train snippets. Hubness is snippet-side (query-free). The test split is not",
        "used. Each row changes one thing relative to the row it names.",
        "",
        "| Variant | What changes | dev-tune NDCG@10 | dev-tune MRR@10 | dev-tune R@10 | "
        "dev-holdout NDCG@10 | dev-holdout MRR@10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        t, h = r["sets"].get("dev-tune", {}), r["sets"].get("dev-holdout", {})
        lines.append(
            f"| {r['name']} | {r['note']} | {t.get('ndcg@10', 0):.4f} | {t.get('mrr@10', 0):.4f} | "
            f"{t.get('recall@10', 0):.3f} | {h.get('ndcg@10', 0):.4f} | {h.get('mrr@10', 0):.4f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--sets", default="dev-tune,dev-holdout")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    rows = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else []
    libs: dict = {}
    variants = list(VARIANTS) + [("final", "the submitted pipeline", FINAL_CONFIG)]
    for name, note, cfg in variants:
        if args.only and args.only not in name:
            continue
        sets = evaluate_variant(cfg, args.sets.split(","), libs)
        rows = [r for r in rows if r["name"] != name]
        rows.append({"name": name, "note": note, "config": cfg.to_dict(), "sets": sets})
        order = [n for n, _, _ in variants]
        rows.sort(key=lambda r: order.index(r["name"]) if r["name"] in order else 999)
        write(rows)
        print(name, {k: round(v["ndcg@10"], 4) for k, v in sets.items()}, flush=True)


if __name__ == "__main__":
    main()
