"""Baselines on the APPS **test** split — reporting only, run after the pipeline was frozen.

Nothing here feeds back into any choice (D-035): the submitted pipeline was fixed on dev-tune
before this script was first run. It scores simple single-model baselines (raw query and snippet
text, cosine only) through the same MTEB-equivalent metric code, and lists published numbers.

    python -m eval.apps_test_baselines --models bge-small,jina-code-0.5b
Writes results/apps_test_baselines.{json,md} (read by the Metrics page and the README).
"""

from __future__ import annotations

import argparse
import json
import logging

from src import config
from src.codesearch.data import load_apps
from src.codesearch.metrics import evaluate_run
from src.codesearch.pipeline import PipelineConfig, build_library, run_dict

OUT_JSON = config.RESULTS_DIR / "apps_test_baselines.json"
OUT_MD = config.RESULTS_DIR / "apps_test_baselines.md"

PUBLISHED = [
    ("bge-small-en-v1.5 (33M)", 0.056, "published: Granite-R2 paper, Table 11"),
    ("e5-base-v2", 0.115, "published: CodeXEmbed paper, Table 1"),
    ("OpenAI text-embedding-ada-002 (API)", 0.087, "published: CodeXEmbed paper, Table 1"),
    ("Voyage-code-002 (API)", 0.265, "published: CodeXEmbed paper, Table 1"),
    ("SFR-Embedding-Code-400M", 0.486, "published: CodeXEmbed paper, Table 1"),
    ("CodeSage-large-v2 (1.3B)", 0.505, "published: CodeXEmbed paper, Table 1"),
    ("gte-modernbert-base (149M)", 0.564, "published: Granite-R2 paper, Table 11"),
    ("Qwen3-Embedding-0.6B", 0.752, "published: jina-code-embeddings paper, Table 2"),
    ("jina-code-embeddings-0.5b", 0.842, "published: jina-code-embeddings paper, Table 2"),
    ("jina-code-embeddings-1.5b", 0.866, "published: jina-code-embeddings paper, Table 2"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="bge-small")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    test = load_apps("test")
    rows = [
        {"system": name, "ndcg_at_10": nd, "mrr_at_10": None, "source": src}
        for name, nd, src in PUBLISHED
    ]
    for key in args.models.split(","):
        cfg = PipelineConfig(models=((key, 1.0),), query_views=(("raw", 1.0),))
        lib = build_library(list(test.corpus), list(test.corpus.values()), cfg)
        m = evaluate_run(run_dict(test.queries, lib, cfg), test.qrels)
        rows.append(
            {
                "system": f"{key}, raw text, cosine only",
                "ndcg_at_10": round(m["ndcg@10"], 5),
                "mrr_at_10": round(m["mrr@10"], 5),
                "source": "ours: measured on this machine (single model baseline)",
            }
        )
        print(key, m, flush=True)
    official = config.PROJECT_ROOT / "release" / "appsretrieval_results.json"
    if official.exists():
        s = json.loads(official.read_text())["scores"]["test"][0]
        rows.append(
            {
                "system": "Submitted pipeline (PrePostPipelineEncoder)",
                "ndcg_at_10": s["ndcg_at_10"],
                "mrr_at_10": s["mrr_at_10"],
                "source": "ours: official mteb.evaluate run (release/appsretrieval_results.json)",
            }
        )
    rows.sort(key=lambda r: r["ndcg_at_10"])
    OUT_JSON.write_text(json.dumps(rows, indent=2))
    lines = [
        "# APPS test split: submitted pipeline vs baselines (reporting only)",
        "",
        "Measured rows use the same metric code as MTEB (NDCG@10, MRR@10) on the 3 765 test",
        "queries against the 8 765-snippet corpus; they were run after the pipeline was frozen.",
        "",
        "| System | NDCG@10 | MRR@10 | Source |",
        "|---|---|---|---|",
    ]
    for r in rows:
        mrr = "—" if r["mrr_at_10"] is None else f"{r['mrr_at_10']:.4f}"
        lines.append(f"| {r['system']} | {r['ndcg_at_10']:.4f} | {mrr} | {r['source']} |")
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
