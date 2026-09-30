"""Sweep BM25/vector fusion settings; report MRR / R@10 / P@1 per query set.

Tuning uses the *original* queries only; the paraphrased set is held out and reported for
validation (D-024). Usage:
  python -m eval.sweep_fusion [--tag bge-small-plain]   → results/fusion_sweep_<tag>.{json,md}
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

from eval.metrics import query_metrics
from src import config
from src.config import RankingConfig

GOLD = Path(__file__).parent / "gold.json"
PARA = Path(__file__).parent / "gold_paraphrased.json"

BASE = RankingConfig()
VARIANTS: dict[str, RankingConfig] = {
    "bm25": replace(BASE, use_vectors=False),
    "vector": replace(BASE, use_bm25=False),
    "rrf w_vec=1.0": BASE,
    "rrf w_vec=0.75": replace(BASE, w_vector=0.75),
    "rrf w_vec=0.5": replace(BASE, w_vector=0.5),
    "rrf w_vec=0.25": replace(BASE, w_vector=0.25),
    "rrf adaptive (1.0 / ident 0.25)": replace(BASE, ident_w_vector=0.25),
    "rrf adaptive (1.0 / ident 0.0)": replace(BASE, ident_w_vector=0.0),
    "score w_vec=1.0": replace(BASE, fusion="score"),
    "score w_vec=0.5": replace(BASE, fusion="score", w_vector=0.5),
    "score w_vec=0.25": replace(BASE, fusion="score", w_vector=0.25),
    "score adaptive (1.0 / ident 0.25)": replace(BASE, fusion="score", ident_w_vector=0.25),
    "score adaptive2 (+ NL bm25 0.5)": replace(
        BASE, fusion="score", ident_w_vector=0.25, nl_w_bm25=0.5
    ),
    # Embedding-heavier fusion for identifier-free questions (round 3 check).
    "score adaptive2 (NL bm25 0.25)": replace(
        BASE, fusion="score", ident_w_vector=0.25, nl_w_bm25=0.25
    ),
    "score adaptive2 (NL bm25 0.1)": replace(
        BASE, fusion="score", ident_w_vector=0.25, nl_w_bm25=0.1
    ),
    "score adaptive2 (NL bm25 0.0)": replace(
        BASE, fusion="score", ident_w_vector=0.25, nl_w_bm25=0.0
    ),
}


def reworded_split(query_id: str) -> str:
    """par-01…10 = reworded-tuning (may be used for selection); par-11…20 = held out."""
    return "reworded-tuning" if int(query_id.split("-")[1]) <= 10 else "reworded-held-out"


def load_sets() -> dict[str, list[dict]]:
    sets: dict[str, list[dict]] = {"original": json.loads(GOLD.read_text())["queries"]}
    if PARA.exists():
        for q in json.loads(PARA.read_text())["queries"]:
            sets.setdefault(reworded_split(q["id"]), []).append(q)
    return sets


def evaluate(cfg: RankingConfig, queries: list[dict], k: int = 20) -> dict:
    from src.retrieval.search import get_retriever

    r = get_retriever()
    ms = []
    for q in queries:
        hits = r.search(q["query"], k, cfg)
        res = [
            {"file_path": h.file_path, "start_line": h.start_line, "end_line": h.end_line}
            for h in hits
        ]
        ms.append(query_metrics(res, q["gold"]))
    return {m: round(statistics.fmean(x[m] for x in ms), 3) for m in ("mrr", "r@10", "p@1")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = args.tag or f"{config.EMBED_MODEL.split('/')[-1]}-{config.EMBED_TEXT}"
    sets = load_sets()
    out = {"embed_model": config.EMBED_MODEL, "embed_text": config.EMBED_TEXT, "results": {}}
    lines = [
        f"# Fusion sweep — `{config.EMBED_MODEL}` ({config.EMBED_TEXT} chunk text)",
        "",
        "Selection uses `original` + `reworded-tuning`; `reworded-held-out` is never used for "
        "choices. MRR / R@10 / P@1.",
        "",
        "| Variant | " + " | ".join(sets) + " |",
        "|---|" + "---|" * len(sets),
    ]
    for name, cfg in VARIANTS.items():
        row = {s: evaluate(cfg, qs) for s, qs in sets.items()}
        out["results"][name] = row
        cells = [f"{v['mrr']:.3f} / {v['r@10']:.3f} / {v['p@1']:.3f}" for v in row.values()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
        print(lines[-1], flush=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / f"fusion_sweep_{tag}.json").write_text(json.dumps(out, indent=2))
    (config.RESULTS_DIR / f"fusion_sweep_{tag}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
