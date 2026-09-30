"""Synthetic multi-version benchmark for evolutionary retrieval (bonus, R-BX-1).

Library: the 5 000 APPS **train** snippets evolved into versions v1 → v2 → v3 by
behaviour-preserving mutations (renames, comments, fast I/O, `main()` wrapping, reformatting;
`src/codesearch/synthetic.py`): v2 changes 50 % of v1, v3 changes 40 % of v2, the rest carry over
byte-identical. All 15 000 (snippet, version) entries are indexed together.

Queries: the stdin-style train problems of dev-tune (parameter choice) and dev-holdout
(reported), each given a version hint in rotation: none (→ the latest version is wanted),
"(latest version)", "(original version)", "in v2".

Relevance (graded): an entry of the target lineage whose text equals the wanted version's text =
2; any other version of the target lineage = 1; everything else 0.

Systems:
  flat            rank all entries by dense score, hint left in the query
  flat+strip      same, hint phrase removed before encoding
  lineage         hint stripped; lineages inferred from content only (no keys given);
                  collapse to one entry per lineage; version chosen by hint / recency
  lineage-oracle  as lineage but with the true lineage ids (upper bound for grouping)

    python -m eval.evolution_bench [--model jina-code-0.5b]
Writes results/evolution_bench.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

from src import config
from src.codesearch.data import dev_splits
from src.codesearch.encoders import Encoder
from src.codesearch.lineage import (
    collapse,
    infer_lineages,
    members_by_lineage,
    parse_version_hint,
)
from src.codesearch.metrics import ndcg_at_k
from src.codesearch.pipeline import PipelineConfig, encode_queries
from src.codesearch.synthetic import evolve

VERSIONS = ["v1", "v2", "v3"]
HINTS = ["", " (latest version)", " (original version)", " in v2"]
OUT_JSON = config.RESULTS_DIR / "evolution_bench.json"
OUT_MD = config.RESULTS_DIR / "evolution_bench.md"


def build(model: str):
    splits = dev_splits()
    base = splits["dev-tune"].corpus
    history = evolve(base, VERSIONS, [0.0, 0.5, 0.4])
    entries = [(sid, v, history[v][sid]) for v in VERSIONS for sid in base]
    enc = Encoder(model)
    t = time.perf_counter()
    emb = enc.encode([e[2] for e in entries], "doc")
    enc_s = time.perf_counter() - t
    return history, entries, emb, enc_s


def gold_for(qid_doc: str, hint_kind: str, history, entries) -> dict[int, int]:
    wanted = {"": "v3", "latest": "v3", "original": "v1", "v2": "v2"}[hint_kind]
    want_text = history[wanted][qid_doc]
    rel = {}
    for i, (sid, _v, text) in enumerate(entries):
        if sid == qid_doc:
            rel[i] = 2 if text == want_text else 1
    return rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jina-code-0.5b")
    ap.add_argument("--cos-min", type=float, default=0.9)
    ap.add_argument("--jaccard-min", type=float, default=0.3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    history, entries, emb, enc_s = build(args.model)
    texts = [e[2] for e in entries]
    vlabels = [e[1] for e in entries]
    vrank = np.array([VERSIONS.index(v) for v in vlabels])
    t = time.perf_counter()
    inferred = infer_lineages(
        texts, vlabels, emb, cos_min=args.cos_min, jaccard_min=args.jaccard_min
    )
    lineage_s = time.perf_counter() - t
    sids = [e[0] for e in entries]
    true_ids = {s: i for i, s in enumerate(dict.fromkeys(sids))}
    oracle = [true_ids[s] for s in sids]
    grouping = _grouping_quality(inferred, oracle)
    cfg = PipelineConfig(models=((args.model, 1.0),))
    splits = dev_splits()
    report = {
        "model": args.model,
        "entries": len(entries),
        "unique_texts": len(set(texts)),
        "encode_s": round(enc_s, 1),
        "lineage_s": round(lineage_s, 1),
        "lineages_inferred": len(set(inferred)),
        "lineages_true": len(set(oracle)),
        "grouping": grouping,
        "params": {"cos_min": args.cos_min, "jaccard_min": args.jaccard_min},
        "sets": {},
    }
    for set_name in ("dev-tune", "dev-holdout"):
        s = splits[set_name]
        qids = list(s.queries)
        hints = [HINTS[i % len(HINTS)] for i in range(len(qids))]
        raw_q = [s.queries[q] + h for q, h in zip(qids, hints, strict=True)]
        parsed = [parse_version_hint(q, VERSIONS) for q in raw_q]
        Q_raw = encode_queries(raw_q, args.model, cfg)
        Q_clean = encode_queries([p.cleaned_query for p in parsed], args.model, cfg)
        kinds = [
            ""
            if not h
            else ("latest" if "latest" in h else "original" if "original" in h else "v2")
            for h in hints
        ]
        golds = [
            gold_for(list(s.qrels[q])[0], k, history, entries)
            for q, k in zip(qids, kinds, strict=True)
        ]
        # Corpus snippets of the dev split: the gold doc id is the snippet id in `base`.
        res = {}
        S_raw = Q_raw @ emb.T
        S_clean = Q_clean @ emb.T
        res["flat"] = _score_flat(S_raw, golds, sids)
        res["flat+strip"] = _score_flat(S_clean, golds, sids)
        for name, lin in (("lineage", inferred), ("lineage-oracle", oracle)):
            mem = members_by_lineage(lin)
            ranked = []
            t = time.perf_counter()
            for qi in range(len(qids)):
                hits = collapse(
                    S_clean[qi], lin, vrank, parsed[qi], vlabels, top_k=10, members_of=mem
                )
                ranked.append([h.index for h in hits])
            ms = (time.perf_counter() - t) * 1000 / len(qids)
            res[name] = _metrics(ranked, golds, sids) | {"collapse_ms_per_query": round(ms, 2)}
        report["sets"][set_name] = res
        print(set_name, {k: round(v["ndcg@10"], 4) for k, v in res.items()}, flush=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    _write_md(report)


def _score_flat(S: np.ndarray, golds, sids: list[str]) -> dict:
    ranked = [list(np.argsort(-S[i])[:10]) for i in range(S.shape[0])]
    return _metrics(ranked, golds, sids)


def _metrics(ranked: list[list[int]], golds: list[dict[int, int]], sids: list[str]) -> dict:
    """Graded NDCG (every version of the target lineage relevant — rewards listing duplicates),
    de-duplicated NDCG (only the first entry of each lineage counts; ideal = the wanted version at
    rank 1), exact-version@1, lineage MRR@10 and distinct lineages in the top 10."""
    nd, nd_dedup, exact1, lin_mrr, distinct = [], [], [], [], []
    for r, g in zip(ranked, golds, strict=True):
        rel = {str(k): v for k, v in g.items()}
        nd.append(ndcg_at_k([str(i) for i in r], rel, 10))
        seen: set[str] = set()
        dedup = []
        for i in r:
            lin = sids[int(i)]
            dedup.append(f"{i}" if lin not in seen else f"dup{i}")
            seen.add(lin)
        nd_dedup.append(_ndcg_single_ideal(dedup, rel, 2))
        distinct.append(len({sids[int(i)] for i in r}))
        exact1.append(1.0 if r and g.get(int(r[0]), 0) == 2 else 0.0)
        first = next((k for k, i in enumerate(r, 1) if int(i) in g), None)
        lin_mrr.append(1 / first if first else 0.0)
    return {
        "ndcg@10": float(np.mean(nd)),
        "ndcg@10_dedup": float(np.mean(nd_dedup)),
        "exact_version@1": float(np.mean(exact1)),
        "lineage_mrr@10": float(np.mean(lin_mrr)),
        "distinct_lineages@10": float(np.mean(distinct)),
    }


def _ndcg_single_ideal(ranked: list[str], rel: dict[str, int], top: int) -> float:
    """DCG@10 of `ranked` (duplicates already renamed to irrelevant ids) over the DCG of one
    perfect hit (the wanted version at rank 1): a lineage should be listed once."""
    import math

    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(k + 1) for k, d in enumerate(ranked[:10], 1))
    return dcg / (2**top - 1)


def _grouping_quality(pred: list[int], true: list[int]) -> dict:
    """Pairwise precision / recall of 'same lineage' decisions."""
    from collections import Counter

    def pairs(labels):
        c = Counter(labels)
        return sum(v * (v - 1) // 2 for v in c.values())

    joint = pairs(list(zip(pred, true, strict=True)))
    p, t = pairs(pred), pairs(true)
    return {
        "pair_precision": round(joint / p, 4) if p else 1.0,
        "pair_recall": round(joint / t, 4) if t else 1.0,
    }


def _write_md(r: dict) -> None:
    lines = [
        "# Evolutionary retrieval benchmark (synthetic APPS versions)",
        "",
        f"Model `{r['model']}`; {r['entries']} entries ({r['unique_texts']} distinct texts) over "
        f"versions v1–v3; lineages inferred from content: {r['lineages_inferred']} "
        f"(true {r['lineages_true']}), pair precision {r['grouping']['pair_precision']}, "
        f"pair recall {r['grouping']['pair_recall']} (cos ≥ {r['params']['cos_min']}, "
        f"shape-Jaccard ≥ {r['params']['jaccard_min']}); lineage inference {r['lineage_s']} s.",
        "",
        "Relevance: wanted version of the target lineage = 2, other versions = 1. "
        "Queries = stdin-style APPS train problems with a rotating version hint.",
        "",
        "- **NDCG@10 dedup** — each lineage counts once (later versions of an already-listed",
        "  lineage score 0); ideal = the wanted version at rank 1. This is the task's metric:",
        "  a user wants the right version of the right snippet, not three copies of it.",
        "- **NDCG@10 graded** — every version of the target lineage relevant; it *rewards* listing",
        "  near-duplicates, so collapse is penalised by construction (shown for completeness).",
        "- **Exact version @1** — the top hit is the wanted version (identical texts count).",
        "- **Distinct lineages @10** — how many different snippets the top 10 shows.",
        "",
        "| Set | System | NDCG@10 dedup | Exact version @1 | Lineage MRR@10 | "
        "Distinct lineages @10 | NDCG@10 graded |",
        "|---|---|---|---|---|---|---|",
    ]
    for set_name, res in r["sets"].items():
        for sysname, m in res.items():
            lines.append(
                f"| {set_name} | {sysname} | {m['ndcg@10_dedup']:.4f} | "
                f"{m['exact_version@1']:.4f} | {m['lineage_mrr@10']:.4f} | "
                f"{m['distinct_lineages@10']:.1f} | {m['ndcg@10']:.4f} |"
            )
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
