"""Calibrate the agent's refinement policy (D-028).

Variants differ only in refinement (gate, graph/catalog expansion, constrained rewrite). For
each, the full pipeline answers every benchmark question; MRR is computed on the FIRST results
(phase A, no LLM) and on the FINAL results (after phase B, which may call the LLM). Selection
uses `original` + `reworded-tuning`; `reworded-held-out` is only reported. A refinement setting is
adopted only if it does not lower the selection MRR of the no-refinement baseline.

Usage: python -m eval.calibrate_agent [--no-llm]   → results/agent_calibration.{json,md}
"""

from __future__ import annotations

import argparse
import json
import statistics

from eval.metrics import query_metrics
from eval.sweep_fusion import reworded_split
from src import config
from src.agent.expand import ExpandConfig
from src.agent.refine import RefineConfig

SETS = ("original", "reworded-tuning", "reworded-held-out")


def load() -> list[dict]:
    from eval.run_eval import load_queries

    qs = load_queries()
    for q in qs:
        if q["set"] != "original":
            q["set"] = reworded_split(q["id"])
    return qs


def variants(with_llm: bool) -> dict[str, tuple[RefineConfig, ExpandConfig]]:
    off = RefineConfig(expand=False, rewrite=False)
    v = {"no refinement": (off, ExpandConfig(enabled=False))}
    for gate in ("low", "always"):
        for lam in (0.1, 0.25, 0.5):
            for seeds in (3, 5):
                v[f"expand gate={gate} λ={lam} seeds={seeds}"] = (
                    RefineConfig(gate=gate, rewrite=False),
                    ExpandConfig(lam=lam, seeds=seeds),
                )
    if with_llm:
        for gate in ("low", "always"):
            for w in (1.0, 0.5):
                v[f"rewrite only gate={gate} w2={w}"] = (
                    RefineConfig(gate=gate, expand=False, round2_weight=w),
                    ExpandConfig(enabled=False),
                )
        for w in (1.0, 0.5):
            v[f"expand λ=0.1 seeds=3 + rewrite gate=low w2={w}"] = (
                RefineConfig(gate="low", round2_weight=w),
                ExpandConfig(lam=0.1, seeds=3),
            )
    return v


def _refinement_rank(r: dict) -> tuple:
    """Tie-break among equal selection MRR: prefer refinement ON (the brief asks to keep it
    unless it lowers MRR), then the gentlest setting (gate=low, smallest λ, fewest seeds)."""
    ref, exp = r["refine"], r["expand"]
    on = int(ref["expand"] and exp["enabled"]) + int(ref["rewrite"])
    return (on, ref["gate"] == "low", -exp["lam"], -exp["seeds"])


def run_variant(queries, refine: RefineConfig, expand: ExpandConfig, llm_phase: bool) -> list:
    from src.agent.agent import answer, continue_with_llm

    rows = []
    for q in queries:
        r = answer(q["query"], use_llm=llm_phase, refine=refine, expand=expand)
        first = _res(r)
        final = first
        if llm_phase:
            continue_with_llm(r)
            final = _res(r)
        rows.append(
            {
                "id": q["id"],
                "set": q["set"],
                "first": query_metrics(first, q["gold"]),
                "final": query_metrics(final, q["gold"]),
                "latency_ms": r.latency_ms,
                "final_latency_ms": r.final_latency_ms,
                "rounds": r.refinement_rounds,
                "changed": r.results_changed_by_llm,
            }
        )
    return rows


def _res(r) -> list[dict]:
    return [
        {"file_path": s.file_path, "start_line": s.start_line, "end_line": s.end_line}
        for s in r.snippets
    ]


def summarise(rows: list[dict], which: str) -> dict:
    out = {}
    for s in SETS:
        sub = [r for r in rows if r["set"] == s]
        out[s] = round(statistics.fmean(r[which]["mrr"] for r in sub), 4) if sub else 0.0
    sel = [r for r in rows if r["set"] in ("original", "reworded-tuning")]
    out["selection"] = round(statistics.fmean(r[which]["mrr"] for r in sel), 4)
    out["all"] = round(statistics.fmean(r[which]["mrr"] for r in rows), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-llm", action="store_true", help="skip the constrained-rewrite variants")
    args = ap.parse_args()
    queries = load()
    results = {}
    for name, (refine, expand) in variants(not args.no_llm).items():
        llm_phase = refine.rewrite
        rows = run_variant(queries, refine, expand, llm_phase)
        results[name] = {
            "refine": refine.to_dict(),
            "expand": expand.__dict__,
            "first": summarise(rows, "first"),
            "final": summarise(rows, "final"),
            "p95_first_ms": round(
                sorted(r["latency_ms"] for r in rows)[int(0.95 * (len(rows) - 1))], 1
            ),
            "refined": sum(1 for r in rows if r["rounds"]),
            "rows": rows,
        }
        f = results[name]["final"]
        print(
            f"{name:42} sel={f['selection']:.4f} orig={f['original']:.4f} "
            f"tune={f['reworded-tuning']:.4f} held={f['reworded-held-out']:.4f} "
            f"refined={results[name]['refined']}",
            flush=True,
        )
    base = results["no refinement"]["final"]["selection"]
    eligible = {n: r for n, r in results.items() if r["final"]["selection"] >= base}
    best = max(
        eligible,
        key=lambda n: (eligible[n]["final"]["selection"], _refinement_rank(eligible[n])),
    )
    lines = [
        "# Calibration of the agent's refinement policy",
        "",
        "MRR of the final results (after phase B where the variant uses the LLM). Selection = "
        "original + reworded-tuning (46 questions); reworded-held-out (10) is never used for "
        "choices. `refined` = questions on which at least one refinement round ran.",
        "",
        "| Variant | selection | original | reworded-tuning | reworded-held-out | all | refined "
        "| p95 first results |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n, r in results.items():
        f = r["final"]
        mark = "**" if n == best else ""
        lines.append(
            f"| {mark}{n}{mark} | {f['selection']} | {f['original']} | "
            f"{f['reworded-tuning']} | {f['reworded-held-out']} | {f['all']} | "
            f"{r['refined']} | {r['p95_first_ms'] / 1000:.2f} s |"
        )
    lines += [
        "",
        f"Chosen (highest selection MRR that does not fall below the no-refinement "
        f"baseline {base}; ties → refinement on, gentlest setting): **{best}**.",
    ]
    out = {
        "chosen": best,
        "baseline_selection": base,
        "variants": {n: {k: v for k, v in r.items() if k != "rows"} for n, r in results.items()},
        "rows": {n: r["rows"] for n, r in results.items()},
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "agent_calibration.json").write_text(json.dumps(out, indent=1))
    (config.RESULTS_DIR / "agent_calibration.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
