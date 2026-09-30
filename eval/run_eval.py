"""Run the benchmark over retrieval ablations and the answer pipeline; write results/.

Query sets (reported separately):
  original     eval/gold.json               36 auto-derived queries (semantic/structural/usage)
  reworded-*   eval/gold_paraphrased.json   LLM-written questions with no lexical overlap with
                                            their target; par-01…10 = tuning, par-11…20 = held out
Structural queries are also scored against the independent checker's labels
(eval/gold_check.json, see eval/structural_checker.py).

Usage:
  python -m eval.run_eval                                  # all systems, all sets
  python -m eval.run_eval --systems bm25,hybrid --merge    # update some systems in results/
  python -m eval.run_eval --explain                        # also time the streamed explanation
  python -m eval.run_eval --import-rows agent_v1=path.json # score another run's per-query rows

Systems:
  bm25, vector, hybrid_rrf (equal-weight RRF, the old default), hybrid (config.RANKING),
  hybrid_graph (config.AGENT_RANKING), hybrid_rerank   → retriever top-20
  agent_norefine → planner + search + read + assess + verify, refinement off (ablation)
  agent_nollm    → + refinement round 1 (graph/catalog expansion), no LLM
  agent          → + phase B (LLM planning fallback, constrained rewrite); scored on the final
                   results; latency = time to the FIRST results, final latency reported too;
                   the explanation is timed separately (--explain)
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from eval.metrics import aggregate, percentile, query_metrics
from src import config
from src.config import RankingConfig

logger = logging.getLogger(__name__)

GOLD = Path(__file__).parent / "gold.json"
PARA = Path(__file__).parent / "gold_paraphrased.json"
CHECK = Path(__file__).parent / "gold_check.json"


def retrieval_systems() -> dict[str, RankingConfig]:
    return {
        "bm25": RankingConfig(use_vectors=False),
        "vector": RankingConfig(use_bm25=False),
        "hybrid_rrf": RankingConfig(),
        "hybrid": config.RANKING,
        "hybrid_graph": config.AGENT_RANKING,
        "hybrid_rerank": replace(config.RANKING, use_rerank=True),
    }


# agent_norefine: planner + search + read + verify, refinement off (ablation)
# agent_nollm:    + refinement round 1 (graph/catalog expansion), no LLM at all
# agent:          + phase B (LLM planning fallback, constrained rewrite); final results scored
# agent_rewrite_on: agent with refinement round 2 (constrained rewrite) forced on — ablation
AGENT_SYSTEMS = ("agent_norefine", "agent_nollm", "agent", "agent_rewrite_on")
ALL_SYSTEMS = (*retrieval_systems(), *AGENT_SYSTEMS)


def reworded_split(query_id: str) -> str:
    """par-01…10 may be used for selection (reworded-tuning); par-11…20 are held out."""
    return "reworded-tuning" if int(query_id.split("-")[1]) <= 10 else "reworded-held-out"


def load_queries() -> list[dict]:
    qs = [dict(q, set=q.get("set", "original")) for q in json.loads(GOLD.read_text())["queries"]]
    if PARA.exists():
        qs += [
            dict(q, set=reworded_split(q["id"])) for q in json.loads(PARA.read_text())["queries"]
        ]
    return qs


def _check_labels() -> dict[str, list[dict]]:
    return json.loads(CHECK.read_text()) if CHECK.exists() else {}


def _row(q: dict, results: list[dict], latency_ms: float, checks: dict) -> dict:
    row = {
        "id": q["id"],
        "set": q["set"],
        "type": q["type"],
        "latency_ms": round(latency_ms, 1),
        "results": results,
        "metrics": query_metrics(results, q["gold"]),
    }
    if q["id"] in checks:
        row["metrics_check"] = query_metrics(results, checks[q["id"]])
    return row


def _run_retrieval(name: str, queries: list[dict], checks: dict) -> list[dict]:
    from src.retrieval.search import get_retriever

    r = get_retriever()
    cfg = retrieval_systems()[name]
    r.search("warm up the models", 5, cfg)
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        hits = r.search(q["query"], 20, cfg)
        ms = (time.perf_counter() - t0) * 1000
        results = [
            {"file_path": h.file_path, "start_line": h.start_line, "end_line": h.end_line}
            for h in hits
        ]
        rows.append(_row(q, results, ms, checks))
    return rows


def _citation_valid(snippet: dict) -> bool:
    """Independent re-check: the snippet text must equal the file's lines at that range."""
    path = config.REPO_ROOT / snippet["file_path"]
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    except OSError:
        return False
    s, e = snippet["start_line"], snippet["end_line"]
    return 1 <= s <= e <= len(lines) and "\n".join(lines[s - 1 : e]) == snippet["text"]


def _run_agent(
    name: str, queries: list[dict], checks: dict, use_cache: bool, time_explanation: bool
) -> list[dict]:
    from dataclasses import replace as _replace

    from src.agent.agent import answer, continue_with_llm, stream_explanation
    from src.agent.llm import LLMClient

    use_llm = name in {"agent", "agent_rewrite_on"}
    refine = config.REFINE
    if name == "agent_norefine":
        refine = _replace(config.REFINE, expand=False, rewrite=False)
    elif name == "agent_rewrite_on":
        refine = _replace(config.REFINE, rewrite=True)
    llm = LLMClient(use_cache=use_cache) if use_llm else None
    if use_llm and not llm.available():
        raise RuntimeError("Ollama is not reachable; start it or skip the 'agent' system.")
    answer("warm up the models", use_llm=False)
    rows = []
    for i, q in enumerate(queries, 1):
        res = answer(q["query"], use_llm=use_llm, llm=llm, refine=refine)
        first = _locs(res)
        if use_llm:
            continue_with_llm(res, llm)
        d = res.to_dict()
        results = _locs(res)
        valid = [_citation_valid(s) for s in d["snippets"]]
        row = _row(q, results, d["latency_ms"], checks)
        row.update(
            {
                "metrics_first": query_metrics(first, q["gold"]),
                "final_latency_ms": d["final_latency_ms"],
                "route": d["route"].get("type"),
                "route_category": d["route"].get("category"),
                "rewritten_query": d["route"].get("rewritten_query"),
                "refinement_rounds": d["refinement_rounds"],
                "changed_by_llm": d["results_changed_by_llm"],
                "steps": len(d["steps"]),
                "step_kinds": sorted({s["kind"] for s in d["steps"]}),
                "tool_calls": d["tool_calls"],
                "llm_calls": sum(1 for s in d["steps"] if s["kind"] == "llm")
                + sum(1 for s in d["steps"] if s["name"] == "LLM planning fallback"),
                "tokens": d["tokens"],
                "confidence": d["confidence"],
                "citations": len(valid),
                "citations_valid": sum(valid),
                "error": d["error"],
            }
        )
        if use_llm and time_explanation and res.snippets:
            t0 = time.perf_counter()
            first_tok, text = None, []
            for piece in stream_explanation(res, llm):
                first_tok = (
                    first_tok if first_tok is not None else (time.perf_counter() - t0) * 1000
                )
                text.append(piece)
            row["explain_first_token_ms"] = round(first_tok or 0.0, 1)
            row["explain_total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            row["explanation"] = "".join(text)[:600]
        rows.append(row)
        logger.info(
            "%s %d/%d %s first %.2fs final %.2fs mrr=%.2f rounds=%d%s",
            name,
            i,
            len(queries),
            q["id"],
            d["latency_ms"] / 1000,
            d["final_latency_ms"] / 1000,
            row["metrics"]["mrr"],
            d["refinement_rounds"],
            " (changed by LLM)" if d["results_changed_by_llm"] else "",
        )
    return rows


def _locs(res) -> list[dict]:
    return [
        {"file_path": s.file_path, "start_line": s.start_line, "end_line": s.end_line}
        for s in res.snippets
    ]


def _import_rows(path: Path, queries: list[dict], checks: dict) -> list[dict]:
    """Score another harness's per-query rows (same ids) against the current labels."""
    data = json.loads(path.read_text())
    rows_in = next(iter(data.values())) if isinstance(data, dict) else data
    by_id = {r["id"]: r for r in rows_in}
    rows = []
    for q in queries:
        r = by_id.get(q["id"])
        if r is None:
            continue
        row = _row(q, r["results"], r["latency_ms"], checks)
        for k in (
            "tool_calls",
            "llm_calls",
            "tokens",
            "route",
            "route_category",
            "citations",
            "citations_valid",
            "error",
        ):
            if k in r:
                row[k] = r[k]
        row.setdefault("rewritten_query", None)
        rows.append(row)
    return rows


def _count(items) -> dict:
    out: dict = defaultdict(int)
    for i in items:
        out[i] += 1
    return dict(out)


def _summarise(rows: list[dict]) -> dict:
    out = {"overall": aggregate(rows)}
    by_set: dict[str, list] = defaultdict(list)
    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_set[r["set"]].append(r)
        if r["set"] == "original":
            by_type[r["type"]].append(r)
    out["by_set"] = {s: aggregate(v) for s, v in sorted(by_set.items())}
    out["by_type"] = {t: aggregate(v) for t, v in sorted(by_type.items())}
    checked = [dict(r, metrics=r["metrics_check"]) for r in rows if "metrics_check" in r]
    if checked:
        out["structural_checker_labels"] = aggregate(checked)
    if rows and "tool_calls" in rows[0]:
        agent = {
            "tool_calls_mean": round(statistics.fmean(r.get("tool_calls", 0) for r in rows), 2),
            "tool_calls_max": max(r.get("tool_calls", 0) for r in rows),
            "llm_calls_mean": round(statistics.fmean(r.get("llm_calls", 0) for r in rows), 2),
            "llm_calls_max": max(r.get("llm_calls", 0) for r in rows),
            "tokens_mean_per_query": round(
                statistics.fmean(
                    r.get("tokens", {}).get("prompt", 0) + r.get("tokens", {}).get("completion", 0)
                    for r in rows
                ),
                1,
            ),
            "rewrite_rate": round(
                statistics.fmean(1.0 if r.get("rewritten_query") else 0.0 for r in rows), 3
            ),
            "refined_rate": round(
                statistics.fmean(1.0 if r.get("refinement_rounds") else 0.0 for r in rows), 3
            ),
            "changed_by_llm": sum(1 for r in rows if r.get("changed_by_llm")),
            "steps_mean": round(statistics.fmean(r.get("steps", 0) for r in rows), 1),
            "final_latency_p95_ms": round(
                percentile([r.get("final_latency_ms", r["latency_ms"]) for r in rows], 95), 1
            ),
            "routing": dict(sorted(_count(r.get("route_category") for r in rows).items())),
            "citations_total": sum(r.get("citations", 0) for r in rows),
            "citation_validity": round(
                sum(r.get("citations_valid", 0) for r in rows)
                / max(1, sum(r.get("citations", 0) for r in rows)),
                4,
            ),
            "errors": sum(1 for r in rows if r.get("error")),
        }
        ft = [r["explain_first_token_ms"] for r in rows if "explain_first_token_ms" in r]
        tt = [r["explain_total_ms"] for r in rows if "explain_total_ms" in r]
        if ft:
            agent["explain_first_token_p50_ms"] = round(percentile(ft, 50), 1)
            agent["explain_first_token_p95_ms"] = round(percentile(ft, 95), 1)
            agent["explain_total_p50_ms"] = round(percentile(tt, 50), 1)
            agent["explain_total_p95_ms"] = round(percentile(tt, 95), 1)
        out["agent"] = agent
    return out


def _fmt(o: dict | None) -> str:
    if not o:
        return "—"
    return f"{o['mrr']:.3f} / {o['p@1']:.2f} / {o['r@10']:.2f}"


def _markdown(report: dict) -> str:
    m = report["index"]
    sets = report["gold"]["by_set"]
    lines = [
        "# Evaluation results",
        "",
        f"Generated {report['generated_at']} on {report['machine']['cpu']} "
        f"({report['machine']['cores']} cores, {report['machine']['ram_gb']} GB RAM), CPU-only "
        f"inference. Target: Gladys Assistant @ `{str(m['repo_commit'])[:7]}` "
        f"({m['files_indexed']} files). Query sets: {sets}. Labels are auto-derived "
        "(docs/approach.md §A.5); structural labels are cross-checked by an independent text "
        "checker (`results/structural_check.md`).",
        "",
        "## Main table — MRR / P@1 / R@10, time to first results",
        "",
        "`reworded-tuning` (par-01…10) may be used to choose settings; `reworded-held-out` "
        "(par-11…20) never is. Latency = time until the first verified results are returned.",
        "",
        "| System | original | reworded-tuning | reworded-held-out | all | p50 | p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in report["systems"].items():
        o = s["overall"]
        b = s["by_set"]
        lines.append(
            f"| {name} | {_fmt(b.get('original'))} | {_fmt(b.get('reworded-tuning'))} | "
            f"{_fmt(b.get('reworded-held-out'))} | {_fmt(o)} | "
            f"{o['latency_p50_ms'] / 1000:.2f} s | {o['latency_p95_ms'] / 1000:.2f} s |"
        )
    lines += [
        "",
        "## Original set by query type — MRR / P@1 / R@10",
        "",
        "| System | semantic | structural (index labels) | structural (checker labels) | usage |",
        "|---|---|---|---|---|",
    ]
    for name, s in report["systems"].items():
        b = s["by_type"]
        lines.append(
            f"| {name} | {_fmt(b.get('semantic'))} | {_fmt(b.get('structural'))} | "
            f"{_fmt(s.get('structural_checker_labels'))} | {_fmt(b.get('usage'))} |"
        )
    agents = {n: s["agent"] for n, s in report["systems"].items() if "agent" in s}
    if agents:
        lines += [
            "",
            "## Answer pipeline behaviour",
            "",
            "| System | steps/query | refined | LLM calls mean/max | results changed by LLM | "
            "final results p95 | explanation first token p50/p95 | explanation total p50/p95 | "
            "citation validity | citations | errors |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, a in agents.items():
            ft = (
                f"{a['explain_first_token_p50_ms'] / 1000:.1f} / "
                f"{a['explain_first_token_p95_ms'] / 1000:.1f} s"
                if "explain_first_token_p50_ms" in a
                else "—"
            )
            tt = (
                f"{a['explain_total_p50_ms'] / 1000:.1f} / {a['explain_total_p95_ms'] / 1000:.1f} s"
                if "explain_total_p50_ms" in a
                else "—"
            )
            lines.append(
                f"| {name} | {a.get('steps_mean', '—')} | {a.get('refined_rate', 0):.2f} | "
                f"{a['llm_calls_mean']} / {a['llm_calls_max']} | {a.get('changed_by_llm', 0)} | "
                f"{a.get('final_latency_p95_ms', 0) / 1000:.2f} s | {ft} | {tt} | "
                f"{a['citation_validity'] * 100:.1f}% | {a['citations_total']} | "
                f"{a['errors']} |"
            )
    t = m.get("timings_sec", {})
    lines += [
        "",
        "## Indexing cost",
        "",
        "| Stage | Seconds |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in t.items()],
        "",
        f"Peak RSS: {m.get('peak_rss_mb')} MB · index size on disk: "
        f"{(m.get('index_bytes_total') or 0) / 1e6:.1f} MB · counts: {m.get('counts')}",
        "",
        "## Configuration",
        "",
        f"- Embedding: `{report['config']['embed_model']}` "
        f"({report['config']['embed_text']} chunk text)",
        f"- LLM: `{report['config']['llm_model']}`, num_gpu={report['config']['llm_num_gpu']}"
        " (CPU)",
        f"- Refinement: `{report['config']['refine']}`",
        f"- Ranking configs: `{report['config']['ranking']}`",
        f"- LLM response cache used: {report['config']['use_llm_cache']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    ap.add_argument("--sets", default="original,reworded-tuning,reworded-held-out")
    ap.add_argument("--use-llm-cache", action="store_true")
    ap.add_argument("--explain", action="store_true", help="time the streamed explanation")
    ap.add_argument(
        "--import-rows",
        action="append",
        default=[],
        help="name=path to a per-query JSON from another run",
    )
    ap.add_argument("--out", default=str(config.RESULTS_DIR))
    ap.add_argument(
        "--merge",
        action="store_true",
        help="update these systems inside existing results instead of replacing",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    wanted_sets = {s.strip() for s in args.sets.split(",")}
    queries = [q for q in load_queries() if q["set"] in wanted_sets]
    checks = _check_labels()
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - set(ALL_SYSTEMS)
    if unknown:
        ap.error(f"unknown systems {sorted(unknown)}")

    report_systems, per_query = {}, {}
    for name in systems:
        t0 = time.perf_counter()
        logger.info("=== %s on %d queries", name, len(queries))
        if name in retrieval_systems():
            rows = _run_retrieval(name, queries, checks)
        else:
            rows = _run_agent(name, queries, checks, args.use_llm_cache, args.explain)
        report_systems[name] = _summarise(rows) | {"wall_sec": round(time.perf_counter() - t0, 1)}
        per_query[name] = rows
        logger.info("%s: %s", name, report_systems[name]["by_set"])
    for spec in args.import_rows:
        name, path = spec.split("=", 1)
        rows = _import_rows(Path(path), queries, checks)
        report_systems[name] = _summarise(rows)
        per_query[name] = rows

    manifest = json.loads(config.MANIFEST_PATH.read_text())
    import psutil

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": {
            "cpu": "Apple M1 Pro" if platform.machine() == "arm64" else platform.processor(),
            "cores": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30),
            "platform": platform.platform(),
        },
        "gold": {
            "n": len(queries),
            "by_set": _count(q["set"] for q in queries),
            "by_type": _count(q["type"] for q in queries),
        },
        "index": {
            k: manifest.get(k)
            for k in (
                "repo_commit",
                "files_indexed",
                "counts",
                "timings_sec",
                "index_bytes",
                "index_bytes_total",
                "embedding",
                "resolution",
            )
        }
        | {"peak_rss_mb": manifest.get("peak_rss_mb_this_run")},
        "config": {
            "embed_model": config.EMBED_MODEL,
            "embed_text": config.EMBED_TEXT,
            "llm_model": config.LLM_MODEL,
            "llm_num_gpu": config.LLM_NUM_GPU,
            "refine": config.REFINE.to_dict(),
            "budget": config.BUDGET.to_dict(),
            "use_llm_cache": args.use_llm_cache,
            "ranking": {n: c.to_dict() for n, c in retrieval_systems().items() if n in systems},
        },
        "systems": report_systems,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.merge and (out / "eval_results.json").exists():
        old_report = json.loads((out / "eval_results.json").read_text())
        old_pq = json.loads((out / "eval_per_query.json").read_text())
        merged = old_report["systems"] | report_systems
        order = [n for n in (*ALL_SYSTEMS, *merged) if n in merged]
        report["systems"] = {n: merged[n] for n in dict.fromkeys(order)}
        report["config"]["ranking"] = old_report["config"]["ranking"] | report["config"]["ranking"]
        per_query = old_pq | per_query
    (out / "eval_results.json").write_text(json.dumps(report, indent=2))
    (out / "eval_per_query.json").write_text(json.dumps(per_query, indent=1))
    (out / "eval_results.md").write_text(_markdown(report))
    print(_markdown(report))


if __name__ == "__main__":
    main()
