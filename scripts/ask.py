"""Ask the code-intelligence agent a question from the terminal.

Usage: python -m scripts.ask "which files call setValue before emit?" [--no-llm] [--json]
       [--optimize] [--explain]

First results are printed as soon as the deterministic phase finishes; LLM refinement and the
streamed explanation follow (they never delay the first results). --trace prints every step.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src import config
from src.agent.agent import answer, continue_with_llm, explain, stream_explanation


def _print_step(step) -> None:
    print(
        f"  [{step.phase:6}] {step.kind:7} {step.name:40.40} {step.elapsed_ms:7.0f} ms  "
        f"{step.summary[:90]}",
        flush=True,
    )


def _print_results(res) -> None:
    print(
        f"\nroute: {res.route.get('type')}  confidence: {res.confidence}  first results: "
        f"{res.latency_ms / 1000:.2f}s  refinement rounds: {res.refinement_rounds}"
    )
    print(f"\n{res.answer}\n")
    for s in res.snippets:
        print(f"── {s.location}  {s.symbol or ''}  [{s.reason}]")
        for i, line in enumerate(s.text.split("\n"), s.start_line):
            mark = ">" if i in s.highlight_lines else " "
            print(f"{mark}{i:5d} | {line}")
        print()
    for path in res.call_paths[:10]:
        print("path: " + " → ".join(path))
    for o in res.optimizations:
        print(f"optimisation [{o['check']}] {o['location']}: {o['message']}")
    if res.error:
        print(f"error: {res.error}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question")
    ap.add_argument("--no-llm", action="store_true", help="deterministic only (no phase B)")
    ap.add_argument(
        "--explain",
        action="store_true",
        help="stream an LLM explanation (optional, off by default)",
    )
    ap.add_argument("--no-explain", action="store_true", help="never explain (overrides env)")
    ap.add_argument("--optimize", action="store_true", help="add optimisation suggestions")
    ap.add_argument("--trace", action="store_true", help="print every structured step")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    on_step = _print_step if args.trace and not args.json else None
    res = answer(args.question, use_llm=not args.no_llm, optimize=args.optimize, on_step=on_step)
    use_llm = not args.no_llm
    want_explanation = use_llm and (args.explain or config.EXPLAIN_DEFAULT) and not args.no_explain
    if args.json:
        if use_llm:
            continue_with_llm(res)
        if want_explanation:
            explain(res)
        print(json.dumps(res.to_dict(), indent=2))
        return
    _print_results(res)  # first results: never wait for the LLM
    if use_llm and continue_with_llm(res, on_step=on_step):
        print("\n=== results updated by grounded refinement ===")
        _print_results(res)
    if want_explanation and res.snippets:
        print("\nexplanation: ", end="", flush=True)
        for piece in stream_explanation(res):
            sys.stdout.write(piece)
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    main()
