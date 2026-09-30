"""Index one version of a snippet library (P1). Only changed files / snippets are re-processed.

Codebase at a git ref (version label = the ref):
    python -m scripts.index_version gladys --git data/versions/gladys.git --ref v5.1.1
The APPS corpus (all 8 765 snippets) as a snippet set:
    python -m scripts.index_version apps --apps-corpus --version v1
A snippet set from JSON lines ({"id", "text", optional "lineage_key"} per line):
    python -m scripts.index_version mylib --jsonl snippets.jsonl --version 2026-09-27
A synthetic next version of an indexed snippet set (behaviour-preserving mutations):
    python -m scripts.index_version apps --evolve-from v1 --version v2 --rate 0.1
Options: --model KEY (repeatable; default: the submitted pipeline's models for snippet sets,
CODESEARCH_CODEBASE_MODEL for codebases).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

from src import config
from src.codesearch import versions as vstore
from src.codesearch.synthetic import evolve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("library")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--git", help="bare or normal git dir of the codebase")
    src.add_argument("--apps-corpus", action="store_true")
    src.add_argument("--jsonl")
    src.add_argument("--evolve-from", help="existing version to mutate into --version")
    ap.add_argument("--ref", help="git ref (tag/commit/branch) for --git")
    ap.add_argument("--version", help="version label (default: the git ref)")
    ap.add_argument("--rate", type=float, default=0.1, help="fraction mutated for --evolve-from")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", action="append")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
    )
    t0 = time.perf_counter()
    stats: dict = {}
    if args.git:
        if not args.ref:
            ap.error("--git needs --ref")
        git = Path(args.git)
        snippets = vstore.snippets_from_git(git, args.ref, stats=stats)
        when = subprocess.run(
            ["git", f"--git-dir={git}", "log", "-1", "--format=%ct", stats["commit"]],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        version = args.version or args.ref
        source = {
            "kind": "codebase",
            "git": str(git),
            "ref": args.ref,
            "order": int(when),
            "description": "Gladys Assistant (JavaScript), function-level snippets",
        }
        models = args.model or [config.CODESEARCH_CODEBASE_MODEL]
    else:
        from src.codesearch.submission import FINAL_CONFIG

        models = args.model or [k for k, _ in FINAL_CONFIG.models]
        if not args.version:
            ap.error("--version is required for snippet sets")
        version = args.version
        if args.apps_corpus:
            from src.codesearch.data import load_apps

            corpus = load_apps("test").corpus  # the full 8 765-snippet corpus (texts only)
            snippets = [vstore.Snippet(id=k, text=v, lineage_key=k) for k, v in corpus.items()]
            desc = "CoIR APPS corpus: 8 765 Python solutions"
        elif args.jsonl:
            snippets = vstore.snippets_from_jsonl(Path(args.jsonl))
            desc = f"snippet set from {Path(args.jsonl).name}"
        else:
            prev, _ = vstore.load_version(args.library, args.evolve_from, models[0])
            base = {s.id: s.text for s in prev}
            hist = evolve(base, [args.evolve_from, version], [0.0, args.rate], seed=args.seed)
            snippets = [
                vstore.Snippet(id=k, text=t, lineage_key=k) for k, t in hist[version].items()
            ]
            changed = sum(hist[version][k] != base[k] for k in base)
            stats["mutated"] = changed
            desc = f"synthetic evolution of {args.evolve_from} ({changed} snippets mutated)"
        existing = vstore.list_versions(args.library)
        source = {
            "kind": "snippets",
            "order": max([m.get("source", {}).get("order", 0) for m in existing] + [0]) + 1
            if not any(m["version"] == version for m in existing)
            else next(m["source"].get("order", 0) for m in existing if m["version"] == version),
            "description": desc,
        }
    view = "raw"
    if not args.git:
        from src.codesearch.submission import FINAL_CONFIG

        view = FINAL_CONFIG.snippet_view
    man = vstore.index_version(args.library, version, snippets, models, source, stats, view)
    lin = vstore.VERSIONS_DIR / args.library / "lineages.json"
    lin.unlink(missing_ok=True)  # rebuilt lazily for the all-versions view
    enc = {k: {"encoded": v["encoded"], "reused": v["cached"]} for k, v in man["encode"].items()}
    print(
        json.dumps(
            {
                "library": args.library,
                "version": version,
                "snippets": len(snippets),
                "files_parsed": stats.get("files_parsed"),
                "files_reused": stats.get("files_reused"),
                "embeddings": enc,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
