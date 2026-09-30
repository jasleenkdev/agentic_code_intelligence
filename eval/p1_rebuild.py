"""P1: full-build vs incremental-rebuild time for versioned indexes (R-P1-1).

Runs with fresh, isolated caches (embedding cache, chunk cache and version store under
.cache/p1/) so the first build of each library really is a full build:

  gladys   v5.0.4 (full) → v5.1.0 (incremental, 4 releases of changes) → v5.1.1 (1 commit)
  apps     2 000 APPS train snippets as v1 (full) → v2 (10 % of snippets mutated)
           → v3 (1 % mutated)

    python -m eval.p1_rebuild [--model gte-modernbert] [--apps-n 2000]
Writes results/p1_rebuild.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P1_DIR = ROOT / ".cache" / "p1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gte-modernbert")
    ap.add_argument("--apps-n", type=int, default=2000)
    ap.add_argument("--git", default=str(ROOT / "data" / "versions" / "gladys.git"))
    args = ap.parse_args()
    shutil.rmtree(P1_DIR, ignore_errors=True)
    os.environ["CODESEARCH_EMB_CACHE"] = str(P1_DIR / "emb_cache")
    os.environ["CODESEARCH_CHUNK_CACHE"] = str(P1_DIR / "chunk_cache")
    os.environ["CODESEARCH_VERSIONS_DIR"] = str(P1_DIR / "versions")
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from src import config
    from src.codesearch.data import dev_splits
    from src.codesearch.synthetic import evolve
    from src.codesearch.versions import Snippet, index_version, snippets_from_git

    rows = []

    def record(library, version, kind, snippets, stats, t0):
        man = index_version(library, version, snippets, [args.model], {"kind": kind}, stats)
        enc = man["encode"][args.model]
        row = {
            "library": library,
            "version": version,
            "build": kind,
            "snippets": len(snippets),
            "files_parsed": stats.get("files_parsed"),
            "files_reused": stats.get("files_reused"),
            "encoded": enc["encoded"],
            "reused_embeddings": enc["cached"],
            "chunk_s": round(stats.get("chunk_s", 0.0) + stats.get("list_s", 0.0), 2),
            "encode_s": round(enc["seconds"], 1),
            "total_s": round(time.perf_counter() - t0, 1),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    git = Path(args.git)
    for i, ref in enumerate(["v5.0.4", "v5.1.0", "v5.1.1"]):
        t0 = time.perf_counter()
        st: dict = {}
        sn = snippets_from_git(git, ref, stats=st)
        record("gladys", ref, "full" if i == 0 else "incremental", sn, st, t0)

    corpus = dev_splits()["train-pool"].corpus
    base = dict(list(corpus.items())[: args.apps_n])
    hist = evolve(base, ["v1", "v2", "v3"], [0.0, 0.10, 0.01], seed=11)
    for i, v in enumerate(["v1", "v2", "v3"]):
        t0 = time.perf_counter()
        sn = [Snippet(id=k, text=t, lineage_key=k) for k, t in hist[v].items()]
        record("apps-derived", v, "full" if i == 0 else "incremental", sn, {}, t0)

    out = {"model": args.model, "machine": "Apple M1 Pro, CPU only", "rows": rows}
    (config.RESULTS_DIR / "p1_rebuild.json").write_text(json.dumps(out, indent=2))
    lines = [
        "# P1 — full vs incremental index rebuilds",
        "",
        f"Model `{args.model}`, CPU only (Apple M1 Pro). Fresh caches for this run (the first",
        "build of each library is a true full build). Incremental = same command on the next",
        "version:",
        "files are re-parsed only if their git blob changed, snippets re-encoded only if their",
        "text changed (content-hash embedding cache).",
        "",
        "| Library | Version | Build | Snippets | Files parsed / reused | Encoded / reused | "
        "Chunk s | Encode s | Total s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        files = (
            f"{r['files_parsed']} / {r['files_reused']}" if r["files_parsed"] is not None else "—"
        )
        lines.append(
            f"| {r['library']} | {r['version']} | {r['build']} | {r['snippets']} | {files} | "
            f"{r['encoded']} / {r['reused_embeddings']} | {r['chunk_s']} | {r['encode_s']} | "
            f"{r['total_s']} |"
        )
    (config.RESULTS_DIR / "p1_rebuild.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
