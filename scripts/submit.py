"""Official AppsRetrieval run, exactly as the hackathon guidelines show (R-P0-1).

    ./run.sh submit                 # test split → release/appsretrieval_results.json
    ./run.sh submit --split train   # plumbing check on the train split (never used for scores)

MTEB's result cache is pointed at a fresh directory for every run so `mteb.evaluate(...)` with
the guidelines' exact arguments really recomputes (its default is to skip cached tasks).
Everything runs on the CPU. Also writes release/run_info.json (timings, pipeline, machine).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "release"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_cache = ROOT / ".cache" / "mteb_run"
    shutil.rmtree(run_cache, ignore_errors=True)
    os.environ["MTEB_CACHE"] = str(run_cache)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    # ---- the guidelines' evaluation pattern ----
    import json as _json  # noqa: F401  (kept as in the guidelines)

    import mteb
    import numpy as np  # noqa: F401  (kept as in the guidelines)
    from mteb.models.abs_encoder import AbsEncoder  # noqa: F401
    from mteb.models.model_meta import ModelMeta  # noqa: F401
    from mteb.types import PromptType  # noqa: F401

    from src.codesearch.submission import FINAL_CONFIG, PrePostPipelineEncoder

    model = PrePostPipelineEncoder()
    task = (
        mteb.get_task("AppsRetrieval", eval_splits=[args.split])
        if args.split != "test"
        else (mteb.get_task("AppsRetrieval"))
    )
    t0 = time.perf_counter()
    result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": 64})
    wall = time.perf_counter() - t0
    task_result = list(result.task_results)[0]
    RELEASE.mkdir(exist_ok=True)
    out = (
        Path(args.out)
        if args.out
        else (
            RELEASE / "appsretrieval_results.json"
            if args.split == "test"
            else ROOT / ".cache" / f"appsretrieval_results_{args.split}.json"
        )
    )
    with open(out, "w") as f:
        # mteb 2.21.8's to_dict() holds a datetime; default=str writes it as an ISO string
        # (the guidelines' json.dump(...) call is otherwise unchanged).
        json.dump(task_result.to_dict(), f, indent=2, default=str)
    # ---- end of the guidelines' pattern ----

    scores = json.loads(out.read_text())["scores"][args.split][0]
    info = {
        "split": args.split,
        "ndcg_at_10": scores["ndcg_at_10"],
        "mrr_at_10": scores["mrr_at_10"],
        "wall_clock_s": round(wall, 1),
        "index_s": round(model.timings.get("index_s", 0.0), 1),
        "search_s": round(model.timings.get("search_s", 0.0), 1),
        # Snippet embeddings encoded in this run vs reused from the content-hash cache (a cold
        # run encodes all 8 765; queries are always listed in the log).
        "snippet_embeddings": model.lib.build_stats.get("encode", {}) if model.lib else {},
        "pipeline": FINAL_CONFIG.to_dict(),
        "mteb_version": mteb.__version__,
        "git_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "machine": f"{platform.machine()} {platform.platform()}, CPU only",
        "python": sys.version.split()[0],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if args.split == "test":
        (RELEASE / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
