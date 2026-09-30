"""Build the code-intelligence index for ./sample_repo.

Usage: python -m scripts.index [--stages structure,catalog,chunks,bm25,vectors] [-v]
"""

from __future__ import annotations

import argparse
import logging

from src.indexer.pipeline import STAGES, build_index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default=",".join(STAGES), help="comma-separated subset of stages")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    unknown = set(stages) - set(STAGES)
    if unknown:
        ap.error(f"unknown stages: {sorted(unknown)}")
    manifest = build_index(stages)
    skipped = manifest.get("files_skipped", [])
    reasons: dict[str, int] = {}
    for s in skipped:
        reasons[s["reason"].split(":")[0]] = reasons.get(s["reason"].split(":")[0], 0) + 1
    print("\n=== Index summary ===")
    print(f"repo commit      : {manifest.get('repo_commit')}")
    print(f"files indexed    : {manifest.get('files_indexed')}")
    print(f"skip records     : {len(skipped)} {reasons}")
    print(f"parse-error files: {manifest.get('files_with_parse_errors')}")
    print(f"counts           : {manifest['counts']}")
    print(f"resolution       : {manifest.get('resolution')}")
    print(f"timings (s)      : {manifest['timings_sec']}")
    print(f"peak RSS (MB)    : {manifest['peak_rss_mb_this_run']}")
    print(f"index size (MB)  : {manifest['index_bytes_total'] / 1e6:.1f} {manifest['index_bytes']}")


if __name__ == "__main__":
    main()
