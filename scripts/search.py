"""Search an indexed snippet library from the terminal (retrieval only).

    python -m scripts.search "Given n integers, print the maximum pairwise product" --library apps
    python -m scripts.search "purge old device states (latest version)" --library gladys \
        --version all
Options: --version LABEL|all (default: latest), -k N, --json, --full (print whole snippets).
"""

from __future__ import annotations

import argparse
import json
import logging

from src.codesearch.engine import get_engine


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--library", default="apps")
    ap.add_argument("--version", default=None)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    eng = get_engine()
    eng.warm(args.library, args.version or eng.libraries_map()[args.library][-1])
    res = eng.search(args.query, args.library, args.version, k=args.k)
    if args.json:
        print(json.dumps(res, indent=2))
        return
    c = res["category"]
    print(
        f"query category: {c['style']}{'' if c['style'] == 'code description' else ' program'}; "
        f"topics {c['topics'] or '-'}; "
        f"multi-test {c['multi_test']}; sections {c['sections']}"
    )
    if res["preprocessing"]["hint"]:
        print(f"version hint: {res['preprocessing']['hint']['phrase']!r}")
    for p in res["passes"]:
        print(f"  [{p['ms']:7.1f} ms] {p['name']}: {p['detail']}")
    print(f"latency {res['latency_ms']:.0f} ms over {res['library_size']} snippets\n")
    for r in res["results"]:
        loc = f"{r['path']}:{r['start_line']}-{r['end_line']}" if r["path"] else r["id"]
        print(f"#{r['rank']}  score {r['score']:.3f}  {loc}  @{r['version']}  ({r['style']})")
        if r.get("matched_literals"):
            print(f"    shared literals: {r['matched_literals']}")
        if r.get("lineage"):
            vs = ", ".join(
                f"{v['version']}{'=' if v['same_text'] else '~'}" for v in r["lineage"]["versions"]
            )
            print(f"    lineage versions: {vs}")
        lines = r["text"].rstrip().split("\n")
        for ln in lines if args.full else lines[:8]:
            print("    | " + ln)
        if not args.full and len(lines) > 8:
            print(f"    | … ({len(lines) - 8} more lines)")
        print()


if __name__ == "__main__":
    main()
