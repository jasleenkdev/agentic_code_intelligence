"""Generate the auto-labelled gold benchmark (docs/approach.md §A.5 — labels are NOT hand-made).

Every query and its ground truth come from the index or from grep over the indexed files:
- semantic   : JSDoc `@description` of a function → question; gold = that function's range.
- structural : call-order pairs (certain, same-function), callers, callees from the call graph.
- usage      : front routes (deeplinks), events, API routes from the catalog; string literals
               from grep (every indexed line containing the quoted literal).

Deterministic: fixed seed, sorted inputs. Output: eval/gold.json.
Usage: python -m eval.generate_gold [--per-type 12]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from src import config
from src.graph.queries import find_call_sequence, get_callees, get_callers
from src.indexer import store

SEED = 42
OUT = Path(__file__).parent / "gold.json"
GENERIC = frozenset(
    """get set init start stop create update destroy add remove emit on send connect disconnect
    handle find list save load call run render setState then push""".split()
)


def _split_words(name: str) -> str:
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", name)
    return " ".join(p.lower() for p in parts)


def _clean_doc(doc: str) -> str | None:
    m = re.search(r"@description\s+(.+?)(?:\s@\w+|$)", doc)
    text = (m.group(1) if m else doc).strip()
    text = re.sub(r"\s+", " ", text).rstrip(".")
    if len(text) < 25 or len(text) > 160 or "@" in text:
        return None
    return text


def _service_names(conn) -> set[str]:
    return {
        r["name"].lower() for r in conn.execute("SELECT name FROM catalog WHERE kind = 'service'")
    }


def _doc_contradicts_path(doc: str, path: str, services: set[str]) -> bool:
    """A JSDoc that talks about *another* integration's device/service and never names its own
    was copy-pasted (e.g. "Poll values of an Tuya device" on services/melcloud/…/melcloud.poll.js,
    "This function stops the Bluetooth service." on services/lan-manager/index.js)."""
    m = re.match(r"server/services/([^/]+)/", path)
    if not m:
        return False
    own = m.group(1).lower()
    low = doc.lower()
    if own in low or own.replace("-", " ") in low:
        return False
    for svc in services:
        if svc == own or svc in own or own in svc:
            continue
        if re.search(rf"\b{re.escape(svc)}\s+(device|devices|service|integration)\b", low):
            return True
    return False


def semantic(conn, rng: random.Random, n: int) -> list[dict]:
    rows = conn.execute(
        "SELECT s.id, s.name, s.qualname, s.start_line, s.end_line, s.doc, f.path FROM symbols s"
        " JOIN files f ON f.id = s.file_id WHERE s.doc LIKE '%@description%' AND s.kind IN"
        " ('function','method','arrow') AND f.path LIKE 'server/%'"
        " AND (s.end_line - s.start_line) BETWEEN 6 AND 80 ORDER BY f.path, s.start_line"
    ).fetchall()
    by_dir: dict[str, list] = defaultdict(list)
    seen_docs: set[str] = set()
    services = _service_names(conn)
    for r in rows:
        doc = _clean_doc(r["doc"] or "")
        if not doc or doc.lower() in seen_docs:
            continue
        if _doc_contradicts_path(doc, r["path"], services):
            continue  # copy-pasted JSDoc → its label would be wrong
        seen_docs.add(doc.lower())
        by_dir["/".join(r["path"].split("/")[:3])].append((r, doc))
    dirs = sorted(by_dir)
    rng.shuffle(dirs)
    out = []
    for d in dirs:
        if len(out) >= n:
            break
        r, doc = rng.choice(by_dir[d])
        out.append(
            {
                "id": f"sem-{len(out) + 1:02d}",
                "type": "semantic",
                "set": "original",
                "query": f"Find the code that does this: {doc}",
                "gold": [
                    {
                        "file_path": r["path"],
                        "start_line": r["start_line"],
                        "end_line": r["end_line"],
                    }
                ],
                "source": f"JSDoc of {r['qualname']}",
            }
        )
    return out


def structural(conn, rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    per = max(1, n // 3)
    # --- call order: same-function, certain, distinctive project callees -----------------
    rows = conn.execute(
        "SELECT c.caller_id, c.ordinal, c.in_branch, c.in_callback, t.name AS tname FROM calls c"
        " JOIN symbols t ON t.id = c.callee_symbol_id JOIN symbols s ON s.id = c.caller_id"
        " WHERE c.resolution IN ('static','heuristic') AND s.kind != 'module'"
        " ORDER BY c.caller_id, c.ordinal"
    ).fetchall()
    by_fn: dict[int, list] = defaultdict(list)
    for r in rows:
        by_fn[r["caller_id"]].append(r)
    pairs: set[tuple[str, str]] = set()
    for calls in by_fn.values():
        names = [c for c in calls if len(c["tname"]) >= 7 and c["tname"] not in GENERIC]
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if a["tname"] != b["tname"] and not a["in_branch"] and not a["in_callback"]:
                    pairs.add((a["tname"], b["tname"]))
    pairs_l = sorted(pairs)
    rng.shuffle(pairs_l)
    seq = []
    for a, b in pairs_l:
        res = find_call_sequence(conn, a, b, same_function=True)
        if 1 <= res["total"] <= 6:
            seq.append(
                {
                    "id": "",
                    "type": "structural",
                    "subtype": "sequence",
                    "query": f"Which files call {a} before {b}?",
                    "gold": [
                        {
                            "file_path": m["file_path"],
                            "start_line": m["start_line"],
                            "end_line": m["end_line"],
                        }
                        for m in res["matches"]
                    ],
                    "source": "find_call_sequence(same_function=True)",
                }
            )
        if len(seq) >= per:
            break
    # --- callers ------------------------------------------------------------------------
    rows = conn.execute(
        "SELECT t.name, COUNT(DISTINCT c.caller_id) n FROM calls c JOIN symbols t ON"
        " t.id = c.callee_symbol_id WHERE c.resolution IN ('static','heuristic') AND"
        " length(t.name) >= 8 GROUP BY t.name HAVING n BETWEEN 2 AND 6 ORDER BY t.name"
    ).fetchall()
    names = [r["name"] for r in rows if r["name"] not in GENERIC]
    rng.shuffle(names)
    callers = []
    for name in names:
        if len(conn.execute("SELECT 1 FROM symbols WHERE name = ?", (name,)).fetchall()) != 1:
            continue  # unique definition keeps the label unambiguous
        res = get_callers(conn, name)
        gold = [
            {"file_path": c["file_path"], "start_line": c["call_line"], "end_line": c["call_line"]}
            for c in res["callers"]
            if not c["resolution"].startswith("name-match")
        ]
        if 2 <= len(gold) <= 8:
            callers.append(
                {
                    "id": "",
                    "type": "structural",
                    "subtype": "callers",
                    "query": f"Who calls {name}?",
                    "gold": gold,
                    "source": "get_callers (resolved edges)",
                }
            )
        if len(callers) >= per:
            break
    # --- callees ------------------------------------------------------------------------
    rows = conn.execute(
        "SELECT s.id, s.name, s.qualname FROM symbols s JOIN calls c ON c.caller_id = s.id"
        " WHERE c.resolution IN ('static','heuristic') AND s.kind IN ('function','method')"
        " AND length(s.name) >= 8 GROUP BY s.id HAVING COUNT(DISTINCT c.callee_symbol_id)"
        " BETWEEN 2 AND 6 ORDER BY s.id"
    ).fetchall()
    cands = [r for r in rows if r["name"] not in GENERIC]
    rng.shuffle(cands)
    callees = []
    for r in cands:
        if len(conn.execute("SELECT 1 FROM symbols WHERE name = ?", (r["name"],)).fetchall()) != 1:
            continue
        res = get_callees(conn, r["name"])
        gold = []
        for c in res["callees"]:
            if c["target_location"]:
                path, span = c["target_location"].rsplit(":", 1)
                a, b = (int(x) for x in span.split("-"))
                item = {"file_path": path, "start_line": a, "end_line": b}
                if item not in gold:
                    gold.append(item)
        if 2 <= len(gold) <= 6:
            callees.append(
                {
                    "id": "",
                    "type": "structural",
                    "subtype": "callees",
                    "query": f"What does {r['name']} call?",
                    "gold": gold,
                    "source": "get_callees (resolved targets)",
                }
            )
        if len(callees) >= n - per * 2:
            break
    out = seq + callers + callees
    for i, q in enumerate(out, 1):
        q["id"] = f"str-{i:02d}"
    return out


def _grep_lines(repo: Path, files: list[str], needle: str) -> list[dict]:
    hits = []
    for path in files:
        try:
            lines = (repo / path).read_text(encoding="utf-8", errors="replace").split("\n")
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if needle in line:
                hits.append({"file_path": path, "start_line": i, "end_line": i})
    return hits


def usage(conn, rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    files = [r["path"] for r in conn.execute("SELECT path FROM files ORDER BY path")]

    def catalog_gold(kind: str, name: str) -> list[dict]:
        return [
            {"file_path": r["path"], "start_line": r["line"], "end_line": r["line"]}
            for r in conn.execute(
                "SELECT DISTINCT f.path, c.line FROM catalog c JOIN files f ON f.id = c.file_id"
                " WHERE c.kind = ? AND c.name = ? AND c.role != 'demo_mock'"
                " ORDER BY f.path, c.line",
                (kind, name),
            )
        ]

    # --- deeplinks (front routes with a page component and ≥ 1 usage) -------------------
    rows = conn.execute(
        "SELECT name, detail FROM catalog WHERE kind = 'front_route' AND role = 'definition'"
        " AND detail LIKE '%Page' ORDER BY name"
    ).fetchall()
    routes = [r for r in rows if 2 <= len(catalog_gold("front_route", r["name"])) <= 10]
    rng.shuffle(routes)
    for r in routes[:4]:
        phrase = _split_words(r["detail"].removesuffix("Page"))
        out.append(
            {
                "id": "",
                "type": "usage",
                "subtype": "deeplink",
                "query": f"Where is the {phrase} page deeplink used?",
                "gold": catalog_gold("front_route", r["name"]),
                "source": f"catalog front_route {r['name']} ({r['detail']})",
            }
        )
    # --- events: 2 by constant name, 2 by phrase ----------------------------------------
    rows = conn.execute(
        "SELECT name, COUNT(*) n FROM catalog WHERE kind = 'event' GROUP BY name"
        " HAVING n BETWEEN 3 AND 10 ORDER BY name"
    ).fetchall()
    events = [r["name"] for r in rows]
    rng.shuffle(events)
    for i, name in enumerate(events[:4]):
        if i < 2:
            q = f"Where is {name} used?"
        else:
            q = (
                f"Where is the {_split_words(name.split('.', 1)[1].replace('_', ' ')).lower()}"
                " event emitted or handled?"
            )
        out.append(
            {
                "id": "",
                "type": "usage",
                "subtype": "event",
                "query": q,
                "gold": catalog_gold("event", name),
                "source": f"catalog event {name}",
            }
        )
    # --- API routes called from the front end -------------------------------------------
    rows = conn.execute(
        "SELECT name FROM catalog WHERE kind = 'api_route' AND role = 'usage' GROUP BY name"
        " HAVING COUNT(*) BETWEEN 1 AND 4 ORDER BY name"
    ).fetchall()
    apis = [r["name"] for r in rows]
    rng.shuffle(apis)
    for name in apis[:2]:
        method, path = name.split(" ", 1)
        out.append(
            {
                "id": "",
                "type": "usage",
                "subtype": "api_route",
                "query": f"Where is the {method} {path} endpoint defined and called?",
                "gold": catalog_gold("api_route", name),
                "source": f"catalog api_route {name}",
            }
        )
    # --- distinctive string literals (grep) ---------------------------------------------
    rows = conn.execute(
        "SELECT value, COUNT(*) n FROM strings WHERE context IN ('argument','property',"
        "'assignment') AND length(value) BETWEEN 12 AND 60 AND value NOT LIKE '%/%' AND value"
        " NOT LIKE '% %' AND value GLOB '*[a-z]*' GROUP BY value HAVING n BETWEEN 2 AND 5"
        " ORDER BY value"
    ).fetchall()
    lits = [r["value"] for r in rows if re.fullmatch(r"[\w.\-:]+", r["value"])]
    rng.shuffle(lits)
    for value in lits:
        if len(out) >= n:
            break
        gold = _grep_lines(config.REPO_ROOT, files, f"'{value}'") + _grep_lines(
            config.REPO_ROOT, files, f'"{value}"'
        )
        if 2 <= len(gold) <= 8:
            out.append(
                {
                    "id": "",
                    "type": "usage",
                    "subtype": "literal",
                    "query": f"Where is the string '{value}' used?",
                    "gold": gold,
                    "source": "grep for the quoted literal",
                }
            )
    for i, q in enumerate(out, 1):
        q["id"] = f"use-{i:02d}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-type", type=int, default=12)
    args = ap.parse_args()
    conn = store.connect(config.DB_PATH, readonly=True)
    rng = random.Random(SEED)
    queries = (
        semantic(conn, rng, args.per_type)
        + structural(conn, rng, args.per_type)
        + usage(conn, rng, args.per_type)
    )
    manifest = json.loads(config.MANIFEST_PATH.read_text())
    doc = {
        "generated_from_commit": manifest.get("repo_commit"),
        "seed": SEED,
        "labels": "auto-derived from AST/call graph/catalog/grep; no manual labelling",
        "queries": queries,
    }
    for q in queries:
        q.setdefault("set", "original")
    OUT.write_text(json.dumps(doc, indent=2))
    counts: dict[str, int] = defaultdict(int)
    for q in queries:
        counts[q["type"]] += 1
    print(f"wrote {len(queries)} queries to {OUT}: {dict(counts)}")


if __name__ == "__main__":
    main()
