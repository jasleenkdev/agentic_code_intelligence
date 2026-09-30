"""Deterministic call-graph queries: symbol lookup, callers, callees, ordered call sequences.

Everything here is computed from the SQLite index; the LLM never supplies these facts (D-002).
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass

PROJECT_RESOLUTIONS = ("static", "heuristic")


def _sym_dict(r: sqlite3.Row) -> dict:
    return {
        "symbol_id": r["id"],
        "name": r["name"],
        "qualname": r["qualname"],
        "kind": r["kind"],
        "file_path": r["path"],
        "start_line": r["start_line"],
        "end_line": r["end_line"],
    }


def find_symbols(conn: sqlite3.Connection, name: str, kind: str | None = None) -> list[dict]:
    """Symbols matching `name` (module symbols excluded).

    `foo` matches by name. `Owner.foo` matches qualname `Owner.foo`, a qualname ending in `.foo`
    whose owner resembles `Owner`, or a function bound as `Owner.prototype.foo = foo`.
    """
    name = name.strip().removesuffix("()")
    bare = name.split(".")[-1]
    rows = conn.execute(
        "SELECT s.*, f.path FROM symbols s JOIN files f ON f.id = s.file_id"
        " WHERE (s.name = ? OR s.qualname = ?) AND s.kind != 'module'"
        " ORDER BY f.path, s.start_line",
        (bare, name),
    ).fetchall()
    bound: set[int] = set()
    owner = ""
    if "." in name:
        owner = name.rsplit(".", 1)[0].split(".")[-1].lower()
        for b in conn.execute(
            "SELECT owner, target_symbol_id FROM bindings WHERE member = ? AND"
            " target_symbol_id IS NOT NULL",
            (bare,),
        ):
            if owner and owner in b["owner"].lower():
                bound.add(b["target_symbol_id"])
    out = []
    for r in rows:
        if owner and r["qualname"] != name and r["id"] not in bound:
            prefix = r["qualname"].rsplit(".", 1)[0].lower() if "." in r["qualname"] else ""
            if not prefix or (owner not in prefix and prefix not in owner):
                continue
        if kind and r["kind"] != kind:
            continue
        out.append(_sym_dict(r))
    return out


def _call_dict(r: sqlite3.Row) -> dict:
    return {
        "call_id": r["id"],
        "callee_text": r["callee_text"],
        "callee_name": r["callee_name"],
        "line": r["line"],
        "ordinal": r["ordinal"],
        "awaited": bool(r["awaited"]),
        "in_branch": bool(r["in_branch"]),
        "in_callback": bool(r["in_callback"]),
        "in_loop": bool(r["in_loop"]),
        "first_arg": r["first_arg"],
        "resolution": r["resolution"],
        "callee_symbol_id": r["callee_symbol_id"],
    }


def get_callers(conn: sqlite3.Connection, symbol: str, depth: int = 1) -> dict:
    """Call sites that invoke `symbol` (resolved edges first, then unresolved name matches)."""
    depth = max(1, min(depth, 3))
    targets = find_symbols(conn, symbol)
    bare = symbol.strip().removesuffix("()").split(".")[-1]
    results: list[dict] = []
    seen_calls: set[int] = set()
    frontier = {t["symbol_id"] for t in targets}
    for level in range(1, depth + 1):
        if not frontier:
            break
        q = (
            "SELECT c.*, s.id AS sid, s.name, s.qualname, s.kind, s.start_line, s.end_line,"
            " f.path, t.qualname AS target FROM calls c JOIN symbols s ON s.id = c.caller_id"
            " JOIN files f ON f.id = c.file_id JOIN symbols t ON t.id = c.callee_symbol_id"
            f" WHERE c.callee_symbol_id IN ({','.join('?' * len(frontier))})"
            " ORDER BY f.path, c.line"
        )
        next_frontier: set[int] = set()
        for r in conn.execute(q, list(frontier)):
            if r["id"] in seen_calls:
                continue
            seen_calls.add(r["id"])
            results.append(
                {
                    "depth": level,
                    "caller": r["qualname"],
                    "caller_kind": r["kind"],
                    "file_path": r["path"],
                    "caller_start_line": r["start_line"],
                    "caller_end_line": r["end_line"],
                    "call_line": r["line"],
                    "call_text": r["callee_text"],
                    "calls": r["target"],
                    "resolution": r["resolution"],
                }
            )
            if r["kind"] != "module":
                next_frontier.add(r["sid"])
        frontier = next_frontier
    # Unresolved call sites with the same bare name: possible callers (dynamic dispatch).
    for r in conn.execute(
        "SELECT c.*, s.qualname, s.kind, s.start_line, s.end_line, f.path FROM calls c"
        " JOIN symbols s ON s.id = c.caller_id JOIN files f ON f.id = c.file_id"
        " WHERE c.callee_name = ? AND c.resolution = 'unresolved' ORDER BY f.path, c.line"
        " LIMIT 200",
        (bare,),
    ):
        results.append(
            {
                "depth": 1,
                "caller": r["qualname"],
                "caller_kind": r["kind"],
                "file_path": r["path"],
                "caller_start_line": r["start_line"],
                "caller_end_line": r["end_line"],
                "call_line": r["line"],
                "call_text": r["callee_text"],
                "calls": bare,
                "resolution": "name-match (unresolved)",
            }
        )
    return {"targets": targets, "callers": results}


def get_callees(conn: sqlite3.Connection, symbol: str, depth: int = 1) -> dict:
    """Project calls made by `symbol` (externals like logger/array methods counted, not listed)."""
    depth = max(1, min(depth, 3))
    sources = find_symbols(conn, symbol)
    results: list[dict] = []
    external = 0
    frontier = {s["symbol_id"] for s in sources}
    visited = set(frontier)
    for level in range(1, depth + 1):
        if not frontier:
            break
        q = (
            "SELECT c.*, s.qualname AS caller, f.path, t.qualname AS target, t.start_line AS"
            " t_start, t.end_line AS t_end, tf.path AS t_path FROM calls c JOIN symbols s ON"
            " s.id = c.caller_id JOIN files f ON f.id = c.file_id LEFT JOIN symbols t ON"
            " t.id = c.callee_symbol_id LEFT JOIN files tf ON tf.id = t.file_id"
            f" WHERE c.caller_id IN ({','.join('?' * len(frontier))}) ORDER BY c.caller_id,"
            " c.ordinal"
        )
        next_frontier: set[int] = set()
        for r in conn.execute(q, list(frontier)):
            if r["resolution"] == "external":
                external += 1
                continue
            results.append(
                {
                    "depth": level,
                    "caller": r["caller"],
                    "file_path": r["path"],
                    "line": r["line"],
                    "ordinal": r["ordinal"],
                    "call_text": r["callee_text"],
                    "awaited": bool(r["awaited"]),
                    "in_branch": bool(r["in_branch"]),
                    "resolution": r["resolution"],
                    "target": r["target"],
                    "target_location": (
                        f"{r['t_path']}:{r['t_start']}-{r['t_end']}" if r["t_path"] else None
                    ),
                }
            )
            tid = r["callee_symbol_id"]
            if tid is not None and tid not in visited:
                visited.add(tid)
                next_frontier.add(tid)
        frontier = next_frontier
    return {"sources": sources, "callees": results, "external_calls_omitted": external}


# ---------------------------------------------------------------------- call sequences


@dataclass
class _Occ:
    """One (possibly transitive) occurrence of a matching call inside an enclosing function."""

    pos: tuple[int, ...]  # ordinal path: (ordinal in F, ordinal in G, ...)
    line: int  # line of the call site *in F* (the hop entry point for transitive ones)
    call_line: int  # line of the actual matching call
    call_file: str
    call_text: str
    via: list[str]  # qualnames of intermediate functions
    uncertain: bool  # branch/callback/heuristic-hop involvement
    reasons: list[str]


def _matches(call: sqlite3.Row, name: str, target_names: dict[int, tuple[str, str]]) -> bool:
    arg = None
    if "(" in name and name.endswith(")"):
        name, arg = name[:-1].split("(", 1)
        arg = arg.strip().strip("'\"") or None
    name = name.strip()
    bare = name.split(".")[-1]
    ok = (
        call["callee_text"] == name
        or call["callee_name"] == name
        or call["callee_text"].endswith("." + name)
    )
    if not ok and call["callee_symbol_id"] in target_names:
        tname, tqual = target_names[call["callee_symbol_id"]]
        ok = tqual == name or (tname == bare and "." not in name)
    if ok and arg:
        ok = bool(call["first_arg"]) and arg in call["first_arg"]
    return ok


class _SequenceFinder:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.calls_by_caller: dict[int, list[sqlite3.Row]] = {}
        self.sym: dict[int, sqlite3.Row] = {}

    def symbol(self, sid: int) -> sqlite3.Row:
        if sid not in self.sym:
            self.sym[sid] = self.conn.execute(
                "SELECT s.*, f.path FROM symbols s JOIN files f ON f.id = s.file_id WHERE s.id = ?",
                (sid,),
            ).fetchone()
        return self.sym[sid]

    def calls(self, sid: int) -> list[sqlite3.Row]:
        if sid not in self.calls_by_caller:
            self.calls_by_caller[sid] = self.conn.execute(
                "SELECT * FROM calls WHERE caller_id = ? ORDER BY ordinal", (sid,)
            ).fetchall()
        return self.calls_by_caller[sid]

    def occurrences(
        self, sid: int, name: str, tnames: dict, hops: int, stack: tuple[int, ...] = ()
    ) -> list[_Occ]:
        out: list[_Occ] = []
        path = self.symbol(sid)["path"]
        for c in self.calls(sid):
            reasons = []
            if c["in_branch"]:
                reasons.append(f"line {c['line']} is inside a conditional branch")
            if c["in_callback"]:
                reasons.append(f"line {c['line']} is inside a callback")
            if _matches(c, name, tnames):
                out.append(
                    _Occ(
                        (c["ordinal"],),
                        c["line"],
                        c["line"],
                        path,
                        c["callee_text"],
                        [],
                        bool(reasons),
                        reasons,
                    )
                )
            tid = c["callee_symbol_id"]
            if (
                hops > 0
                and tid is not None
                and c["resolution"] in PROJECT_RESOLUTIONS
                and tid not in stack
                and tid != sid
            ):
                if c["resolution"] == "heuristic":
                    reasons = reasons + [f"hop via {c['callee_text']} is a heuristic resolution"]
                target = self.symbol(tid)
                for sub in self.occurrences(tid, name, tnames, hops - 1, stack + (sid,)):
                    out.append(
                        _Occ(
                            (c["ordinal"],) + sub.pos,
                            c["line"],
                            sub.call_line,
                            sub.call_file,
                            sub.call_text,
                            [target["qualname"]] + sub.via,
                            sub.uncertain or bool(reasons),
                            reasons + sub.reasons,
                        )
                    )
        return out


def _callers_of_matching(conn: sqlite3.Connection, name: str) -> set[int]:
    bare = name.split("(")[0].strip().split(".")[-1]
    rows = conn.execute(
        "SELECT DISTINCT c.caller_id FROM calls c LEFT JOIN symbols t ON t.id ="
        " c.callee_symbol_id WHERE c.callee_name = ? OR t.name = ?",
        (bare, bare),
    ).fetchall()
    return {r["caller_id"] for r in rows}


def _expand_callers(conn: sqlite3.Connection, ids: set[int], hops: int) -> set[int]:
    out = set(ids)
    frontier = set(ids)
    for _ in range(hops):
        if not frontier:
            break
        q = (
            "SELECT DISTINCT caller_id FROM calls WHERE resolution IN ('static','heuristic') AND"
            f" callee_symbol_id IN ({','.join('?' * len(frontier))})"
        )
        new = {r["caller_id"] for r in conn.execute(q, list(frontier))} - out
        out |= new
        frontier = new
    return out


def find_call_sequence(
    conn: sqlite3.Connection,
    first: str,
    second: str,
    same_function: bool = True,
    max_hops: int = 2,
    limit: int = 100,
) -> dict:
    """Functions in which a call to `first` happens before a call to `second`.

    same_function=True: both calls directly in the same function body, ordered by ordinal.
    same_function=False: also follows up to `max_hops` resolved call-graph hops (F calls G, G calls
    `second`), comparing positions lexicographically. Each result carries `certainty`:
    "certain" only when the first call is unconditional (not in a branch/callback), the second is
    not in a callback, and no heuristic hop is involved; otherwise "possible" with reasons.
    """
    hops = 0 if same_function else max(0, min(max_hops, 2))
    tnames = {
        r["id"]: (r["name"], r["qualname"])
        for r in conn.execute("SELECT id, name, qualname FROM symbols WHERE kind != 'module'")
    }
    cand = _expand_callers(conn, _callers_of_matching(conn, first), hops) & _expand_callers(
        conn, _callers_of_matching(conn, second), hops
    )
    finder = _SequenceFinder(conn)
    results = []
    for sid in sorted(cand):
        firsts = finder.occurrences(sid, first, tnames, hops)
        if not firsts:
            continue
        seconds = finder.occurrences(sid, second, tnames, hops)
        best = None
        for a in firsts:
            for b in seconds:
                if a.pos < b.pos and not (b.pos[: len(a.pos)] == a.pos):
                    uncertain = (
                        a.uncertain
                        or any("callback" in r for r in b.reasons)
                        or any("heuristic" in r for r in b.reasons)
                    )
                    hop_count = max(len(a.via), len(b.via))
                    key = (uncertain, hop_count, a.pos, b.pos)
                    if best is None or key < best[0]:
                        best = (key, a, b, uncertain, hop_count)
        if best is None:
            continue
        _, a, b, uncertain, hop_count = best
        s = finder.symbol(sid)
        reasons = a.reasons + [r for r in b.reasons if "callback" in r or "heuristic" in r]
        results.append(
            {
                "function": s["qualname"],
                "kind": s["kind"],
                "file_path": s["path"],
                "start_line": s["start_line"],
                "end_line": s["end_line"],
                "first_call": {
                    "text": a.call_text,
                    "line": a.call_line,
                    "file_path": a.call_file,
                    "entry_line": a.line,
                    "via": a.via,
                },
                "second_call": {
                    "text": b.call_text,
                    "line": b.call_line,
                    "file_path": b.call_file,
                    "entry_line": b.line,
                    "via": b.via,
                },
                "hops": hop_count,
                "certainty": "possible" if uncertain else "certain",
                "reasons": list(dict.fromkeys(reasons)),
            }
        )
    results.sort(key=lambda r: (r["hops"], r["certainty"] != "certain", r["file_path"]))
    by_file: dict[str, int] = defaultdict(int)
    for r in results:
        by_file[r["file_path"]] += 1
    return {
        "first": first,
        "second": second,
        "same_function": same_function,
        "matches": results[:limit],
        "total": len(results),
        "files": dict(by_file),
    }
