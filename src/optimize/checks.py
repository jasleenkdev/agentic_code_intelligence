"""Static optimisation checks on the code paths an answer surfaced (BONUS-1).

Checks (all static, all advisory — nothing is executed):
- ``sequential-awaits``   consecutive independent `await`s in one block → `Promise.all`.
- ``await-in-loop``       an awaited call inside a loop body / iteration callback serialises I/O.
- ``loop-invariant-call`` a call inside a loop whose arguments do not use the loop variable.
- ``duplicated-literal``  a route/URL/config string built in ≥ 3 places → share one constant.
- ``uncalled-function``   a non-exported function with no call sites, bindings or callback uses
                          in the call graph → possibly dead code.

Each finding points at a verified file:line. An optional LLM pass rewrites the messages using only
the retrieved snippets; without an LLM the template messages are returned as-is.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

from tree_sitter import Node

from src import config
from src.indexer import store
from src.indexer.extractor import LOOP_TYPES, make_parser

logger = logging.getLogger(__name__)

IO_NAME_RE = re.compile(r"(get|find|fetch|query|load|read|request|select|count|call)", re.I)
LITERAL_RE = re.compile(r"^(/[\w\-:.${}]+){2,}$|^[a-z]+://|^[a-z][\w-]*(\.[a-z][\w-]*){2,}$")
MAX_FINDINGS = 12


def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _await_call(stmt: Node) -> tuple[Node, list[str]] | None:
    """If `stmt` is `await f()` or `const x = await f()`, return (await node, declared names)."""
    if stmt.type == "expression_statement" and stmt.named_children:
        expr = stmt.named_children[0]
        if expr.type == "await_expression":
            return expr, []
        if expr.type == "assignment_expression":
            right = expr.child_by_field_name("right")
            left = expr.child_by_field_name("left")
            if right is not None and right.type == "await_expression" and left is not None:
                return right, re.findall(r"[A-Za-z_$][\w$]*", left.text.decode())
    if stmt.type in {"lexical_declaration", "variable_declaration"}:
        for decl in stmt.named_children:
            value = decl.child_by_field_name("value")
            name = decl.child_by_field_name("name")
            if value is not None and value.type == "await_expression" and name is not None:
                return value, re.findall(r"[A-Za-z_$][\w$]*", name.text.decode())
    return None


def _loop_vars(loop: Node, src: bytes) -> set[str]:
    names: set[str] = set()
    for field in ("left", "initializer", "init"):
        n = loop.child_by_field_name(field)
        if n is not None:
            names |= set(re.findall(r"[A-Za-z_$][\w$]*", _text(n, src)))
    return names - {"const", "let", "var", "of", "in"}


def _function_findings(src: bytes, fn: Node, base_line: int, path: str, symbol: str) -> list[dict]:
    findings: list[dict] = []
    stack = [fn]
    while stack:
        node = stack.pop()
        stack.extend(node.named_children)
        # --- sequential independent awaits within one block --------------------------------
        if node.type == "statement_block":
            stmts = node.named_children
            for a, b in zip(stmts, stmts[1:], strict=False):
                aa, ba = _await_call(a), _await_call(b)
                if not aa or not ba:
                    continue
                declared = aa[1]
                b_text = _text(b, src)
                if declared and any(re.search(rf"\b{re.escape(n)}\b", b_text) for n in declared):
                    continue  # the second await uses the first result
                findings.append(
                    {
                        "check": "sequential-awaits",
                        "line": a.start_point[0] + 1,
                        "end_line": b.end_point[0] + 1,
                        "message": "Two independent awaits run one after the other; if neither "
                        "depends on the other's side effects they can run concurrently with "
                        "Promise.all([...]).",
                    }
                )
        # --- loops ------------------------------------------------------------------------
        loop_body = None
        loop_vars: set[str] = set()
        if node.type in LOOP_TYPES:
            loop_body = node.child_by_field_name("body")
            loop_vars = _loop_vars(node, src)
        elif node.type == "call_expression":
            fnode = node.child_by_field_name("function")
            args = node.child_by_field_name("arguments")
            if (
                fnode is not None
                and fnode.type == "member_expression"
                and args is not None
                and args.named_children
            ):
                prop = fnode.child_by_field_name("property")
                cb = args.named_children[0]
                if (
                    prop is not None
                    and _text(prop, src) in {"forEach", "map"}
                    and cb.type in {"arrow_function", "function_expression", "function"}
                ):
                    loop_body = cb.child_by_field_name("body")
                    params = cb.child_by_field_name("parameters") or cb.child_by_field_name(
                        "parameter"
                    )
                    loop_vars = (
                        set(re.findall(r"[A-Za-z_$][\w$]*", _text(params, src)))
                        if params is not None
                        else set()
                    )
        if loop_body is None:
            continue
        inner = [loop_body]
        while inner:
            n = inner.pop()
            if (
                n.type in {"function_declaration", "arrow_function", "function_expression"}
                and n is not loop_body.parent
            ):
                continue
            inner.extend(n.named_children)
            if n.type == "await_expression":
                findings.append(
                    {
                        "check": "await-in-loop",
                        "line": n.start_point[0] + 1,
                        "end_line": n.end_point[0] + 1,
                        "message": "An awaited call inside a loop runs iterations sequentially; "
                        "if iterations are independent, collect the promises and await "
                        "Promise.all (or batch with a concurrency limit).",
                    }
                )
            if n.type == "call_expression" and loop_vars:
                fnode = n.child_by_field_name("function")
                args = n.child_by_field_name("arguments")
                if fnode is None or args is None:
                    continue
                ftext, atext = _text(fnode, src), _text(args, src)
                idents = set(re.findall(r"[A-Za-z_$][\w$]*", ftext + atext))
                if (idents & loop_vars) or not args.named_children:
                    continue
                if IO_NAME_RE.search(ftext.split(".")[-1]):
                    findings.append(
                        {
                            "check": "loop-invariant-call",
                            "line": n.start_point[0] + 1,
                            "end_line": n.end_point[0] + 1,
                            "message": f"`{ftext[:60]}{atext[:40]}` does not depend on the loop "
                            "variable; compute it once before the loop.",
                        }
                    )
    for f in findings:
        f.update({"file_path": path, "symbol": symbol})
    return findings


def _duplicated_literals(conn: sqlite3.Connection, path: str, start: int, end: int) -> list[dict]:
    out = []
    rows = conn.execute(
        "SELECT s.value, s.line FROM strings s JOIN files f ON f.id = s.file_id WHERE f.path = ?"
        " AND s.line BETWEEN ? AND ? AND s.context NOT IN ('import', 'key')",
        (path, start, end),
    ).fetchall()
    seen = set()
    for r in rows:
        v = r["value"].strip()
        if v in seen or len(v) < 8 or not LITERAL_RE.search(v):
            continue
        seen.add(v)
        n_files = conn.execute(
            "SELECT COUNT(DISTINCT file_id) FROM strings WHERE value = ?", (v,)
        ).fetchone()[0]
        if n_files >= 3:
            out.append(
                {
                    "check": "duplicated-literal",
                    "file_path": path,
                    "line": r["line"],
                    "end_line": r["line"],
                    "symbol": None,
                    "message": f"The literal '{v[:70]}' is hard-coded in {n_files} files; define "
                    "it once (a route/config constant) so a change cannot miss a copy.",
                }
            )
    return out


def _uncalled(conn: sqlite3.Connection, symbol_id: int) -> dict | None:
    s = conn.execute(
        "SELECT s.name, s.kind, s.start_line, s.end_line, s.is_exported, f.path, p.kind AS pk"
        " FROM symbols s JOIN files f ON f.id = s.file_id LEFT JOIN symbols p ON p.id ="
        " s.parent_id WHERE s.id = ?",
        (symbol_id,),
    ).fetchone()
    if s is None or s["kind"] not in {"function", "arrow"} or s["is_exported"]:
        return None
    name = s["name"]
    used = (
        conn.execute(
            "SELECT 1 FROM calls WHERE callee_symbol_id = ? OR callee_name = ? LIMIT 1",
            (symbol_id, name),
        ).fetchone()
        or conn.execute(
            "SELECT 1 FROM bindings WHERE target = ? OR target_symbol_id = ? LIMIT 1",
            (name, symbol_id),
        ).fetchone()
        or conn.execute(
            "SELECT 1 FROM calls WHERE args_text LIKE ? LIMIT 1", (f"%{name}%",)
        ).fetchone()
        or conn.execute("SELECT 1 FROM exports WHERE local_name = ? LIMIT 1", (name,)).fetchone()
    )
    if used:
        return None
    return {
        "check": "uncalled-function",
        "file_path": s["path"],
        "line": s["start_line"],
        "end_line": s["end_line"],
        "symbol": name,
        "message": f"`{name}` has no call sites, bindings, exports or callback uses in the "
        "call graph; it may be dead code (verify dynamic uses before deleting).",
    }


def static_findings(
    snippets: list, conn: sqlite3.Connection | None = None, repo_root: Path | None = None
) -> list[dict]:
    """Run all checks on the functions enclosing the surfaced snippets."""
    conn = conn or store.connect(config.DB_PATH, readonly=True)
    repo_root = repo_root or config.REPO_ROOT
    parser = make_parser()
    findings: list[dict] = []
    done: set[int] = set()
    for sn in snippets:
        sym = conn.execute(
            "SELECT s.id, s.qualname, s.start_line, s.end_line FROM symbols s JOIN files f ON"
            " f.id = s.file_id WHERE f.path = ? AND s.kind IN ('function','arrow','method') AND"
            " s.start_line <= ? AND s.end_line >= ? ORDER BY (s.end_line - s.start_line) DESC"
            " LIMIT 1",
            (sn.file_path, sn.start_line, sn.start_line),
        ).fetchone()
        findings += _duplicated_literals(conn, sn.file_path, sn.start_line, sn.end_line)
        if sym is None or sym["id"] in done:
            continue
        done.add(sym["id"])
        try:
            src = (repo_root / sn.file_path).read_bytes()
        except OSError:
            continue
        root = parser.parse(src).root_node
        target = root.descendant_for_point_range(
            (sym["start_line"] - 1, 0), (sym["end_line"] - 1, 0)
        )
        while target is not None and not (
            target.start_point[0] + 1 == sym["start_line"]
            and target.end_point[0] + 1 == sym["end_line"]
        ):
            target = target.parent
        if target is not None:
            findings += _function_findings(
                src, target, sym["start_line"], sn.file_path, sym["qualname"]
            )
        dead = _uncalled(conn, sym["id"])
        if dead:
            findings.append(dead)
    # Deduplicate and attach verified evidence lines.
    from src.agent.verifier import verify

    out, seen = [], set()
    for f in findings:
        key = (f["check"], f["file_path"], f["line"])
        if key in seen:
            continue
        seen.add(key)
        sn = verify(
            f["file_path"], f["line"], min(f["end_line"], f["line"] + 6), repo_root=repo_root
        )
        if sn is None:
            continue
        f["location"] = f"{f['file_path']}:{f['line']}"
        f["evidence"] = sn.text
        f["advisory"] = True
        out.append(f)
        if len(out) >= MAX_FINDINGS:
            break
    return out


EXPLAIN_PROMPT = (
    "You review JavaScript for performance and maintainability. For each numbered finding, "
    "write one or two sentences explaining the concrete fix, using ONLY the code shown. If the "
    "code shows the finding is wrong, say 'not applicable'. Reply as JSON "
    '{"explanations": ["...", "..."]} with one entry per finding, in order.\n'
)


def explain_findings(findings: list[dict], llm) -> int:
    """LLM rewrites up to 6 findings' messages from the shown code only. Returns how many got an
    explanation (0 without an LLM). Called after the first results are shown (agent phase B)."""
    if not findings or llm is None or not llm.available():
        return 0
    from src.agent.llm import LLMUnavailable, parse_json_object

    body = "\n".join(
        f"{i}. [{f['check']}] {f['location']}: {f['message']}\n```js\n{f['evidence'][:400]}\n```"
        for i, f in enumerate(findings[:6], 1)
    )
    done = 0
    try:
        resp = llm.chat([{"role": "user", "content": EXPLAIN_PROMPT + body}], max_tokens=350)
        exps = parse_json_object(resp.text).get("explanations", [])
        for f, e in zip(findings, exps, strict=False):
            if isinstance(e, str) and e.strip():
                f["explanation"] = e.strip()
                done += 1
    except (LLMUnavailable, ValueError) as exc:
        logger.info("optimisation explanation skipped: %s", exc)
    return done


def suggest_optimizations(snippets: list, llm=None) -> list[dict]:
    """Static findings, optionally explained by the LLM in the same call."""
    findings = static_findings(snippets)
    explain_findings(findings, llm)
    return findings
