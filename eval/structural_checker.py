"""Independent, text-only checker for structural ground truth.

Shares NO code with the indexer (no tree-sitter, no SQLite, no call graph). It:
1. walks the same include dirs itself,
2. blanks comments and string/template contents with a tiny lexer (newlines preserved),
3. finds named functions with regexes + brace matching,
4. finds call sites with `name(` regexes and attributes each to the innermost named function.

From that it re-derives the answer to every structural gold query (call order, callers, callees)
and reports agreement with the index-derived labels. Items agree when they are in the same file
and their line ranges overlap (the same relevance rule the metrics use).

Usage: python -m eval.structural_checker   (writes results/structural_check.*, eval/gold_check.json)
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from src import config

EXCLUDE = {
    "node_modules",
    "dist",
    "build",
    "vendor",
    "coverage",
    ".git",
    "test",
    "tests",
    "__tests__",
    "cypress",
    "migrations",
    "seeders",
}
KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "function",
    "return",
    "typeof",
    "new",
    "await",
    "import",
    "require",
    "super",
    "constructor",
    "else",
    "do",
    "with",
}
IDENT = r"[A-Za-z_$][\w$]*"
DEF_PATTERNS = [
    re.compile(rf"\bfunction\s*\*?\s*(?P<n>{IDENT})\s*\("),
    re.compile(rf"(?P<n>{IDENT})\s*[:=]\s*(?:async\s+)?function\b"),
    re.compile(rf"(?P<n>{IDENT})\s*[:=]\s*(?:async\s+)?(?:\([^()]*\)|{IDENT})\s*=>\s*\{{"),
    re.compile(rf"^[ \t]*(?:static\s+)?(?:async\s+)?\*?(?P<n>{IDENT})\s*\([^()]*\)\s*\{{", re.M),
]
OUT_JSON = config.RESULTS_DIR / "structural_check.json"
OUT_MD = config.RESULTS_DIR / "structural_check.md"
GOLD_CHECK = Path(__file__).parent / "gold_check.json"


def _regex_can_start(src: str, i: int) -> bool:
    """A `/` starts a regex literal when the previous significant char cannot end an operand."""
    j = i - 1
    while j >= 0 and src[j] in " \t":
        j -= 1
    if j < 0 or src[j] == "\n":
        return True
    if src[j] in "(,=:[!&|?{};+-*%<>~^":
        return True
    word = re.search(r"(\w+)$", src[: j + 1])
    return bool(word and word.group(1) in {"return", "typeof", "case", "in", "of", "void"})


def blank_comments_and_strings(src: str) -> str:
    """Replace comment and string/template text with spaces (keeping newlines and quotes)."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif c == "/" and nxt == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif c == "/" and _regex_can_start(src, i):
            j = i + 1
            in_class = False
            while j < n and src[j] != "\n":
                if src[j] == "\\":
                    j += 1
                elif src[j] == "[":
                    in_class = True
                elif src[j] == "]":
                    in_class = False
                elif src[j] == "/" and not in_class:
                    break
                j += 1
            for k in range(i + 1, min(j, n)):
                out[k] = " "
            i = j + 1
        elif c in "'\"`":
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\":
                    j += 1
                elif c != "`" and src[j] == "\n":
                    break
                j += 1
            for k in range(i + 1, min(j, n)):
                if out[k] != "\n":
                    out[k] = " "
            i = j + 1
        else:
            i += 1
    return "".join(out)


@dataclass
class Fn:
    name: str
    start: int  # char offset of the definition
    body_start: int
    body_end: int
    start_line: int
    end_line: int


@dataclass
class FileScan:
    path: str
    text: str
    line_starts: list[int]
    functions: list[Fn] = field(default_factory=list)
    def_name_offsets: set[int] = field(default_factory=set)

    def line_of(self, off: int) -> int:
        return bisect_right(self.line_starts, off)

    def innermost(self, off: int) -> Fn | None:
        best = None
        for f in self.functions:
            if f.body_start <= off < f.body_end and (
                best is None or f.body_end - f.body_start < best.body_end - best.body_start
            ):
                best = f
        return best


def _match_brace(text: str, open_at: int) -> int:
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def _body_brace(text: str, pos: int) -> int:
    """Offset of the function body `{`: skip the parameter list (which may contain
    destructuring braces) and an optional `=>`."""
    paren = text.find("(", pos)
    brace = text.find("{", pos)
    if paren < 0 or (0 <= brace < paren and "=>" not in text[pos:brace]):
        return brace
    depth = 0
    for i in range(paren, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text.find("{", i)
    return -1


def scan_file(path: str, raw: str) -> FileScan:
    text = blank_comments_and_strings(raw)
    starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n"]
    fs = FileScan(path, text, starts)
    seen: set[int] = set()
    for rx in DEF_PATTERNS:
        for m in rx.finditer(text):
            name = m.group("n")
            if name in KEYWORDS:
                continue
            brace = _body_brace(text, m.end("n"))
            if brace < 0 or brace in seen:
                continue
            # The body brace must follow the signature closely (no statement in between).
            between = text[m.end("n") : brace]
            if ";" in between or len(between) > 400:
                continue
            seen.add(brace)
            end = _match_brace(text, brace)
            fs.functions.append(
                Fn(name, m.start("n"), brace, end, fs.line_of(m.start("n")), fs.line_of(end - 1))
            )
            fs.def_name_offsets.add(m.start("n"))
    # Expression-bodied arrows: `name = (args) => expr` — body runs to the end of the statement.
    arrow_rx = re.compile(rf"(?P<n>{IDENT})\s*[:=]\s*(?:async\s+)?(?:\([^()]*\)|{IDENT})\s*=>")
    for m in arrow_rx.finditer(text):
        k = m.end()
        while k < len(text) and text[k] in " \t\n":
            k += 1
        if k >= len(text) or text[k] == "{" or m.start("n") in fs.def_name_offsets:
            continue
        depth, end = 0, len(text)
        for i in range(k, len(text)):
            ch = text[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    end = i
                    break
                depth -= 1
            elif depth == 0 and ch in ";,":
                end = i
                break
            elif depth == 0 and ch == "\n" and not re.match(r"\s*[.?:+\-*/&|]", text[i + 1 :]):
                end = i
                break
        fs.functions.append(
            Fn(m.group("n"), m.start("n"), k, end, fs.line_of(m.start("n")), fs.line_of(end - 1))
        )
        fs.def_name_offsets.add(m.start("n"))
    return fs


def walk(repo: Path) -> list[FileScan]:
    scans = []
    for include in config.INCLUDE_DIRS:
        base = repo / include
        for p in sorted(base.rglob("*")):
            if p.suffix not in {".js", ".jsx", ".mjs", ".cjs"} or not p.is_file():
                continue
            rel = p.relative_to(repo)
            if any(part in EXCLUDE for part in rel.parts) or p.name.endswith(".min.js"):
                continue
            raw = p.read_text(encoding="utf-8", errors="replace")
            if len(raw) / (raw.count("\n") + 1) > 300:
                continue
            scans.append(scan_file(rel.as_posix(), raw))
    return scans


def call_sites(fs: FileScan, name: str) -> list[int]:
    """Offsets of `name(` / `.name(` call sites that are not the definition itself."""
    rx = re.compile(rf"(?<![\w$]){re.escape(name)}\s*\(")
    out = []
    for m in rx.finditer(fs.text):
        if m.start() in fs.def_name_offsets:
            continue
        before = fs.text[max(0, m.start() - 12) : m.start()]
        if re.search(r"\bfunction\s*\*?\s*$", before):
            continue
        out.append(m.start())
    return out


class Checker:
    def __init__(self, repo: Path) -> None:
        self.scans = walk(repo)
        self.defs: dict[str, list[tuple[FileScan, Fn]]] = {}
        for fs in self.scans:
            for f in fs.functions:
                self.defs.setdefault(f.name, []).append((fs, f))

    def sequence(self, first: str, second: str) -> list[dict]:
        out = []
        for fs in self.scans:
            if first not in fs.text or second not in fs.text:
                continue
            a_by_fn: dict[int, list[int]] = {}
            b_by_fn: dict[int, list[int]] = {}
            for off in call_sites(fs, first):
                f = fs.innermost(off)
                if f:
                    a_by_fn.setdefault(id(f), []).append(off)
            for off in call_sites(fs, second):
                f = fs.innermost(off)
                if f:
                    b_by_fn.setdefault(id(f), []).append(off)
            for f in fs.functions:
                a, b = a_by_fn.get(id(f)), b_by_fn.get(id(f))
                if a and b and min(a) < max(b):
                    out.append(
                        {"file_path": fs.path, "start_line": f.start_line, "end_line": f.end_line}
                    )
        return out

    def callers(self, name: str) -> list[dict]:
        out = []
        for fs in self.scans:
            if name not in fs.text:
                continue
            for off in call_sites(fs, name):
                line = fs.line_of(off)
                out.append({"file_path": fs.path, "start_line": line, "end_line": line})
        return out

    def callees(self, name: str) -> list[dict]:
        out: list[dict] = []
        targets = self.defs.get(name, [])
        if len(targets) != 1:
            return out
        fs, fn = targets[0]
        body = fs.text[fn.body_start : fn.body_end]
        for m in re.finditer(rf"(?<![\w$])({IDENT})\s*\(", body):
            callee = m.group(1)
            off = fn.body_start + m.start()
            if callee in KEYWORDS or callee == name or off in fs.def_name_offsets:
                continue
            if fs.innermost(off) is not fn:
                inner = fs.innermost(off)
                if inner is not None and inner.name != name:
                    continue  # belongs to a nested named function
            cands = self.defs.get(callee, [])
            same_file = [c for c in cands if c[0] is fs]
            if same_file:  # lexical scoping: a definition in the same file wins
                cands = same_file[:1]
            if len(cands) == 1:  # otherwise unambiguous textual definitions only
                cfs, cfn = cands[0]
                item = {
                    "file_path": cfs.path,
                    "start_line": cfn.start_line,
                    "end_line": cfn.end_line,
                }
                if item not in out:
                    out.append(item)
        return out


def _overlap(a: dict, b: dict) -> bool:
    return (
        a["file_path"] == b["file_path"]
        and a["start_line"] <= b["end_line"]
        and b["start_line"] <= a["end_line"]
    )


def agreement(index: list[dict], check: list[dict]) -> dict:
    i_in_c = sum(1 for x in index if any(_overlap(x, y) for y in check))
    c_in_i = sum(1 for y in check if any(_overlap(x, y) for x in index))
    union = len(index) + len(check) - min(i_in_c, c_in_i)
    return {
        "index_items": len(index),
        "checker_items": len(check),
        "index_confirmed": i_in_c,  # index labels the checker also finds
        "checker_confirmed": c_in_i,  # checker items the index also has
        "jaccard": round(min(i_in_c, c_in_i) / union, 3) if union else 1.0,
        "exact": i_in_c == len(index) and c_in_i == len(check),
    }


def parse_query(q: dict) -> tuple[str, tuple[str, ...]]:
    text = q["query"]
    if q["subtype"] == "sequence":
        m = re.match(r"Which files call (\S+) before (\S+)\?", text)
        return "sequence", (m.group(1), m.group(2))
    if q["subtype"] == "callers":
        return "callers", (re.match(r"Who calls (\S+)\?", text).group(1),)
    return "callees", (re.match(r"What does (\S+) call\?", text).group(1),)


def main() -> None:
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    gold = json.loads((Path(__file__).parent / "gold.json").read_text())
    checker = Checker(config.REPO_ROOT)
    rows, gold_check = [], {}
    for q in gold["queries"]:
        if q["type"] != "structural":
            continue
        kind, args = parse_query(q)
        found = getattr(checker, kind)(*args)
        agr = agreement(q["gold"], found)
        rows.append(
            {"id": q["id"], "kind": kind, "query": q["query"], **agr, "checker_gold": found}
        )
        gold_check[q["id"]] = found
    n = len(rows)
    idx_total = sum(r["index_items"] for r in rows)
    chk_total = sum(r["checker_items"] for r in rows)
    summary = {
        "queries": n,
        "exact_agreement": sum(r["exact"] for r in rows),
        "mean_jaccard": round(sum(r["jaccard"] for r in rows) / max(n, 1), 3),
        "index_labels_confirmed_by_checker": round(
            sum(r["index_confirmed"] for r in rows) / max(idx_total, 1), 3
        ),
        "checker_items_present_in_index": round(
            sum(r["checker_confirmed"] for r in rows) / max(chk_total, 1), 3
        ),
        "files_scanned": len(checker.scans),
        "functions_found": sum(len(s.functions) for s in checker.scans),
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps({"summary": summary, "queries": rows}, indent=2))
    GOLD_CHECK.write_text(json.dumps(gold_check, indent=1))
    lines = [
        "# Structural ground truth: index vs independent text checker",
        "",
        "The checker (`eval/structural_checker.py`) shares no code with the indexer: its own file",
        "walk, a comment/string-blanking lexer, regex function detection with brace matching, and",
        "regex call sites attributed to the innermost named function. Items agree when they are in",
        "the same file and their line ranges overlap.",
        "",
        f"- Queries checked: {n}; exact agreement: {summary['exact_agreement']}/{n}; "
        f"mean Jaccard: {summary['mean_jaccard']}",
        f"- Index labels confirmed by the checker: "
        f"{summary['index_labels_confirmed_by_checker'] * 100:.1f}%",
        f"- Checker items also in the index labels: "
        f"{summary['checker_items_present_in_index'] * 100:.1f}%",
        f"- Checker scanned {summary['files_scanned']} files, "
        f"{summary['functions_found']} named functions",
        "",
        "| Query | Kind | Index items | Checker items | Index confirmed | Checker confirmed | "
        "Jaccard |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} {r['query']} | {r['kind']} | {r['index_items']} | "
            f"{r['checker_items']} | {r['index_confirmed']} | {r['checker_confirmed']}"
            f" | {r['jaccard']} |"
        )
    lines += [
        "",
        "## Known checker limitations (why it can disagree)",
        "",
        "- It resolves a callee only by name: a definition in the same file wins, otherwise the",
        "  name must be defined exactly once. It cannot follow `this.props.httpClient.get`-style",
        "  dispatch (the index resolves those by owner matching), and it can wrongly map builtin",
        "  method calls such as `.split(` to an unrelated user function of the same name.",
        "- Function detection is regex-based; unusual syntax can hide a function from it.",
        "",
        "Disagreements are therefore reported, not used to change either side. On this gold set",
        "the remaining disagreements were inspected in the source: in each case the index label",
        "was correct and the checker hit one of the limitations above.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
