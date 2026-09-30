"""Query and snippet pre-processing / categorisation (text only, deterministic).

Queries (programming-problem statements) are split into sections — statement, input format,
output format, examples, notes/constraints — cleaned of LaTeX and judge boilerplate, and
categorised by the kind of program they ask for (stdin script vs function vs class method) plus a
few topic tags. Snippets are normalised (comments/docstrings/blank lines removed) and categorised
the same way, so the two can be matched on program form as well as meaning.
"""

from __future__ import annotations

import io
import re
import tokenize
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------- query side

# Section headers as they appear in APPS (Codeforces "-----Input-----", AtCoder "Input",
# LeetCode "Example 1:", Codewars "## Examples", CodeChef "-----Sample Input-----").
_SECTION_RE = re.compile(
    r"^\s*(?:-{3,}\s*)?(?:#+\s*)?"
    r"(?P<name>input(?:\s+format)?|output(?:\s+format)?|examples?(?:\s*\d+)?|"
    r"sample\s+input(?:\s*\d+)?|sample\s+output(?:\s*\d+)?|sample(?:\s*\d+)?|notes?|"
    r"constraints?|explanation|subtasks?|scoring|input\s+/\s*output|example\s+\d+)"
    r"\s*:?\s*(?:-{3,})?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_LATEX = [
    (r"\\le(?:q)?\b", "<="),
    (r"\\ge(?:q)?\b", ">="),
    (r"\\ne(?:q)?\b", "!="),
    (r"\\lt\b", "<"),
    (r"\\gt\b", ">"),
    (r"\\cdot\b|\\times\b", "*"),
    (r"\\ldots\b|\\dots\b|\\cdots\b", "..."),
    (r"\\oplus\b", " xor "),
    (r"\\bmod\b|\\mod\b", " mod "),
    (r"\\sum\b", " sum "),
    (r"\\max\b", " max "),
    (r"\\min\b", " min "),
    (r"\\lfloor\b|\\rfloor\b|\\lceil\b|\\rceil\b", ""),
    (r"\\(?:text|mathrm|mathbf|texttt|operatorname)\{([^}]*)\}", r"\1"),
    (r"\\[,;:!]", " "),
    (r"\\left|\\right", ""),
]
_LATEX_RE = [(re.compile(a), b) for a, b in _LATEX]

TOPIC_WORDS: dict[str, tuple[str, ...]] = {
    "graph": ("graph", "vertex", "vertices", "edge", "edges", "node", "path", "cycle"),
    "tree": ("tree", "root", "subtree", "leaf", "leaves", "parent"),
    "string": ("string", "substring", "palindrome", "character", "letters", "lowercase"),
    "math": ("prime", "divisor", "gcd", "modulo", "factorial", "integer", "digit"),
    "geometry": ("point", "polygon", "circle", "coordinate", "rectangle", "segment", "area"),
    "grid": ("grid", "matrix", "cell", "row", "column", "board"),
    "sorting": ("sort", "sorted", "order", "permutation"),
    "dp": ("maximum", "minimum", "number of ways", "subsequence", "optimal"),
    "game": ("game", "player", "win", "wins", "move", "turn"),
    "queries": ("queries", "query"),
}


@dataclass
class QueryInfo:
    """A parsed problem statement."""

    raw: str
    style: str  # "stdin" | "function" | "class"
    multi_test: bool
    interactive: bool
    topics: list[str]
    sections: dict[str, str]
    function_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw")
        d["sections"] = {k: len(v) for k, v in self.sections.items()}
        return d


def clean_text(text: str) -> str:
    """Remove LaTeX markup, `$` math delimiters, section rules and redundant whitespace."""
    t = text.replace("$$$", "").replace("$", "")
    for rx, rep in _LATEX_RE:
        t = rx.sub(rep, t)
    t = re.sub(r"\\([a-zA-Z]+)", r"\1", t)  # remaining \commands → plain word
    t = re.sub(r"-{3,}", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


def split_sections(text: str) -> dict[str, str]:
    """Statement / input / output / examples / notes. Unknown headers fold into the nearest."""
    parts: dict[str, list[str]] = {"statement": []}
    current = "statement"
    last = 0
    for m in _SECTION_RE.finditer(text):
        name = m.group("name").lower()
        # Inside the examples, bare "Input" / "Output" lines label sample data, not sections.
        if current == "examples" and re.fullmatch(r"(?:input|output)(?:\s*\d+)?", name):
            continue
        parts.setdefault(current, []).append(text[last : m.start()])
        if name.startswith("input") and "output" in name:
            current = "input"
        elif name.startswith("input"):
            current = "input"
        elif name.startswith("output"):
            current = "output"
        elif name.startswith(("example", "sample", "explanation")):
            current = "examples"
        else:
            current = "notes"
        last = m.end()
    parts.setdefault(current, []).append(text[last:])
    out = {k: "\n".join(x.strip() for x in v if x.strip()) for k, v in parts.items()}
    # Codewars / LeetCode inline examples: "Example 1: Input: s = ..." inside the statement.
    st = out.get("statement", "")
    m = re.search(r"\n\s*(?:for )?examples?\b[^\n]*[:\n]", st, re.IGNORECASE)
    if m and m.start() > 80:
        out["examples"] = (st[m.start() :] + "\n" + out.get("examples", "")).strip()
        out["statement"] = st[: m.start()].strip()
    return {k: v for k, v in out.items() if v}


def categorise_query(text: str) -> QueryInfo:
    low = text.lower()
    sections = split_sections(text)
    fnames = re.findall(r"\b(?:def|function)\s+([A-Za-z_]\w*)", text)
    fnames += re.findall(r"`([a-z_][A-Za-z0-9_]*)(?:\(\))?`", text)
    fnames += re.findall(r"^\s*([a-z_][A-Za-z0-9_]*)\(", sections.get("examples", ""), re.MULTILINE)
    has_io = "input" in sections and "output" in sections
    if re.search(r"class solution\b|\bdef \w+\(self\b", low):
        style = "class"
    elif not has_io and (
        fnames
        or re.search(r"\b(?:write|complete|create|implement) (?:a |the )?(?:function|method)", low)
        or re.search(r"\breturns?\b", low)
        and "print" not in low
    ):
        style = "function"
    else:
        style = "stdin"
    multi = bool(
        re.search(r"\b(?:t|q)\b\s*(?:\(|—|-)?[^.\n]{0,40}\btest cases?\b", low)
        or re.search(r"number of test cases|first line contains (?:a single integer )?t\b", low)
        or "test cases" in low
    )
    topics = [
        name
        for name, words in TOPIC_WORDS.items()
        if sum(len(re.findall(rf"\b{re.escape(w)}\b", low)) for w in words) >= 2
    ]
    return QueryInfo(
        raw=text,
        style=style,
        multi_test=multi,
        interactive="interactive" in low or "flush" in low,
        topics=topics,
        sections=sections,
        function_names=sorted(set(fnames))[:5],
    )


def query_view(info: QueryInfo, view: str) -> str:
    """Text variant of a parsed query fed to an encoder.

    full        cleaned full text
    statement   problem story only (no I/O format, examples, notes)
    no_examples statement + input/output format + notes (examples and sample I/O dropped)
    """
    s = info.sections
    if view == "raw":
        return info.raw
    if view == "full":
        return clean_text(info.raw)
    if view == "statement":
        return clean_text(s.get("statement", info.raw))
    if view == "no_examples":
        keep = [s.get("statement", "")]
        if s.get("input"):
            keep.append("Input: " + s["input"])
        if s.get("output"):
            keep.append("Output: " + s["output"])
        if s.get("notes"):
            keep.append(s["notes"])
        return clean_text("\n".join(k for k in keep if k)) or clean_text(info.raw)
    raise ValueError(f"unknown query view {view!r}")


# ---------------------------------------------------------------- snippet side


@dataclass
class SnippetInfo:
    style: str  # "stdin" | "function" | "class"
    reads_stdin: bool
    lines: int
    functions: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


_STDIN_RE = re.compile(r"\binput\s*\(|sys\.stdin|stdin\.read|open\(0\)|fileinput")


def categorise_snippet(code: str) -> SnippetInfo:
    reads = bool(_STDIN_RE.search(code))
    funcs = re.findall(r"^\s*def\s+([A-Za-z_]\w*)", code, re.MULTILINE)
    if re.search(r"^class Solution\b", code, re.MULTILINE):
        style = "class"
    elif reads:
        style = "stdin"
    elif funcs:
        style = "function"
    else:
        style = "stdin"  # bare script (e.g. prints a constant) behaves like a program
    return SnippetInfo(style, reads, code.count("\n") + 1, funcs[:8])


def strip_comments(code: str) -> str:
    """Drop comments and docstrings (tokenize-based; falls back to a regex on bad syntax)."""
    try:
        out = []
        prev = tokenize.INDENT
        last_line, last_col = -1, 0
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            ttype, tstr, (sl, sc), (el, ec), _ = tok
            if sl > last_line:
                last_col = 0
            if sc > last_col and ttype not in (tokenize.NL, tokenize.NEWLINE):
                out.append(" " * (sc - last_col))
            if ttype == tokenize.COMMENT:
                pass
            elif ttype == tokenize.STRING and prev in (
                tokenize.INDENT,
                tokenize.NEWLINE,
                tokenize.DEDENT,
            ):
                pass  # docstring / bare string statement
            else:
                out.append(tstr)
            if ttype not in (tokenize.COMMENT, tokenize.NL):
                prev = ttype
            last_line, last_col = el, ec
        code = "".join(out)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        code = re.sub(r"(?m)^\s*#.*$|\s+#[^'\"\n]*$", "", code)
    return code


def normalise_code(code: str, max_chars: int = 6000) -> str:
    """Comment/docstring-free code with blank lines and trailing spaces removed."""
    code = strip_comments(code.replace("\r\n", "\n").replace("\t", "    "))
    lines = [ln.rstrip() for ln in code.split("\n")]
    code = "\n".join(ln for ln in lines if ln.strip())
    return code[:max_chars]


def snippet_view(code: str, view: str) -> str:
    if view == "raw":
        return code
    if view == "normalised":
        return normalise_code(code)
    raise ValueError(f"unknown snippet view {view!r}")
