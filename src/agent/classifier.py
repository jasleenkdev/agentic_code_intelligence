"""Rule-based query router (docs/approach.md §A.3). No LLM is involved (D-027): relational wording
the rules cannot parse is flagged `needs_extraction` and answered by retrieval instead."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# A "name" in a question: identifier, dotted path, `call()`, quoted string or backticked text.
_NAME = r"[`'\"]?(?P<{g}>[A-Za-z_$][\w$.]*(?:\([^)]*\))?)[`'\"]?"
_OPT_NOUN = r"(?:(?:the\s+)?(?:tool|function|method|api|intent|action|event)\s+)?"

SEQ_BEFORE = re.compile(
    r"(?:call|calls|invoke|invokes|use|uses|run|runs|trigger|triggers)\s+"
    + _OPT_NOUN
    + _NAME.format(g="a")
    + r"\s+(?:before|prior\s+to|and\s+then)\s+(?:calling\s+|invoking\s+)?"
    + _OPT_NOUN
    + _NAME.format(g="b"),
    re.I,
)
SEQ_AFTER = re.compile(
    r"(?:call|calls|invoke|invokes|use|uses|run|runs)\s+"
    + _OPT_NOUN
    + _NAME.format(g="b")
    + r"\s+after\s+(?:calling\s+|invoking\s+)?"
    + _OPT_NOUN
    + _NAME.format(g="a"),
    re.I,
)
SEQ_LOOSE = re.compile(
    _NAME.format(g="a")
    + r"\s+(?:is\s+)?(?:called\s+)?(?:before|prior\s+to)\s+"
    + _NAME.format(g="b"),
    re.I,
)
SEQ_LOOSE_AFTER = re.compile(
    _NAME.format(g="b")
    + r"\s+(?:is\s+|gets?\s+)?(?:called\s+|invoked\s+|run\s+)?after\s+"
    + r"(?:calling\s+)?"
    + _NAME.format(g="a"),
    re.I,
)
CALLERS = [
    re.compile(
        r"(?:who|what|which\s+\w+)\s+(?:calls|invokes|uses)\s+" + _OPT_NOUN + _NAME.format(g="a"),
        re.I,
    ),
    re.compile(r"callers?\s+of\s+" + _OPT_NOUN + _NAME.format(g="a"), re.I),
    re.compile(
        r"where\s+(?:is|are)\s+" + _OPT_NOUN + _NAME.format(g="a") + r"\s+(?:called|invoked)", re.I
    ),
    re.compile(r"call\s+sites?\s+(?:of|for)\s+" + _OPT_NOUN + _NAME.format(g="a"), re.I),
]
CALLEES = [
    re.compile(
        r"what\s+(?:does|do)\s+" + _OPT_NOUN + _NAME.format(g="a") + r"\s+(?:call|invoke)", re.I
    ),
    re.compile(r"callees\s+of\s+" + _OPT_NOUN + _NAME.format(g="a"), re.I),
    re.compile(
        r"(?:functions|methods)\s+called\s+(?:by|from)\s+" + _OPT_NOUN + _NAME.format(g="a"), re.I
    ),
]
DEFINITION = [
    re.compile(
        r"where\s+(?:is|are)\s+"
        + _OPT_NOUN
        + _NAME.format(g="a")
        + r"\s+(?:defined|declared|implemented)",
        re.I,
    ),
    re.compile(r"definition\s+of\s+" + _OPT_NOUN + _NAME.format(g="a"), re.I),
    re.compile(
        r"(?:find|show|locate)\s+(?:the\s+)?(?:function|method|class)\s+" + _NAME.format(g="a"),
        re.I,
    ),
]
USAGE_WORDS = re.compile(
    r"\b(deep\s*-?links?|routes?|urls?|uris?|endpoints?|intents?|events?|constants?|strings?|"
    r"literals?|messages?|websocket|topics?|mcp\s+tools?)\b",
    re.I,
)
USAGE_VERBS = re.compile(
    r"\b(used|usages?|referenced|references?|emitted|emits?|listens?|listened|subscribed|"
    r"handled|consumed)\b",
    re.I,
)
QUOTED = re.compile(r"[`'\"]([^`'\"]{2,120})[`'\"]")
CONST_PATH = re.compile(r"\b[A-Z][A-Z0-9_]{2,}(?:\.[A-Z0-9_]+)+\b")
PATH_LIKE = re.compile(r"(/[\w\-:.{}$]+){2,}|[a-z][a-z0-9+.-]*://\S+")
SEMANTIC_HINTS = re.compile(
    r"^\s*(how|why|explain|what\s+happens|where\s+do\s+we|where\s+does|which\s+code\s+handles)",
    re.I,
)
STOP_NAMES = frozenset({"the", "a", "an", "it", "this", "that", "function", "tool", "code"})


@dataclass
class Route:
    type: str  # "sequence" | "callers" | "callees" | "definition" | "usage" | "semantic"
    category: str  # "structural" | "usage" | "semantic"
    params: dict = field(default_factory=dict)
    method: str = "rules"  # "rules" | "llm" | "default"

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(name: str | None) -> str | None:
    if not name:
        return None
    name = name.strip().strip("`'\".,?!")
    if not name or name.lower() in STOP_NAMES:
        return None
    return name


def classify(query: str) -> Route:
    q = query.strip()
    for rx in (SEQ_BEFORE, SEQ_AFTER, SEQ_LOOSE, SEQ_LOOSE_AFTER):
        m = rx.search(q)
        if m:
            a, b = _clean(m.group("a")), _clean(m.group("b"))
            if a and b:
                return Route("sequence", "structural", {"first": a, "second": b})
    for rx in CALLERS:
        m = rx.search(q)
        if m and _clean(m.group("a")):
            return Route("callers", "structural", {"symbol": _clean(m.group("a"))})
    for rx in CALLEES:
        m = rx.search(q)
        if m and _clean(m.group("a")):
            return Route("callees", "structural", {"symbol": _clean(m.group("a"))})
    for rx in DEFINITION:
        m = rx.search(q)
        if m and _clean(m.group("a")):
            return Route("definition", "structural", {"name": _clean(m.group("a"))})
    quoted = QUOTED.findall(q)
    consts = CONST_PATH.findall(q)
    paths = [m.group(0) for m in PATH_LIKE.finditer(q)]
    if consts:
        return Route("usage", "usage", {"text": consts[0], "exact": True})
    if paths:
        return Route("usage", "usage", {"text": paths[0].rstrip("?.,"), "exact": False})
    if quoted and (USAGE_VERBS.search(q) or USAGE_WORDS.search(q)):
        return Route("usage", "usage", {"text": quoted[0], "exact": False})
    if USAGE_WORDS.search(q) and (USAGE_VERBS.search(q) or re.search(r"\bwhere\b", q, re.I)):
        return Route("usage", "usage", {"text": q, "exact": False})
    if SEMANTIC_HINTS.search(q):
        return Route("semantic", "semantic", {"query": q})
    if re.search(r"\b(before|after|calls?|callers?|callees?)\b", q, re.I):
        # Relational wording the rules could not parse → let the LLM try to extract names.
        return Route("semantic", "semantic", {"query": q, "needs_extraction": True}, "default")
    return Route("semantic", "semantic", {"query": q}, "default")
