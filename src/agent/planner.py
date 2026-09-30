"""Query planning: classify and decompose a question into sub-plans with explicit steps (D-028).

Rules first. "Which files call X before Y?" → resolve X, resolve Y, find call sequences, read,
verify. Multi-part questions ("Who calls X and what does it call?") are split into clauses, with
pronouns resolved to the previous clause's symbol. The LLM is only a *fallback* for relational
wording the rules could not parse, and runs after the first results are shown (phase B).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.agent.classifier import Route, _clean, classify

logger = logging.getLogger(__name__)

# Clause boundaries: "; ", "? <next>", ", and who/what/…", " and then", " and also".
_SPLIT_RE = re.compile(
    r"\s*(?:;|\?\s+(?=\S)|,?\s+and\s+(?:then\s+|also\s+)?(?=(?:who|what|where|which|how|does|do|is|"
    r"list|show|find)\b))\s*",
    re.I,
)
_PRONOUN_RE = re.compile(r"\b(it|its|them|this function|that function)\b", re.I)

STEPS = {
    "sequence": [
        "resolve {first}",
        "resolve {second}",
        "find call sequences",
        "read candidates + call-graph neighbours",
        "verify",
    ],
    "callers": [
        "resolve {symbol}",
        "get callers",
        "read candidates + call-graph neighbours",
        "verify",
    ],
    "callees": [
        "resolve {symbol}",
        "get callees",
        "read candidates + call-graph neighbours",
        "verify",
    ],
    "definition": ["resolve {name}", "read definitions + call-graph neighbours", "verify"],
    "usage": ["catalog lookup", "string / constant usages", "read usage sites", "verify"],
    "semantic": [
        "hybrid search",
        "read candidates + call-graph neighbours",
        "assess confidence",
        "refine if low: graph/catalog expansion, then constrained rewrite (≤ 2 rounds)",
        "verify",
    ],
}


@dataclass
class SubPlan:
    text: str
    route: Route
    steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"text": self.text, "route": self.route.to_dict(), "steps": self.steps}


@dataclass
class Plan:
    query: str
    subplans: list[SubPlan]
    method: str = "rules"  # rules | llm
    decomposed: bool = False

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "method": self.method,
            "decomposed": self.decomposed,
            "subplans": [s.to_dict() for s in self.subplans],
        }

    @property
    def needs_llm_planning(self) -> bool:
        return any(s.route.params.get("needs_extraction") for s in self.subplans)


def _main_symbol(route: Route) -> str | None:
    p = route.params
    return p.get("symbol") or p.get("name") or p.get("second") or p.get("first")


def _steps_for(route: Route) -> list[str]:
    return [
        s.format(**{k: v for k, v in route.params.items() if isinstance(v, str)})
        for s in STEPS[route.type]
    ]


def split_clauses(query: str) -> list[str]:
    parts = [p.strip(" ?.") for p in _SPLIT_RE.split(query.strip()) if p and p.strip(" ?.")]
    return parts or [query]


def plan_query(query: str) -> Plan:
    """Deterministic plan. Decomposes only when ≥ 2 clauses and at least one is structural or
    usage (otherwise the whole question is one semantic sub-plan)."""
    clauses = split_clauses(query)
    if len(clauses) > 1:
        subs: list[SubPlan] = []
        last_symbol: str | None = None
        for clause in clauses:
            text = clause
            if last_symbol and _PRONOUN_RE.search(text):
                text = _PRONOUN_RE.sub(last_symbol, text)
            route = classify(text)
            subs.append(SubPlan(text, route, _steps_for(route)))
            if route.category != "semantic":
                last_symbol = _main_symbol(route) or last_symbol
        if any(s.route.category != "semantic" for s in subs):
            return Plan(query, subs, "rules", decomposed=True)
    route = classify(query)
    return Plan(query, [SubPlan(query, route, _steps_for(route))], "rules")


EXTRACT_PROMPT = """Extract the structure of a code question. Reply ONLY with JSON:
{"type": "sequence"|"callers"|"callees"|"definition"|"usage"|"semantic",
 "first": str|null, "second": str|null, "symbol": str|null, "text": str|null}
- sequence: "does X get called before Y" (first=X, second=Y; for "X after Y" first=Y, second=X)
- callers: who calls symbol; callees: what symbol calls; definition: where symbol is defined
- usage: where a string/constant/route/event is used (text)
Names must be copied verbatim from the question.
Question: """


def llm_plan(query: str, llm) -> Route | None:
    """LLM fallback planner: only for relational wording the rules could not parse. Names must
    appear verbatim in the question, so the LLM cannot invent symbols."""
    from src.agent.llm import LLMUnavailable, parse_json_object

    try:
        data = parse_json_object(
            llm.chat([{"role": "user", "content": EXTRACT_PROMPT + query}], max_tokens=80).text
        )
    except (LLMUnavailable, ValueError) as exc:
        logger.info("llm_plan failed: %s", exc)
        return None
    lower = query.lower()

    def verbatim(v: object) -> str | None:
        v = _clean(v if isinstance(v, str) else None)
        return v if v and v.split("(")[0].lower() in lower else None

    t = data.get("type")
    if t == "sequence" and verbatim(data.get("first")) and verbatim(data.get("second")):
        return Route(
            "sequence",
            "structural",
            {"first": verbatim(data["first"]), "second": verbatim(data["second"])},
            "llm",
        )
    if t in {"callers", "callees"} and verbatim(data.get("symbol")):
        return Route(t, "structural", {"symbol": verbatim(data["symbol"])}, "llm")
    if t == "definition" and verbatim(data.get("symbol")):
        return Route("definition", "structural", {"name": verbatim(data["symbol"])}, "llm")
    if t == "usage" and verbatim(data.get("text")):
        return Route("usage", "usage", {"text": verbatim(data["text"]), "exact": False}, "llm")
    return None
