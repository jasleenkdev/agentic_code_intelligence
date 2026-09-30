"""Planner-driven agentic answer pipeline (D-028; supersedes D-027's single-shot retrieval).

    plan → search → read → assess → refine (≤ 2 grounded rounds) → verify → stop

Phase A (deterministic, returns the first results): the planner classifies and decomposes the
question into sub-plans; each sub-plan runs its tools (graph/catalog lookups or hybrid search),
reads the top candidates and their call-graph neighbours, assesses confidence from deterministic
signals, and — if confidence is low — runs refinement round 1 (graph/catalog expansion from the
best partial hits). Phase B (`continue_with_llm`, runs after the first results are shown): the
LLM planning fallback for relational questions the rules could not parse, and refinement round 2,
a constrained rewrite in which the LLM may only pick real index terms. The explanation streams
last. Every step is a structured `Step` event (kind, inputs, outputs, timing, phase), and every
file:line passes `verifier.verify` (re-read from disk).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from src import config
from src.agent.classifier import Route
from src.agent.expand import ExpandConfig, catalog_chunks, expand_and_rescore, read_neighbours
from src.agent.llm import LLMClient, LLMUnavailable, get_llm
from src.agent.planner import SubPlan, llm_plan, plan_query
from src.agent.refine import (
    RefineConfig,
    assess,
    candidate_terms,
    merge_rounds,
    pick_terms,
    rewrite_query,
)
from src.agent.tools import code_tools
from src.agent.tools.registry import call_tool
from src.agent.verifier import Snippet, verify
from src.config import AgentBudget, RankingConfig

logger = logging.getLogger(__name__)

SEQ_WINDOW = 40  # show at most this many lines around call sites in long functions
SUPPORT_SNIPPETS = 5  # semantic sub-plans show this many snippets (+3 spare for failed checks)
SEARCH_K = 20


@dataclass
class Step:
    """One structured trace event."""

    index: int
    kind: str  # plan | search | read | assess | refine | verify | stop | llm
    name: str
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    summary: str = ""
    result_count: int = 0
    elapsed_ms: float = 0.0
    ok: bool = True
    phase: str = "first"  # first (before the first results) | refine (after, LLM)


@dataclass
class AnswerResult:
    query: str
    route: dict
    answer: str = ""
    snippets: list[Snippet] = field(default_factory=list)
    call_paths: list[list[str]] = field(default_factory=list)
    confidence: str = "low"
    steps: list[Step] = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    llm_used: bool = False
    llm_model: str | None = None
    fallback_used: bool = False
    latency_ms: float = 0.0  # time to the first results (phase A)
    final_latency_ms: float = 0.0  # time to the final results (after phase B, if run)
    refinement_rounds: int = 0
    results_changed_by_llm: bool = False
    tokens: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    dropped_citations: int = 0
    optimizations: list[dict] = field(default_factory=list)
    explanation: str = ""
    error: str | None = None
    state: dict = field(default_factory=dict, repr=False)  # internal, not serialised

    @property
    def locations(self) -> list[str]:
        return [s.location for s in self.snippets]

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "search")

    def to_dict(self) -> dict:
        state, self.state = self.state, {}
        d = asdict(self)
        self.state = state
        d.pop("state", None)
        d["snippets"] = [s.to_dict() for s in self.snippets]
        d["locations"] = self.locations
        d["tool_calls"] = self.tool_calls
        return d


class _Run:
    """Step emitter + counters for one question (shared by all sub-plans)."""

    def __init__(
        self,
        result: AnswerResult,
        llm: LLMClient | None,
        budget: AgentBudget,
        on_step: Callable[[Step], None] | None = None,
        phase: str = "first",
    ) -> None:
        self.t0 = time.perf_counter()
        self.llm = llm
        self.budget = budget
        self.on_step = on_step
        self.phase = phase
        self.result = result
        self.root = result
        self.verified = 0
        self.attempted = 0

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def step(self, kind: str, name: str, **kw) -> Step:
        s = Step(index=len(self.root.steps) + 1, kind=kind, name=name, phase=self.phase, **kw)
        self.root.steps.append(s)
        if self.on_step is not None:
            try:
                self.on_step(s)
            except Exception:  # noqa: BLE001 — a UI callback must never break answering
                logger.debug("on_step callback failed", exc_info=True)
        return s

    def tool(self, name: str, args: dict) -> dict:
        res, ms = call_tool(name, args)
        self.step(
            "search",
            name,
            inputs=args,
            outputs={"count": _result_count(res)},
            summary=_short_summary(name, res),
            result_count=_result_count(res),
            elapsed_ms=ms,
            ok="error" not in res,
        )
        return res

    def verify(self, *args, **kwargs) -> Snippet | None:
        self.attempted += 1
        sn = verify(*args, **kwargs)
        self.verified += sn is not None
        return sn

    def sub(self, result: AnswerResult) -> _Run:
        """A view writing into a sub-plan's result but emitting into the same trace."""
        view = _Run.__new__(_Run)
        view.__dict__.update(self.__dict__)
        view.result = result
        view.verified = view.attempted = 0
        return view


# ------------------------------------------------------------------------------ formatting


def _result_count(res: dict) -> int:
    for key in ("results", "callers", "callees", "matches"):
        if isinstance(res.get(key), list):
            return len(res[key])
    if "files" in res:
        return res.get("total", 0)
    if "text" in res and "start_line" in res:
        return 1
    return 0


def _short_summary(name: str, res: dict) -> str:
    if "error" in res:
        return f"error: {res['error']}"
    if name in {"semantic_search", "keyword_search"}:
        return "; ".join(
            f"{r['file_path']}:{r['start_line']} {r['symbol']}" for r in res["results"][:3]
        )
    if name == "find_call_sequence":
        return f"{res['total']} functions in {len(res['files'])} files"
    if name == "find_string_usages":
        return f"{res['total']} usages in {len(res['files'])} files"
    if name == "read_file_range":
        return f"{res['file_path']}:{res['start_line']}-{res['end_line']}"
    return f"{_result_count(res)} results"


# ------------------------------------------------------------------------------ deterministic


def _bare(name: str) -> str:
    return name.split("(")[0].strip().split(".")[-1]


def _window(start: int, end: int, lines: list[int]) -> tuple[int, int]:
    if end - start + 1 <= SEQ_WINDOW:
        return start, end
    lo, hi = min(lines), max(lines)
    return max(start, lo - 3), min(end, max(hi + 3, lo + 8))


def _deterministic(run: _Run, route: Route) -> bool:
    """Answer structural/usage routes from the index. Returns False if nothing was found."""
    res = run.result
    p = route.params
    snippets: list[Snippet] = []
    # Planned "resolve" steps: look each named symbol up before using it.
    names = {
        "sequence": [p.get("first"), p.get("second")],
        "callers": [p.get("symbol")],
        "callees": [p.get("symbol")],
    }.get(route.type, [])
    for name in filter(None, names):
        run.tool("find_symbol", {"name": _bare(name)})
    if route.type == "usage" and not p.get("exact"):
        from src.catalog.search import search_catalog

        t = time.perf_counter()
        cat = search_catalog(code_tools.get_context().conn, p["text"], limit=3)
        run.step(
            "search",
            "catalog lookup",
            inputs={"text": p["text"]},
            outputs={"entries": [f"{c.kind} {c.name} ({c.score})" for c in cat]},
            result_count=len(cat),
            summary="; ".join(f"{c.kind} {c.name}" for c in cat[:2]) or "no catalog entry",
            elapsed_ms=round((time.perf_counter() - t) * 1000, 1),
        )
    if route.type == "sequence":
        data = run.tool(
            "find_call_sequence",
            {"first": p["first"], "second": p["second"], "same_function": False},
        )
        matches = data.get("matches", [])
        for m in matches[:12]:
            lines = [m["first_call"]["entry_line"], m["second_call"]["entry_line"]]
            a, b = _window(m["start_line"], m["end_line"], lines)
            expect = [_bare(m["first_call"]["text"])] if not m["first_call"]["via"] else None
            sn = run.verify(
                m["file_path"],
                a,
                b,
                expect=expect,
                symbol=m["function"],
                reason=f"{m['certainty']}: {p['first']} (line {lines[0]}) before "
                f"{p['second']} (line {lines[1]}), hops={m['hops']}"
                + (f"; {'; '.join(m['reasons'][:2])}" if m["reasons"] else ""),
                highlight=lines,
            )
            if sn:
                snippets.append(sn)
                res.call_paths.append(
                    [m["function"], *m["first_call"]["via"], m["first_call"]["text"]]
                )
                res.call_paths.append(
                    [m["function"], *m["second_call"]["via"], m["second_call"]["text"]]
                )
        if matches:
            certain = sum(1 for m in matches if m["certainty"] == "certain")
            files = list(data["files"])
            res.answer = (
                f"`{p['first']}` is called before `{p['second']}` in {data['total']} function(s) "
                f"across {len(files)} file(s): {certain} certain, {data['total'] - certain} "
                "possible (branch, callback, or heuristic call-graph hop). Direct same-function "
                "orderings are listed first. Files: "
                + ", ".join(files[:8])
                + ("…" if len(files) > 8 else "")
            )
            res.confidence = "high" if certain else "medium"
        else:
            rev = code_tools.find_call_sequence(p["second"], p["first"], same_function=False)
            if rev["total"]:
                res.answer = (
                    f"No function calls `{p['first']}` before `{p['second']}`; the "
                    f"reverse order occurs in {rev['total']} function(s)."
                )
                for m in rev["matches"][:5]:
                    sn = run.verify(
                        m["file_path"],
                        m["start_line"],
                        min(m["end_line"], m["start_line"] + SEQ_WINDOW),
                        symbol=m["function"],
                        reason="reverse order",
                        highlight=[m["first_call"]["entry_line"], m["second_call"]["entry_line"]],
                    )
                    if sn:
                        snippets.append(sn)
                res.confidence = "medium"
    elif route.type == "callers":
        data = run.tool("get_callers", {"symbol": p["symbol"], "depth": 1})
        callers = data.get("callers", [])
        for c in callers[:15]:
            a, b = _window(c["caller_start_line"], c["caller_end_line"], [c["call_line"]])
            sn = run.verify(
                c["file_path"],
                a,
                b,
                expect=[_bare(p["symbol"])],
                symbol=c["caller"],
                reason=f"calls {c['call_text']} at line {c['call_line']} [{c['resolution']}]",
                highlight=[c["call_line"]],
            )
            if sn:
                snippets.append(sn)
                res.call_paths.append([c["caller"], c["call_text"]])
        if callers:
            resolved = sum(1 for c in callers if not c["resolution"].startswith("name-match"))
            res.answer = (
                f"`{p['symbol']}` has {len(callers)} call site(s): {resolved} resolved "
                f"through the call graph and {len(callers) - resolved} unresolved "
                "same-name calls (dynamic dispatch, listed last)."
            )
            res.confidence = "high" if resolved else "medium"
    elif route.type == "callees":
        data = run.tool("get_callees", {"symbol": p["symbol"], "depth": 1})
        sources: list[Snippet] = []  # shown after the callees the question asks for
        for s in data.get("sources", [])[:3]:
            sn = run.verify(
                s["file_path"],
                s["start_line"],
                s["end_line"],
                symbol=s["qualname"],
                reason="function whose callees are listed",
                highlight=[c["line"] for c in data["callees"] if c["file_path"] == s["file_path"]][
                    :30
                ],
            )
            if sn:
                sources.append(sn)
        seen_targets: set[str] = set()
        for c in data.get("callees", []):
            loc = c["target_location"]
            if not loc or loc in seen_targets or len(seen_targets) >= 8:
                continue
            seen_targets.add(loc)
            path, span = loc.rsplit(":", 1)
            a, b = (int(x) for x in span.split("-"))
            sn = run.verify(
                path,
                a,
                b,
                symbol=c["target"],
                reason=f"callee: {c['call_text']} (line {c['line']}) [{c['resolution']}]",
                highlight=[a],
            )
            if sn:
                snippets.append(sn)
        snippets.extend(sources)
        for c in data.get("callees", [])[:25]:
            res.call_paths.append(
                [
                    c["caller"],
                    c["call_text"] + (f" → {c['target_location']}" if c["target_location"] else ""),
                ]
            )
        if data.get("sources"):
            n = len(data["callees"])
            res.answer = (
                f"`{p['symbol']}` makes {n} project call(s) "
                f"({data['external_calls_omitted']} builtin/library calls omitted), "
                "in source order; resolved targets are linked."
            )
            res.confidence = "high"
    elif route.type == "definition":
        data = run.tool("find_symbol", {"name": p["name"]})
        for s in data.get("results", [])[:10]:
            sn = run.verify(
                s["file_path"],
                s["start_line"],
                s["end_line"],
                expect=[_bare(p["name"])],
                symbol=s["qualname"],
                reason=f"definition ({s['kind']})",
                highlight=[s["start_line"]],
            )
            if sn:
                snippets.append(sn)
        if snippets:
            res.answer = f"`{p['name']}` is defined in {len(snippets)} place(s)."
            res.confidence = "high"
    elif route.type == "usage":
        data = run.tool("find_string_usages", {"text": p["text"], "exact": p.get("exact", False)})
        cat = data.get("catalog", [])
        items = [u for uses in data.get("files", {}).values() for u in uses]
        # Definitions first, then usages; cap the number of snippets shown.
        # Tool order: catalog definitions, exact literal/constant matches, catalog usages, then
        # partial matches.
        items.sort(key=lambda u: (u.get("rank", 3), u["file_path"], u["line"]))
        for u in items[:20]:
            value = str(u["value"])
            expect = [value.split("${")[0][:40], value.split(".")[-1]] if value else None
            sn = run.verify(
                u["file_path"],
                max(1, u["line"] - 2),
                u["line"] + 2,
                expect=expect,
                symbol=u.get("symbol"),
                reason=f"{u['kind']} ({u['context']})",
                highlight=[u["line"]],
            )
            if sn:
                snippets.append(sn)
        if items:
            head = ""
            if cat:
                c = cat[0]
                head = (
                    f"Matched catalog entry {c['kind']} `{c['name']}`"
                    + (f" ({c['detail']})" if c.get("detail") else "")
                    + ". "
                )
            n_files = len(data["files"])
            res.answer = head + f"{len(items)} definition/usage site(s) in {n_files} file(s)."
            res.confidence = "high" if cat or p.get("exact") else "medium"
    res.snippets = snippets
    return bool(snippets)


# ------------------------------------------------------------------------------ read / verify


def _hits_for_snippets(retriever, snippets: list[Snippet]) -> list:
    """Map verified snippets back to index chunks (for neighbour reads)."""
    ids = []
    for sn in snippets:
        cid = retriever.chunk_for_line(sn.file_path, sn.start_line)
        if cid is not None:
            ids.append(cid)
    meta = retriever._meta(list(dict.fromkeys(ids)))
    return [meta[c] for c in dict.fromkeys(ids) if c in meta]


def _read_step(run: _Run, hits: list, seeds: int = 3) -> dict:
    """READ: the top candidates' line ranges were opened by verify(); here we open their
    immediate call-graph neighbours and record what the agent read."""
    t = time.perf_counter()
    retriever = code_tools.get_context().retriever
    neigh = read_neighbours(retriever, hits, seeds)
    outputs = {
        "candidates": [
            f"{h.file_path}:{h.start_line}-{h.end_line} {h.symbol}" for h in hits[:seeds]
        ],
        "neighbours": {
            f"{h.symbol}": [f"{rel}: {label}" for _c, rel, label in neigh.get(h.chunk_id, [])][:6]
            for h in hits[:seeds]
        },
    }
    n = sum(len(v) for v in neigh.values())
    run.step(
        "read",
        "candidates + call-graph neighbours",
        inputs={"top": min(seeds, len(hits))},
        outputs=outputs,
        result_count=n,
        summary=f"read {min(seeds, len(hits))} candidates and {n} neighbours",
        elapsed_ms=round((time.perf_counter() - t) * 1000, 1),
    )
    return neigh


def _hit_snippets(
    run: _Run, hits: list, reason: str, limit: int = SUPPORT_SNIPPETS + 3
) -> list[Snippet]:
    out = []
    for h in hits:
        if len(out) >= limit:
            break
        src = "+".join(sorted(h.ranks)) if h.ranks else "retrieval"
        sn = run.verify(
            h.file_path, h.start_line, h.end_line, symbol=h.symbol, reason=f"{reason} ({src})"
        )
        if sn:
            out.append(sn)
    return out


def _fmt_top(hits: list, n: int = 3) -> list[str]:
    return [f"{h.file_path}:{h.start_line} {h.symbol}" for h in hits[:n]]


# ------------------------------------------------------------------------------ semantic


def _semantic_first(
    run: _Run, text: str, ranking: RankingConfig, refine: RefineConfig, expand: ExpandConfig
) -> None:
    """Phase A for a semantic sub-plan: search → read → assess → (round 1: expansion) →
    assess → verify. Leaves state for phase B in run.result.state."""
    res = run.result
    retriever = code_tools.get_context().retriever
    t = time.perf_counter()
    hits = retriever.search(text, SEARCH_K, ranking)
    run.step(
        "search",
        "hybrid_search",
        inputs={"query": text, "k": SEARCH_K},
        outputs={"top": _fmt_top(hits)},
        result_count=len(hits),
        summary="; ".join(_fmt_top(hits)),
        elapsed_ms=round((time.perf_counter() - t) * 1000, 1),
    )
    run.verified = run.attempted = 0
    snippets = _hit_snippets(run, hits, "ranked retrieval")
    neigh = _read_step(run, hits)
    cat_ids, cat_label = catalog_chunks(retriever, text)
    conf = assess(hits, cat_ids, run.verified, run.attempted, refine.margin_threshold)
    run.step(
        "assess",
        "confidence",
        inputs={"catalog_entry": cat_label},
        outputs=conf,
        summary=f"{conf['label']}: margin {conf['margin']}, agreement {conf['agreement']}, "
        f"catalog in top 3 {conf['catalog_in_top3']}, verified "
        f"{conf['verified_ratio']}",
        ok=conf["label"] == "high",
    )
    rounds = 0
    wants = refine.gate == "always" or conf["label"] == "low"
    if refine.expand and expand.enabled and wants and rounds < refine.max_rounds:
        t = time.perf_counter()
        new_hits, info = expand_and_rescore(retriever, text, hits, expand, SEARCH_K)
        rounds += 1
        run.step(
            "refine",
            "round 1: graph/catalog expansion",
            inputs={"seeds": expand.seeds, "lambda": expand.lam, "from": _fmt_top(hits)},
            outputs=info | {"top": _fmt_top(new_hits)},
            result_count=info.get("added", 0),
            summary=f"added {info.get('added', 0)} neighbour/catalog chunks via "
            f"{info.get('edges', 0)} edges; top changed: {info.get('moved_to_top')}",
            elapsed_ms=round((time.perf_counter() - t) * 1000, 1),
        )
        hits = new_hits
        run.verified = run.attempted = 0
        snippets = _hit_snippets(run, hits, "ranked retrieval + graph expansion")
        conf = assess(hits, cat_ids, run.verified, run.attempted, refine.margin_threshold)
        run.step(
            "assess",
            "confidence after round 1",
            outputs=conf,
            summary=f"{conf['label']}: margin {conf['margin']}",
            ok=conf["label"] == "high",
        )
    run.step(
        "verify",
        "citations",
        inputs={"attempted": run.attempted},
        outputs={"verified": run.verified, "dropped": run.attempted - run.verified},
        result_count=len(snippets),
        summary=f"{run.verified}/{run.attempted} candidate ranges re-read and confirmed",
    )
    res.snippets = snippets
    res.confidence = conf["label"] if snippets else "low"
    res.answer = (
        f"Top {len(snippets)} code locations, ranked by hybrid retrieval (BM25 + "
        "embeddings + graph/catalog signals)"
        + (" and refined through the call graph" if rounds else "")
        + "; every range re-read from the source."
    )
    res.state = {
        "kind": "semantic",
        "text": text,
        "hits": hits,
        "neighbours": neigh,
        "confidence": conf,
        "rounds": rounds,
        "catalog_ids": cat_ids,
    }


def _semantic_rewrite(
    run: _Run, sub_res: AnswerResult, ranking: RankingConfig, refine: RefineConfig
) -> bool:
    """Phase B, round 2: constrained rewrite. Returns True if the results changed."""
    st = sub_res.state
    if not st or st.get("kind") != "semantic" or not refine.rewrite:
        return False
    if st["rounds"] >= refine.max_rounds or run.llm is None or not run.llm.available():
        return False
    if refine.gate != "always" and st["confidence"]["label"] != "low":
        return False
    retriever = code_tools.get_context().retriever
    text, hits = st["text"], st["hits"]
    t = time.perf_counter()
    terms = candidate_terms(retriever, text, hits, st["neighbours"], refine.max_terms)
    run.step(
        "refine",
        "round 2: candidate terms from the index",
        inputs={"from_candidates": len(hits)},
        outputs={
            "terms": len(terms),
            "kinds": {
                k: sum(1 for x in terms if x["kind"] == k) for k in ("identifier", "doc", "catalog")
            },
        },
        result_count=len(terms),
        summary=f"{len(terms)} real identifiers / doc phrases / catalog names offered to the LLM",
        elapsed_ms=round((time.perf_counter() - t) * 1000, 1),
    )
    if not terms:
        return False
    t = time.perf_counter()
    picked = pick_terms(run.llm, text, terms, refine.max_picks)
    run.root.llm_used = True
    run.root.llm_model = run.llm.model
    run.step(
        "llm",
        "pick relevant terms (constrained)",
        inputs={"offered": len(terms)},
        outputs={"picked": [p["text"] for p in picked]},
        result_count=len(picked),
        summary=", ".join(p["text"] for p in picked) or "none picked",
        elapsed_ms=round((time.perf_counter() - t) * 1000, 1),
        ok=bool(picked),
    )
    st["rounds"] += 1
    if not picked:
        return False
    q2 = rewrite_query(picked)
    t = time.perf_counter()
    hits2 = retriever.search(q2, SEARCH_K, ranking)
    run.step(
        "search",
        "hybrid_search (rewritten)",
        inputs={"query": q2, "k": SEARCH_K},
        outputs={"top": _fmt_top(hits2)},
        result_count=len(hits2),
        summary="; ".join(_fmt_top(hits2)),
        elapsed_ms=round((time.perf_counter() - t) * 1000, 1),
    )
    merged = merge_rounds(hits, hits2, refine, SEARCH_K)
    changed = [h.chunk_id for h in merged[:SUPPORT_SNIPPETS]] != [
        h.chunk_id for h in hits[:SUPPORT_SNIPPETS]
    ]
    run.step(
        "refine",
        "round 2: merge rounds",
        inputs={"round2_weight": refine.round2_weight},
        outputs={"top": _fmt_top(merged), "changed": changed},
        summary=f"top-{SUPPORT_SNIPPETS} changed: {changed}",
    )
    run.verified = run.attempted = 0
    snippets = _hit_snippets(run, merged, "ranked retrieval + constrained rewrite")
    conf = assess(merged, st["catalog_ids"], run.verified, run.attempted, refine.margin_threshold)
    run.step(
        "assess",
        "confidence after round 2",
        outputs=conf,
        summary=f"{conf['label']}: margin {conf['margin']}",
        ok=conf["label"] == "high",
    )
    run.step(
        "verify",
        "citations",
        inputs={"attempted": run.attempted},
        outputs={"verified": run.verified, "dropped": run.attempted - run.verified},
        result_count=len(snippets),
        summary=f"{run.verified}/{run.attempted} candidate ranges re-read and confirmed",
    )
    sub_res.snippets = snippets
    sub_res.confidence = conf["label"] if snippets else "low"
    sub_res.route["rewritten_query"] = q2
    st.update(hits=merged, confidence=conf)
    return changed


# ------------------------------------------------------------------------------ explanation

EXPLAIN_PROMPT = """You explain code-search results to a developer. Answer the question in 2-4
sentences using ONLY the code and facts below. Cite locations as path:line. If the code does not
answer the question, say so plainly.

Question: {question}
Facts: {facts}

{snippets}"""


def explanation_messages(res: AnswerResult, max_snippets: int = 3) -> list[dict]:
    blocks = []
    for sn in res.snippets[:max_snippets]:
        body = "\n".join(sn.text.split("\n")[:12])
        blocks.append(f"--- {sn.location} {sn.symbol or ''} [{sn.reason}]\n{body}")
    prompt = EXPLAIN_PROMPT.format(question=res.query, facts=res.answer, snippets="\n".join(blocks))
    return [{"role": "user", "content": prompt[:5000]}]


def stream_explanation(res: AnswerResult, llm: LLMClient | None = None):
    """Yield the explanation text piece by piece. Called *after* the results are shown, so the
    LLM never delays them. Yields nothing if there is nothing to explain or no LLM."""
    llm = llm or get_llm()
    if not res.snippets or not llm.available():
        return
    try:
        yield from llm.stream(explanation_messages(res), config.BUDGET.explanation_max_tokens)
    except LLMUnavailable as exc:
        logger.info("explanation unavailable: %s", exc)


def explain(res: AnswerResult, llm: LLMClient | None = None) -> str:
    """Blocking helper (CLI --json, evaluation): collect the streamed explanation."""
    res.explanation = "".join(stream_explanation(res, llm)).strip()
    return res.explanation


# ------------------------------------------------------------------------------ entry points


def _run_subplan(
    run: _Run, sub: SubPlan, ranking: RankingConfig, refine: RefineConfig, expand: ExpandConfig
) -> AnswerResult:
    sub_res = AnswerResult(query=sub.text, route=sub.route.to_dict())
    view = run.sub(sub_res)
    answered = False
    if sub.route.category in {"structural", "usage"}:
        answered = _deterministic(view, sub.route)
        if answered:
            retriever = code_tools.get_context().retriever
            _read_step(view, _hits_for_snippets(retriever, sub_res.snippets))
            view.step(
                "verify",
                "citations",
                inputs={"attempted": view.attempted},
                outputs={"verified": view.verified, "dropped": view.attempted - view.verified},
                result_count=len(sub_res.snippets),
                summary=f"{view.verified}/{view.attempted} candidate ranges re-read and confirmed",
            )
        else:
            view.step(
                "plan",
                "replan: deterministic lookup empty → hybrid search",
                inputs={"route": sub.route.type},
                ok=False,
            )
    if not answered:
        _semantic_first(view, sub.route.params.get("query", sub.text), ranking, refine, expand)
    return sub_res


def _assemble(res: AnswerResult, subs: list[AnswerResult]) -> None:
    per = max(3, (SUPPORT_SNIPPETS + 3) // max(1, len(subs))) if len(subs) > 1 else None
    res.snippets = [sn for s in subs for sn in (s.snippets[:per] if per else s.snippets)]
    res.call_paths = [p for s in subs for p in s.call_paths]
    res.answer = "\n\n".join(
        (f"**{s.query}** — " if len(subs) > 1 else "") + s.answer for s in subs if s.answer
    )
    order = {"high": 2, "medium": 1, "low": 0}
    res.confidence = min((s.confidence for s in subs), key=lambda c: order.get(c, 0))
    res.route = (
        subs[0].route
        if len(subs) == 1
        else {
            "type": "multi",
            "category": "multi",
            "method": "rules",
            "params": {"parts": [s.route for s in subs]},
        }
    )
    for s in subs:
        if s.route.get("rewritten_query"):
            res.route["rewritten_query"] = s.route["rewritten_query"]


def answer(
    query: str,
    use_llm: bool = True,
    optimize: bool = False,
    ranking: RankingConfig | None = None,
    budget: AgentBudget | None = None,
    llm: LLMClient | None = None,
    on_step: Callable[[Step], None] | None = None,
    refine: RefineConfig | None = None,
    expand: ExpandConfig | None = None,
    llm_phase: bool = False,
) -> AnswerResult:
    """Phase A: plan → search → read → assess → refine round 1 → verify → first results.
    With `llm_phase=True`, phase B (`continue_with_llm`) runs before returning. Never raises for
    ordinary failures."""
    ranking = ranking or config.AGENT_RANKING
    budget = budget or config.BUDGET
    refine = refine or config.REFINE
    expand = expand or config.EXPAND
    llm_client = (llm or get_llm()) if use_llm else None
    res = AnswerResult(query=query, route={})
    run = _Run(res, llm_client, budget, on_step)
    code_tools.get_context().ranking = ranking
    try:
        t = time.perf_counter()
        plan = plan_query(query)
        res.plan = plan.to_dict()
        run.step(
            "plan",
            "decompose" if plan.decomposed else "classify",
            inputs={"query": query},
            outputs=res.plan,
            result_count=len(plan.subplans),
            summary=" | ".join(f"{s.route.type}: " + " → ".join(s.steps) for s in plan.subplans),
            elapsed_ms=round((time.perf_counter() - t) * 1000, 1),
        )
        subs = [_run_subplan(run, sp, ranking, refine, expand) for sp in plan.subplans]
        res.state = {
            "subs": subs,
            "plan": plan,
            "ranking": ranking,
            "refine": refine,
            "expand": expand,
        }
        _assemble(res, subs)
        res.refinement_rounds = sum(s.state.get("rounds", 0) for s in subs)
        if optimize and res.snippets:
            from src.optimize.checks import suggest_optimizations

            # Static checks only here; their LLM explanation runs in phase B.
            res.optimizations = suggest_optimizations(res.snippets, None)
            run.step(
                "search",
                "optimize",
                outputs={"suggestions": len(res.optimizations)},
                result_count=len(res.optimizations),
                summary=f"{len(res.optimizations)} suggestions",
            )
    except Exception as exc:  # noqa: BLE001 — the UI must get a message, not a stack trace
        logger.exception("answer failed")
        res.error = f"{type(exc).__name__}: {exc}"
        res.answer = res.answer or "Something went wrong while answering; see the error field."
    res.latency_ms = res.final_latency_ms = round(run.elapsed() * 1000, 1)
    run.step(
        "stop",
        "first results",
        outputs={"snippets": len(res.snippets), "confidence": res.confidence},
        result_count=len(res.snippets),
        elapsed_ms=res.latency_ms,
        summary=f"{len(res.snippets)} verified snippets in {res.latency_ms / 1000:.2f} s",
    )
    if llm_phase and use_llm:
        continue_with_llm(res, llm_client, on_step)
    return res


def continue_with_llm(
    res: AnswerResult, llm: LLMClient | None = None, on_step: Callable[[Step], None] | None = None
) -> bool:
    """Phase B (after the first results are displayed): LLM planning fallback for unparsed
    relational questions, then refinement round 2 (constrained rewrite) for low-confidence
    semantic sub-plans. Updates `res` in place; returns True if the results changed."""
    st = res.state
    if not st or res.error:
        return False
    llm = llm or get_llm()
    run = _Run(res, llm, config.BUDGET, on_step, phase="refine")
    t0 = time.perf_counter()
    changed = False
    try:
        if not llm.available():
            run.step(
                "stop",
                "final results",
                summary="LLM unavailable: first results are final",
                ok=False,
            )
            return False
        subs: list[AnswerResult] = st["subs"]
        for i, (sp, sub_res) in enumerate(zip(st["plan"].subplans, subs, strict=True)):
            if sp.route.params.get("needs_extraction"):
                t = time.perf_counter()
                route = llm_plan(sp.text, llm)
                res.llm_used = True
                run.step(
                    "plan",
                    "LLM planning fallback",
                    inputs={"clause": sp.text},
                    outputs={"route": route.to_dict() if route else None},
                    summary=f"→ {route.type} {route.params}" if route else "no structure",
                    elapsed_ms=round((time.perf_counter() - t) * 1000, 1),
                    ok=bool(route),
                )
                if route is not None and route.category != "semantic":
                    new_sub = _run_subplan(
                        run, SubPlan(sp.text, route), st["ranking"], st["refine"], st["expand"]
                    )
                    if new_sub.snippets and new_sub.state.get("kind") != "semantic":
                        subs[i] = new_sub
                        changed = True
                        continue
            changed |= _semantic_rewrite(run.sub(sub_res), sub_res, st["ranking"], st["refine"])
        if changed:
            _assemble(res, subs)
        res.refinement_rounds = sum(s.state.get("rounds", 0) for s in subs)
        res.results_changed_by_llm = changed
        if res.optimizations and not any("explanation" in o for o in res.optimizations):
            from src.optimize.checks import explain_findings, suggest_optimizations

            t = time.perf_counter()
            if changed:  # the surfaced code changed: recompute the static findings first
                res.optimizations = suggest_optimizations(res.snippets, None)
            n = explain_findings(res.optimizations, llm)
            res.llm_used = res.llm_used or n > 0
            run.step(
                "llm",
                "explain optimisation findings",
                inputs={"findings": len(res.optimizations)},
                outputs={"explained": n},
                result_count=n,
                summary=f"{n} finding(s) explained from the shown code only",
                elapsed_ms=round((time.perf_counter() - t) * 1000, 1),
                ok=n > 0,
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("phase B failed")
        run.step("stop", "phase B error", summary=str(exc), ok=False)
    res.final_latency_ms = round(res.latency_ms + (time.perf_counter() - t0) * 1000, 1)
    run.step(
        "stop",
        "final results",
        outputs={"changed": changed, "snippets": len(res.snippets)},
        result_count=len(res.snippets),
        elapsed_ms=res.final_latency_ms,
        summary=("results updated" if changed else "first results confirmed")
        + f" after {res.refinement_rounds} refinement round(s)",
    )
    return changed
