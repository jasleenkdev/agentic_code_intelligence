"""Planner-driven agent tests with a scripted fake LLM (no Ollama needed)."""

import json

import pytest

from src.agent import agent as agent_mod
from src.agent.classifier import classify
from src.agent.expand import ExpandConfig, expand_and_rescore
from src.agent.llm import LLMResponse, LLMUnavailable, parse_json_object
from src.agent.planner import plan_query, split_clauses
from src.agent.refine import RefineConfig, assess, merge_rounds, rewrite_query
from src.agent.tools import code_tools
from src.config import RankingConfig
from src.retrieval.search import Hit, Retriever
from tests.conftest import FIXTURE_REPO

NO_VEC = RankingConfig(use_vectors=False)
ALWAYS_EXPAND = RefineConfig(gate="always", rewrite=False)
ALWAYS_REWRITE = RefineConfig(gate="always", expand=False)
OFF = RefineConfig(expand=False, rewrite=False)


class FakeLLM:
    model = "fake"

    def __init__(self, replies=(), up=True, stream_text="scan() lists ids at radio.scan.js:11"):
        self.replies = list(replies)
        self.up = up
        self.stream_text = stream_text
        self.calls = []

    def available(self):
        return self.up

    def chat(self, messages, json_mode=True, max_tokens=None, model=None):
        self.calls.append(messages)
        if not self.replies:
            raise LLMUnavailable("no more scripted replies")
        return LLMResponse(
            text=self.replies.pop(0), model="fake", prompt_tokens=10, completion_tokens=5
        )

    def stream(self, messages, max_tokens=220):
        self.calls.append(messages)
        yield from self.stream_text.split(" ")


@pytest.fixture(autouse=True)
def ctx(mini_db, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "REPO_ROOT", FIXTURE_REPO)
    code_tools.set_context(code_tools.ToolContext(mini_db, FIXTURE_REPO))
    yield
    code_tools.set_context(None)


def _kinds(res, phase="first"):
    return [s.kind for s in res.steps if s.phase == phase]


# ------------------------------------------------------------------------------ planning


def test_classifier_routes():
    assert classify("which files call tool prepare before tool connect?").params == {
        "first": "prepare",
        "second": "connect",
    }
    assert classify("who calls listIds?").type == "callers"
    assert classify("where is EVENTS.RADIO.SCAN_STARTED used").params["exact"] is True
    assert classify("how does scanning work?").category == "semantic"


def test_plan_decomposes_and_resolves_pronouns():
    plan = plan_query("Who calls listIds and what does it call?")
    assert plan.decomposed and [s.route.type for s in plan.subplans] == ["callers", "callees"]
    assert plan.subplans[1].route.params["symbol"] == "listIds"
    seq = plan_query("Which files call prepare before connect?").subplans[0]
    assert seq.steps[:3] == ["resolve prepare", "resolve connect", "find call sequences"]
    assert split_clauses("how does scanning work") == ["how does scanning work"]


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": {"b": 2}} hope this helps') == {"a": {"b": 2}}


# ------------------------------------------------------------------------------ phase A


def test_structural_trace_plan_search_read_verify_stop():
    r = agent_mod.answer(
        "which functions call prepare before connect?", use_llm=False, ranking=NO_VEC
    )
    kinds = _kinds(r)
    assert kinds[0] == "plan" and kinds[-1] == "stop"
    names = [s.name for s in r.steps]
    assert names.count("find_symbol") == 2 and "find_call_sequence" in names
    assert "read" in kinds and "verify" in kinds
    assert r.snippets[0].file_path == "server/lib/radio/radio.scan.js"
    assert all(isinstance(s.inputs, dict) and isinstance(s.outputs, dict) for s in r.steps)


def test_multi_part_question_runs_both_subplans():
    r = agent_mod.answer("Who calls listIds and what does it call?", use_llm=False, ranking=NO_VEC)
    assert r.plan["decomposed"] and r.route["type"] == "multi"
    names = [s.name for s in r.steps]
    assert "get_callers" in names and "get_callees" in names
    assert {s.file_path for s in r.snippets} >= {"server/lib/radio/radio.scan.js"}


def test_usage_answer_for_constant():
    r = agent_mod.answer("where is EVENTS.RADIO.SCAN_STARTED used", use_llm=False, ranking=NO_VEC)
    paths = {s.file_path for s in r.snippets}
    assert {"server/lib/constants.js", "server/lib/radio/radio.scan.js"} <= paths


def test_semantic_trace_has_assess_and_expansion_round():
    r = agent_mod.answer(
        "how are radio peripherals scanned?", use_llm=False, ranking=NO_VEC, refine=ALWAYS_EXPAND
    )
    kinds = _kinds(r)
    for k in ("plan", "search", "read", "assess", "refine", "verify", "stop"):
        assert k in kinds, k
    assert r.refinement_rounds == 1 and r.snippets
    assert not r.llm_used


def test_no_refinement_when_disabled():
    r = agent_mod.answer(
        "how are radio peripherals scanned?", use_llm=False, ranking=NO_VEC, refine=OFF
    )
    assert "refine" not in _kinds(r) and r.refinement_rounds == 0


# ------------------------------------------------------------------------------ phase B


def test_constrained_rewrite_uses_only_listed_terms():
    llm = FakeLLM([json.dumps({"picks": [1, 999, "x"]})])
    r = agent_mod.answer(
        "how are radio peripherals scanned?", llm=llm, ranking=NO_VEC, refine=ALWAYS_REWRITE
    )
    assert _kinds(r, "refine") == []  # nothing LLM-related before the first results
    agent_mod.continue_with_llm(r, llm)
    names = [s.name for s in r.steps if s.phase == "refine"]
    assert "round 2: candidate terms from the index" in names
    assert "pick relevant terms (constrained)" in names
    picked = next(s for s in r.steps if s.name == "pick relevant terms (constrained)")
    assert len(picked.outputs["picked"]) == 1  # 999 and "x" were rejected
    prompt = llm.calls[0][0]["content"]
    assert picked.outputs["picked"][0] in prompt  # the pick came from the offered list
    assert r.route.get("rewritten_query")
    assert all(s.verified for s in r.snippets)


def test_llm_planning_fallback_for_unparsed_relational_question():
    q = "tell me about the calls going to listIds"
    assert plan_query(q).needs_llm_planning
    llm = FakeLLM([json.dumps({"type": "callers", "symbol": "listIds"})])
    r = agent_mod.answer(q, llm=llm, ranking=NO_VEC, refine=OFF)
    changed = agent_mod.continue_with_llm(r, llm)
    assert changed and r.results_changed_by_llm
    assert any(s.name == "LLM planning fallback" for s in r.steps)
    assert r.snippets[0].file_path == "server/lib/radio/radio.scan.js"


def test_llm_down_keeps_first_results():
    r = agent_mod.answer(
        "how are radio peripherals scanned?",
        llm=FakeLLM(up=False),
        ranking=NO_VEC,
        refine=ALWAYS_REWRITE,
    )
    before = r.locations
    assert agent_mod.continue_with_llm(r, FakeLLM(up=False)) is False
    assert r.locations == before and r.snippets


def test_explanation_is_streamed_from_snippets_only():
    r = agent_mod.answer("who calls listIds?", use_llm=False, ranking=NO_VEC)
    llm = FakeLLM()
    pieces = list(agent_mod.stream_explanation(r, llm))
    assert "".join(pieces).startswith("scan()")
    prompt = llm.calls[0][0]["content"]
    assert r.snippets[0].location in prompt and "who calls listIds?" in prompt


# ------------------------------------------------------------------------------ helpers


def _hit(cid, score, ranks=None, symbol_id=None):
    return Hit(
        chunk_id=cid,
        file_path="a.js",
        start_line=cid,
        end_line=cid,
        symbol="s",
        kind="function",
        score=score,
        ranks=ranks or {},
        symbol_id=symbol_id,
    )


def test_assess_merge_and_rewrite_query():
    conf = assess([_hit(1, 1.0, {"bm25": 1, "vector": 2}), _hit(2, 0.4)])
    assert conf["label"] == "high" and conf["margin"] == pytest.approx(0.6)
    assert assess([_hit(1, 1.0), _hit(2, 0.95)])["label"] == "low"
    merged = merge_rounds([_hit(1, 1), _hit(2, 1)], [_hit(2, 1), _hit(3, 1)], OFF, 10)
    assert [h.chunk_id for h in merged] == [2, 1, 3]
    assert rewrite_query(
        [
            {"text": "RadioManager.listIds", "kind": "identifier"},
            {"text": "Scan for radio peripherals", "kind": "doc"},
        ]
    ) == ("radio manager list ids Scan for radio peripherals")


def test_expansion_adds_call_graph_neighbours(mini_db):
    r = Retriever(mini_db)
    hits = r.search("scan radio peripherals", 5, NO_VEC)
    out, info = expand_and_rescore(
        r, "scan radio peripherals", hits, ExpandConfig(lam=0.5, seeds=2), 10
    )
    assert info["edges"] > 0
    assert {h.chunk_id for h in hits} <= {h.chunk_id for h in out} | {h.chunk_id for h in hits}


def test_optimisation_findings_explained_in_phase_b_only():
    llm = FakeLLM([json.dumps({"explanations": ["Run both awaits with Promise.all."]})])
    r = agent_mod.answer(
        "who calls unusedHelper or sync? how does sync load config",
        llm=llm,
        ranking=NO_VEC,
        refine=OFF,
        optimize=True,
    )
    if not r.optimizations:  # route-independent: use a question that surfaces radio.sync.js
        r = agent_mod.answer(
            "sync radio status settings loop", llm=llm, ranking=NO_VEC, refine=OFF, optimize=True
        )
    assert r.optimizations and not any("explanation" in o for o in r.optimizations)
    assert not llm.calls  # nothing sent to the LLM before the first results
    agent_mod.continue_with_llm(r, llm)
    assert r.optimizations[0]["explanation"] == "Run both awaits with Promise.all."
    step = next(s for s in r.steps if s.name == "explain optimisation findings")
    assert step.phase == "refine" and step.outputs["explained"] == 1
