"""API tests against the fixture index, with a fake LLM (no Ollama needed)."""

import json

import pytest
from fastapi.testclient import TestClient

from src.agent.llm import LLMResponse
from src.agent.tools import code_tools
from src.api import app as api_mod
from tests.conftest import FIXTURE_REPO


class FakeLLM:
    model = "fake"

    def __init__(self, up=True):
        self.up = up

    def available(self):
        return self.up

    def chat(self, messages, json_mode=True, max_tokens=None, model=None):
        return LLMResponse(
            text=json.dumps({"picks": [], "explanations": ["Use Promise.all."]}), model="fake"
        )

    def stream(self, messages, max_tokens=220):
        yield from ["listIds ", "is called ", "by scan."]


@pytest.fixture()
def client(mini_db, mini_index, monkeypatch):
    from src import config
    from src.agent import llm as llm_mod

    monkeypatch.setattr(config, "REPO_ROOT", FIXTURE_REPO)
    monkeypatch.setattr(config, "DB_PATH", mini_index / "code.db")
    code_tools.set_context(code_tools.ToolContext(mini_db, FIXTURE_REPO))
    fake = FakeLLM(up=False)
    monkeypatch.setattr(llm_mod, "_DEFAULT", fake)
    c = TestClient(api_mod.app)
    c.fake_llm = fake
    yield c
    code_tools.set_context(None)


def _events(resp) -> list[tuple[str, dict]]:
    out = []
    for block in resp.text.split("\n\n"):
        lines = [ln for ln in block.splitlines() if not ln.startswith(":")]
        if not lines:
            continue
        event = next(ln[7:] for ln in lines if ln.startswith("event: "))
        data = json.loads(next(ln[6:] for ln in lines if ln.startswith("data: ")))
        out.append((event, data))
    return out


def test_health_stats_and_openapi(client):
    assert client.get("/api/health").json() == {"ok": True}
    s = client.get("/api/stats").json()
    assert s["files"] >= 7 and s["functions"] > 0 and s["llm"]["available"] is False
    paths = client.get("/openapi.json").json()["paths"]
    for p in (
        "/api/ask",
        "/api/transcribe",
        "/api/file",
        "/api/catalog",
        "/api/results/{rid}/callpaths",
        "/api/results/{rid}/optimizations",
    ):
        assert p in paths


def test_ask_streams_trace_then_results_without_llm(client):
    r = client.post("/api/ask", json={"query": "which functions call prepare before connect?"})
    assert r.headers["content-type"].startswith("text/event-stream")
    ev = _events(r)
    kinds = [e for e, _ in ev]
    assert kinds[0] == "step" and kinds[-1] == "done"
    steps = [d["kind"] for e, d in ev if e == "step"]
    assert steps[0] == "plan" and "search" in steps and "verify" in steps and "stop" in steps
    first = next(d for e, d in ev if e == "results")
    assert first["phase"] == "first" and first["snippets"]
    assert first["snippets"][0]["file_path"] == "server/lib/radio/radio.scan.js"
    assert "explanation" not in kinds  # LLM down: results but no explanation
    assert ev[-1][1]["llm_available"] is False


def test_ask_with_llm_streams_explanation_after_results(client):
    client.fake_llm.up = True
    ev = _events(client.post("/api/ask", json={"query": "who calls listIds?", "explain": True}))
    kinds = [e for e, _ in ev]
    assert kinds.index("results") < kinds.index("explanation")
    assert "".join(d["text"] for e, d in ev if e == "explanation") == "listIds is called by scan."
    assert [d["phase"] for e, d in ev if e == "results"] == ["first", "final"]


def test_ask_validation_error_is_friendly(client):
    r = client.post("/api/ask", json={"query": ""})
    assert r.status_code == 422


def test_callpaths_and_optimizations(client):
    ev = _events(client.post("/api/ask", json={"query": "who calls listIds?"}))
    rid = ev[-1][1]["id"]
    g = client.get(f"/api/results/{rid}/callpaths").json()
    assert {"source": "scan", "target": "this.listIds"} in g["edges"]
    ev = _events(client.post("/api/ask", json={"query": "sync radio status settings loop"}))
    rid = ev[-1][1]["id"]
    o = client.get(f"/api/results/{rid}/optimizations?explain=false").json()
    assert "findings" in o and o["llm_available"] is False
    assert client.get("/api/results/nope/callpaths").status_code == 404


def test_file_viewer_is_confined(client):
    r = client.get(
        "/api/file", params={"path": "server/lib/radio/radio.scan.js", "start": 7, "context": 1}
    ).json()
    assert r["start_line"] == 6 and r["highlight"] == [7, 7]
    assert r["lines"][1] == "  await this.prepare();"
    assert client.get("/api/file", params={"path": "../conftest.py"}).status_code == 404
    assert (
        client.get("/api/file", params={"path": "front/src/utils.js", "start": 999}).status_code
        == 416
    )


def test_catalog_endpoints(client):
    kinds = {k["kind"] for k in client.get("/api/catalog/kinds").json()}
    assert "event" in kinds
    items = client.get("/api/catalog", params={"kind": "event"}).json()["items"]
    assert items[0]["name"] == "EVENTS.RADIO.SCAN_STARTED" and items[0]["usages"] >= 1
    e = client.get(
        "/api/catalog/entry", params={"kind": "event", "name": "EVENTS.RADIO.SCAN_STARTED"}
    ).json()
    assert e["definitions"][0]["file_path"] == "server/lib/constants.js"
    assert any(u["role"] == "emit" for u in e["usages"])
    assert (
        client.get("/api/catalog/entry", params={"kind": "event", "name": "nope"}).status_code
        == 404
    )


def test_transcribe(client, monkeypatch):
    assert (
        client.post("/api/transcribe", files={"audio": ("a.webm", b"", "audio/webm")}).status_code
        == 422
    )
    import src.speech.stt as stt

    monkeypatch.setattr(stt, "transcribe", lambda data: ("who calls purgeStates", 12.0))
    r = client.post("/api/transcribe", files={"audio": ("a.webm", b"xx", "audio/webm")})
    assert r.json() == {"text": "who calls purgeStates", "elapsed_ms": 12.0}


def test_parallel_read_requests_do_not_share_a_connection(client):
    """The Explore page fires catalog, kinds and stats at once; this used to crash the server."""
    from concurrent.futures import ThreadPoolExecutor

    urls = ["/api/catalog/kinds", "/api/catalog?limit=50", "/api/stats", "/api/catalog?q=scan"] * 25
    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(lambda u: client.get(u).status_code, urls))
    assert set(codes) == {200}
