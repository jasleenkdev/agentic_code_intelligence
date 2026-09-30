from src.agent.verifier import verify
from src.optimize.checks import static_findings, suggest_optimizations
from tests.conftest import FIXTURE_REPO


def _snips(*ranges):
    return [verify(p, a, b, repo_root=FIXTURE_REPO) for p, a, b in ranges]


def test_static_findings_on_sync(mini_db):
    snips = _snips(
        ("server/lib/radio/radio.sync.js", 3, 11), ("server/lib/radio/radio.sync.js", 13, 15)
    )
    found = static_findings(snips, conn=mini_db, repo_root=FIXTURE_REPO)
    got = {(f["check"], f["line"]) for f in found}
    assert ("sequential-awaits", 4) in got
    assert ("await-in-loop", 7) in got
    assert ("loop-invariant-call", 7) in got
    assert ("uncalled-function", 13) in got
    assert all(f["advisory"] and f["evidence"] for f in found)


def test_dependent_awaits_not_flagged(mini_db):
    # scan(): `const ids = await this.listIds()` is followed by a loop that uses ids.
    found = static_findings(
        _snips(("server/lib/radio/radio.scan.js", 6, 16)), conn=mini_db, repo_root=FIXTURE_REPO
    )
    assert not [f for f in found if f["check"] == "sequential-awaits"]
    assert [f for f in found if f["check"] == "await-in-loop"]
    # exported, called functions are never reported as uncalled
    assert not [f for f in found if f["check"] == "uncalled-function"]


def test_suggest_without_llm_returns_templates(mini_db, monkeypatch):
    from src import config
    from src.indexer import store

    monkeypatch.setattr(config, "REPO_ROOT", FIXTURE_REPO)
    monkeypatch.setattr(store, "connect", lambda *a, **k: mini_db)
    out = suggest_optimizations(_snips(("server/lib/radio/radio.sync.js", 3, 11)), llm=None)
    assert out and all("message" in f and "explanation" not in f for f in out)
