"""Versioned indexing end to end on a throw-away git repo (uses the small bge-small model)."""

import importlib
import subprocess

import pytest

pytestmark = pytest.mark.slow

V1 = {
    "server/lib/math.js": "function add(a, b) {\n  const s = a + b;\n  return s;\n}\n\n"
    "function mul(a, b) {\n  const p = a * b;\n  return p;\n}\n",
    "server/lib/text.js": "function shout(s) {\n  const u = s.toUpperCase();\n"
    "  return u + '!';\n}\n",
}


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    for path, text in V1.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v1")
    _git(repo, "tag", "v1")
    (repo / "server/lib/text.js").write_text(
        "function shout(s) {\n  const upper = s.toUpperCase();\n  return upper + '!!';\n}\n"
    )
    _git(repo, "commit", "-qam", "v2")
    _git(repo, "tag", "v2")
    monkeypatch.setenv("CODESEARCH_VERSIONS_DIR", str(tmp_path / "versions"))
    monkeypatch.setenv("CODESEARCH_CHUNK_CACHE", str(tmp_path / "chunks"))
    monkeypatch.setenv("CODESEARCH_EMB_CACHE", str(tmp_path / "emb"))
    import src.codesearch.encoders as enc
    import src.codesearch.engine as eng
    import src.codesearch.versions as vs

    importlib.reload(enc)
    importlib.reload(vs)
    importlib.reload(eng)
    yield repo / ".git", vs, eng
    importlib.reload(enc)
    importlib.reload(vs)
    importlib.reload(eng)


def test_incremental_versions_and_all_versions_search(env):
    git, vs, eng = env
    s1, s2 = {}, {}
    sn1 = vs.snippets_from_git(git, "v1", include=("server",), stats=s1)
    assert {s.symbol for s in sn1} == {"add", "mul", "shout"}
    m1 = vs.index_version("toy", "v1", sn1, ["bge-small"], {"kind": "codebase", "order": 1}, s1)
    sn2 = vs.snippets_from_git(git, "v2", include=("server",), stats=s2)
    assert s2["files_parsed"] == 1 and s2["files_reused"] == 1  # only text.js changed
    m2 = vs.index_version("toy", "v2", sn2, ["bge-small"], {"kind": "codebase", "order": 2}, s2)
    assert m1["encode"]["bge-small"]["encoded"] == 3
    assert m2["encode"]["bge-small"]["encoded"] == 1  # only the changed function
    engine = eng.SearchEngine()
    res = engine.search("uppercase a string and add an exclamation mark", "toy", "v1", k=2)
    assert res["results"][0]["symbol"] == "shout" and res["results"][0]["version"] == "v1"
    assert [p["name"] for p in res["passes"]][:2] == ["load index", "pre-process"]
    allv = engine.search("uppercase a string (original version)", "toy", "all", k=3)
    top = allv["results"][0]
    assert top["symbol"] == "shout" and top["version"] == "v1"
    assert {m["version"] for m in top["lineage"]["versions"]} == {"v1", "v2"}
    assert len({r["lineage"]["id"] for r in allv["results"]}) == len(allv["results"])
