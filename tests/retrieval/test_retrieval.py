from src.config import RankingConfig
from src.retrieval.bm25 import split_identifier, tokenize_code, tokenize_query
from src.retrieval.chunker import _windows
from src.retrieval.search import Retriever, identifier_like


def test_identifier_splitting():
    assert split_identifier("getDeviceFeatureStates") == ["get", "device", "feature", "states"]
    assert split_identifier("HTTPClient_retry") == ["http", "client", "retry"]
    toks = tokenize_code("const x = this.eventManager.emit(EVENTS.RADIO)")
    assert "eventmanager" in toks and "event" in toks and "manager" in toks
    assert "const" not in toks
    assert tokenize_query("where do we emit the radio event") == ["emit", "radio", "event"]


def test_windows_overlap():
    assert _windows(1, 50, 120, 10) == [(1, 50)]
    assert _windows(1, 250, 120, 10) == [(1, 120), (111, 230), (221, 250)]


def test_identifier_like():
    assert identifier_like("who calls setValue before emit() on DeviceManager") == [
        "setValue",
        "emit",
        "DeviceManager",
    ]


def test_chunks_cover_functions_and_module_code(mini_db):
    rows = mini_db.execute(
        "SELECT f.path, c.start_line, c.end_line, s.qualname FROM chunks c JOIN files f ON"
        " f.id = c.file_id JOIN symbols s ON s.id = c.symbol_id"
    ).fetchall()
    got = {(r["path"], r["qualname"]) for r in rows}
    assert ("server/lib/radio/radio.scan.js", "scan") in got
    assert ("server/lib/constants.js", "<module>") in got


def test_bm25_hybrid_search_without_vectors(mini_db):
    r = Retriever(mini_db)
    hits = r.search("scan radio peripherals", k=3, cfg=RankingConfig(use_vectors=False))
    assert hits[0].file_path == "server/lib/radio/radio.scan.js"
    assert hits[0].ranks.get("bm25") == 1
    filtered = r.search("scan", k=5, cfg=RankingConfig(use_vectors=False), path_prefix="front/")
    assert all(h.file_path.startswith("front/") for h in filtered)


def test_structural_list_boosts_exact_symbol(mini_db):
    r = Retriever(mini_db)
    cfg = RankingConfig(use_vectors=False, use_graph_boost=True)
    hits = r.search("what does listIds() return", k=3, cfg=cfg)
    assert hits[0].symbol == "listIds" and "structural" in hits[0].ranks
