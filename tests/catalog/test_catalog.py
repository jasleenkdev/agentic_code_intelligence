from src.catalog.builder import _flatten_constants, _route_regex
from src.catalog.search import parse_query, search_catalog


def test_flatten_constants():
    src = b"const EVENTS = { RADIO: { SCAN: 'radio.scan' }, X: 1 };\nconst y = { A: 'b' };"
    assert _flatten_constants(src) == [("EVENTS.RADIO.SCAN", "radio.scan", 1)]


def test_route_regex():
    rx = _route_regex("/dashboard/integration/device/bluetooth/:deviceSelector")
    assert rx.match("/dashboard/integration/device/bluetooth/X")
    assert not rx.match("/dashboard/integration/device/bluetooth/X/edit")


def test_parse_query_kind_hints_and_synonyms():
    kinds, groups = parse_query("where is the Bluetooth-settings deeplink used?")
    assert kinds == {"front_route"}
    assert {"bluetooth"} in groups and {"config", "setting"} in groups


def test_catalog_event_roles_and_routes(mini_db):
    rows = mini_db.execute("SELECT kind, name, role FROM catalog").fetchall()
    got = {(r["kind"], r["name"], r["role"]) for r in rows}
    assert ("event", "EVENTS.RADIO.SCAN_STARTED", "definition") in got
    assert ("event", "EVENTS.RADIO.SCAN_STARTED", "emit") in got
    assert ("uri", "radio://settings", "usage") in got


def test_search_catalog_finds_event(mini_db):
    hits = search_catalog(mini_db, "where is the radio scan started event emitted")
    assert hits and hits[0].name == "EVENTS.RADIO.SCAN_STARTED"
    assert hits[0].definitions[0]["file_path"] == "server/lib/constants.js"
    assert any(u["role"] == "emit" for u in hits[0].usages)
