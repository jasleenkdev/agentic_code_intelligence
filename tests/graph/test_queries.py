from src.graph.queries import find_call_sequence, find_symbols, get_callees, get_callers


def test_find_symbols_dotted_via_binding(mini_db):
    syms = find_symbols(mini_db, "RadioManager.connect")
    assert [s["file_path"] for s in syms] == ["server/lib/radio/radio.connect.js"]
    assert find_symbols(mini_db, "EventManager.emit")[0]["qualname"] == "EventManager.emit"


def test_get_callers_and_callees(mini_db):
    callers = get_callers(mini_db, "prepare")["callers"]
    assert {c["caller"] for c in callers} >= {"scan", "connect"}
    callees = get_callees(mini_db, "scan")
    texts = [c["call_text"] for c in callees["callees"]]
    assert texts == ["this.prepare", "this.eventManager.emit", "this.listIds", "this.connect"]


def test_call_sequence_certain_vs_possible(mini_db):
    res = find_call_sequence(mini_db, "prepare", "listIds")
    m = res["matches"][0]
    assert m["function"] == "scan" and m["certainty"] == "certain" and m["hops"] == 0
    assert (m["first_call"]["line"], m["second_call"]["line"]) == (7, 11)
    res = find_call_sequence(mini_db, "emit", "listIds")
    assert res["matches"][0]["certainty"] == "possible"
    assert "conditional branch" in res["matches"][0]["reasons"][0]
    assert find_call_sequence(mini_db, "listIds", "prepare")["total"] == 0


def test_call_sequence_follows_hops(mini_db):
    # scan calls this.connect (in a loop) which calls this.prepare → prepare inside connect,
    # reached after listIds: listIds happens before prepare via one hop.
    direct = find_call_sequence(mini_db, "listIds", "prepare", same_function=True)
    assert direct["total"] == 0
    hops = find_call_sequence(mini_db, "listIds", "prepare", same_function=False)
    m = hops["matches"][0]
    assert m["function"] == "scan" and m["hops"] == 1
    assert m["second_call"]["via"] == ["connect"]
    assert m["certainty"] == "possible"  # connect() is inside a loop body and a heuristic hop
