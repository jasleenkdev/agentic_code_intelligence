def _resolved(conn, caller: str, callee: str):
    return conn.execute(
        "SELECT c.resolution, s.qualname, f.path FROM calls c JOIN symbols caller ON caller.id ="
        " c.caller_id LEFT JOIN symbols s ON s.id = c.callee_symbol_id LEFT JOIN files f ON"
        " f.id = s.file_id WHERE caller.name = ? AND c.callee_name = ?",
        (caller, callee),
    ).fetchone()


def test_this_method_resolves_through_prototype_binding(mini_db):
    row = _resolved(mini_db, "scan", "connect")
    assert row["resolution"] == "heuristic"
    assert row["path"] == "server/lib/radio/radio.connect.js"


def test_member_on_injected_manager_resolves_by_owner(mini_db):
    row = _resolved(mini_db, "scan", "emit")
    assert row["qualname"] == "EventManager.emit"


def test_esm_named_import_resolves_statically(mini_db):
    row = _resolved(mini_db, "render", "formatName")
    assert row["resolution"] == "static" and row["path"] == "front/src/utils.js"


def test_builtins_are_external(mini_db):
    row = _resolved(mini_db, "formatName", "toUpperCase")
    assert row["resolution"] == "external"


def test_edges_written(mini_db):
    n = mini_db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    assert n >= 5
