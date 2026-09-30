from pathlib import Path

from src.indexer.extractor import extract

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mini_repo"


def _facts(rel: str):
    facts, errors = extract((FIXTURE / rel).read_bytes())
    assert errors == 0
    return facts


def test_symbols_have_line_ranges_and_docs():
    facts = _facts("server/lib/radio/radio.scan.js")
    scan = next(s for s in facts.symbols if s["name"] == "scan")
    assert (scan["start_line"], scan["end_line"]) == (6, 16)
    assert scan["is_async"] and scan["is_exported"]
    assert "Scan for radio peripherals" in scan["doc"]


def test_call_order_and_flags():
    facts = _facts("server/lib/radio/radio.scan.js")
    scan_id = next(s["local_id"] for s in facts.symbols if s["name"] == "scan")
    calls = [c for c in facts.calls if c["caller_local"] == scan_id]
    names = [c["callee_name"] for c in calls]
    assert names == ["prepare", "emit", "listIds", "connect"]
    assert [c["ordinal"] for c in calls] == [0, 1, 2, 3]
    by = {c["callee_name"]: c for c in calls}
    assert by["prepare"]["awaited"] and not by["prepare"]["in_branch"]
    assert by["emit"]["in_branch"]
    assert by["connect"]["in_loop"]
    assert by["emit"]["first_arg"] == "EVENTS.RADIO.SCAN_STARTED"


def test_imports_exports_bindings():
    facts = _facts("server/lib/radio/index.js")
    imported = {(i["local_name"], i["source"]) for i in facts.imports}
    assert ("scan", "./radio.scan") in imported
    assert {
        "owner": "RadioManager",
        "member": "scan",
        "target": "scan",
        "line": 8,
    } in facts.bindings
    assert facts.exports[0]["exported_name"] == "default"
    assert facts.module_system == "cjs"


def test_prototype_function_expression_is_method():
    facts = _facts("server/lib/event.js")
    emit = next(s for s in facts.symbols if s["name"] == "emit")
    assert emit["kind"] == "method" and emit["qualname"] == "EventManager.emit"


def test_strings_and_const_refs():
    facts = _facts("server/lib/radio/radio.scan.js")
    values = {s["value"]: s for s in facts.strings}
    assert values["radio://settings/bluetooth"]["context"] == "argument"
    assert any(r["path"] == "EVENTS.RADIO.SCAN_STARTED" for r in facts.const_refs)


def test_jsx_with_class_attribute_parses():
    facts = _facts("front/src/RadioPage.jsx")
    assert any(s["qualname"] == "RadioPage.render" for s in facts.symbols)
    href = next(s for s in facts.strings if s["value"].startswith("/dashboard"))
    assert href["context"] == "jsx_attribute" and href["detail"] == "href"
    assert facts.module_system == "esm"
