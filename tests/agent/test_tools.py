import pytest

from src.agent.tools import code_tools
from src.agent.tools.registry import ToolArgError, call_tool, validate_args
from tests.conftest import FIXTURE_REPO


@pytest.fixture()
def tool_ctx(mini_db):
    code_tools.set_context(code_tools.ToolContext(mini_db, FIXTURE_REPO))
    yield
    code_tools.set_context(None)


def test_validate_args_coerces_and_rejects():
    assert validate_args(
        "read_file_range", {"file_path": "a.js", "start_line": "3", "end_line": 4.0}
    ) == {"file_path": "a.js", "start_line": 3, "end_line": 4}
    assert validate_args("find_string_usages", {"text": "x", "exact": "true"})["exact"] is True
    with pytest.raises(ToolArgError):
        validate_args("get_callers", {})
    with pytest.raises(ToolArgError):
        validate_args("nope", {})


def test_read_file_range_is_confined_and_exact(tool_ctx):
    r, _ = call_tool(
        "read_file_range",
        {"file_path": "server/lib/radio/radio.scan.js", "start_line": 6, "end_line": 7},
    )
    assert r["text"] == "async function scan(timeout) {\n  await this.prepare();"
    r, _ = call_tool(
        "read_file_range", {"file_path": "../conftest.py", "start_line": 1, "end_line": 2}
    )
    assert "error" in r


def test_find_string_usages_literal_const_and_catalog(tool_ctx):
    r, _ = call_tool("find_string_usages", {"text": "/dashboard/integration/device/radio"})
    assert "front/src/RadioPage.jsx" in r["files"]
    r, _ = call_tool("find_string_usages", {"text": "EVENTS.RADIO.SCAN_STARTED", "exact": True})
    assert r["files"]["server/lib/radio/radio.scan.js"][0]["kind"] == "constant"


def test_all_eight_tools_run(tool_ctx):
    calls = {
        "semantic_search": {"query": "scan radio"},
        "keyword_search": {"query": "listIds"},
        "find_symbol": {"name": "scan"},
        "get_callers": {"symbol": "connect"},
        "get_callees": {"symbol": "scan"},
        "find_call_sequence": {"first": "prepare", "second": "connect"},
        "find_string_usages": {"text": "radio"},
        "read_file_range": {"file_path": "front/src/utils.js", "start_line": 1, "end_line": 3},
    }
    for name, args in calls.items():
        result, ms = call_tool(name, args)
        assert "error" not in result, (name, result)
        assert ms >= 0
