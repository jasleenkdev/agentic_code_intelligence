"""Tool specs + argument validation/coercion; `call_tool` never raises (errors become results)."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.agent.tools.code_tools import TOOLS

logger = logging.getLogger(__name__)

# name -> (description, {param: (type, required, default)})
TOOL_SPECS: dict[str, tuple[str, dict[str, tuple[type, bool, Any]]]] = {
    "semantic_search": (
        "Hybrid natural-language code search. Use for 'how/where do we handle X'.",
        {"query": (str, True, None), "k": (int, False, 10), "path_prefix": (str, False, None)},
    ),
    "keyword_search": (
        "Exact identifier/keyword search (BM25). Use when you know a function or word.",
        {"query": (str, True, None), "k": (int, False, 10), "path_prefix": (str, False, None)},
    ),
    "find_symbol": (
        "Find where a function/class/method is defined, e.g. 'setValue' or 'DeviceManager.get'.",
        {"name": (str, True, None), "kind": (str, False, None)},
    ),
    "get_callers": (
        "Who calls a function (from the call graph).",
        {"symbol": (str, True, None), "depth": (int, False, 1)},
    ),
    "get_callees": (
        "What a function calls (from the call graph).",
        {"symbol": (str, True, None), "depth": (int, False, 1)},
    ),
    "find_call_sequence": (
        "Functions where a call to `first` happens before a call to `second`.",
        {
            "first": (str, True, None),
            "second": (str, True, None),
            "same_function": (bool, False, True),
        },
    ),
    "find_string_usages": (
        "Where a string, constant, route/deeplink, event or intent is used.",
        {"text": (str, True, None), "exact": (bool, False, False)},
    ),
    "read_file_range": (
        "Read exact source lines from a file.",
        {
            "file_path": (str, True, None),
            "start_line": (int, True, None),
            "end_line": (int, True, None),
        },
    ),
}


class ToolArgError(ValueError):
    pass


def _coerce(value: Any, typ: type) -> Any:
    if value is None:
        return None
    if typ is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)
    if typ is int:
        return int(float(value))
    if typ is str:
        return value if isinstance(value, str) else str(value)
    return value


def validate_args(name: str, args: Any) -> dict:
    if name not in TOOL_SPECS:
        raise ToolArgError(f"unknown tool '{name}'. Valid tools: {', '.join(TOOL_SPECS)}")
    if not isinstance(args, dict):
        raise ToolArgError("args must be a JSON object")
    params = TOOL_SPECS[name][1]
    out = {}
    for pname, (typ, required, default) in params.items():
        if pname in args and args[pname] not in (None, ""):
            try:
                out[pname] = _coerce(args[pname], typ)
            except (TypeError, ValueError) as exc:
                raise ToolArgError(f"argument '{pname}' must be {typ.__name__}") from exc
        elif required:
            raise ToolArgError(f"missing required argument '{pname}' for {name}")
        else:
            out[pname] = default
    return out


def call_tool(name: str, args: Any) -> tuple[dict, float]:
    """Validate and run a tool. Returns (result, elapsed_ms). Errors come back as {'error': …}."""
    t0 = time.perf_counter()
    try:
        clean = validate_args(name, args)
        result = TOOLS[name](**clean)
    except ToolArgError as exc:
        result = {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — a tool failure must not kill the agent
        logger.exception("tool %s failed", name)
        result = {"error": f"{type(exc).__name__}: {exc}"}
    return result, round((time.perf_counter() - t0) * 1000, 1)
