"""Domain catalog for the target repo: what the assistant's "tools", intents, events and deeplinks
are, where each is defined, and every place it is used.

Detected from the real Gladys source (see docs/approach.md §A.1):
- ``event`` / ``intent`` / ``action`` / ``ws_message`` / ``job`` / ``constant`` — flattened
  SCREAMING_CASE constant objects (``EVENTS.DEVICE.NEW = 'device.new'``). Usages come from
  constant references; ``emit``/``on`` calls are labelled as emitters/listeners.
- ``mcp_tool`` — LLM tool objects ``{ intent: 'device.turn-on-off', config: { title } }``.
- ``service`` — integration services ``server/services/<name>/index.js``.
- ``api_route`` — HTTP routes declared as object keys ``'get /api/v1/...'``; usages are front-end
  API calls whose path matches the route pattern.
- ``front_route`` — front-end routes / deeplinks (``<Page path="/dashboard/..." />``); usages are
  hrefs, ``route()`` calls, redirects matching the pattern.
- ``uri`` — string literals with a URI scheme (``mqtt://``, ``https://…``).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from src.indexer.extractor import _string_value, _text, make_parser

logger = logging.getLogger(__name__)

GROUP_KIND = {
    "EVENTS": "event",
    "INTENTS": "intent",
    "ACTIONS": "action",
    "WEBSOCKET_MESSAGE_TYPES": "ws_message",
    "JOB_TYPES": "job",
}
EMIT_NAMES = frozenset({"emit"})
LISTEN_NAMES = frozenset({"on", "once", "addListener", "removeListener", "off"})
TOP_CONST_RE = re.compile(rb"^(?:const|let|var)\s+[A-Z][A-Z0-9_]+\s*=\s*\{", re.M)
API_KEY_RE = re.compile(r"^(get|post|put|patch|delete)\s+(/api/\S+)$", re.I)
URI_RE = re.compile(r"^[a-z][a-z0-9+.-]{1,20}://\S+$", re.I)
TEMPLATE_SUB_RE = re.compile(r"\$\{[^}]*\}")
HTTP_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})


def _flatten_constants(src: bytes) -> list[tuple[str, str, int]]:
    """[(path, string_value, line)] for top-level `const NAME = { ... }` objects."""
    parser = make_parser()
    root = parser.parse(src).root_node
    out: list[tuple[str, str, int]] = []
    for stmt in root.named_children:
        if stmt.type not in {"lexical_declaration", "variable_declaration"}:
            continue
        for decl in stmt.named_children:
            if decl.type != "variable_declarator":
                continue
            name = decl.child_by_field_name("name")
            value = decl.child_by_field_name("value")
            if name is None or value is None or value.type != "object":
                continue
            base = _text(name, src)
            if not re.match(r"^[A-Z][A-Z0-9_]+$", base):
                continue
            stack = [(value, base)]
            while stack:
                obj, prefix = stack.pop()
                for pair in obj.named_children:
                    if pair.type != "pair":
                        continue
                    key = pair.child_by_field_name("key")
                    val = pair.child_by_field_name("value")
                    if key is None or val is None:
                        continue
                    path = f"{prefix}.{_text(key, src).strip(chr(39) + chr(34))}"
                    if val.type == "object":
                        stack.append((val, path))
                    elif val.type == "string":
                        out.append((path, _string_value(val, src), pair.start_point[0] + 1))
    return out


def _route_regex(pattern: str) -> re.Pattern:
    parts = re.split(r"(:[A-Za-z_][A-Za-z0-9_]*\??)", pattern.rstrip("/"))
    rx = "".join("[^/]+" if p.startswith(":") else re.escape(p) for p in parts)
    return re.compile(rf"^{rx}/?$")


def _usage_path(value: str) -> str:
    """Normalise a used path: template substitutions → placeholder, drop query/hash."""
    v = TEMPLATE_SUB_RE.sub("X", value)
    return v.split("?")[0].split("#")[0]


class CatalogBuilder:
    def __init__(self, conn: sqlite3.Connection, repo_root: Path) -> None:
        self.conn = conn
        self.repo_root = repo_root
        self.rows: list[tuple] = []
        self.file_ids = {r["path"]: r["id"] for r in conn.execute("SELECT id, path FROM files")}

    def _add(self, kind, name, value, file_id, line, role, symbol_id=None, detail=None) -> None:
        self.rows.append((kind, name, value, file_id, line, role, symbol_id, detail))

    def _enclosing_symbol(self, file_id: int, line: int) -> int | None:
        row = self.conn.execute(
            "SELECT id FROM symbols WHERE file_id = ? AND start_line <= ? AND end_line >= ?"
            " ORDER BY (end_line - start_line) ASC LIMIT 1",
            (file_id, line, line),
        ).fetchone()
        return row["id"] if row else None

    # --------------------------------------------------------------- constants/events/intents
    def constants(self) -> None:
        defs: dict[str, tuple[str, int, int]] = {}
        for path, file_id in self.file_ids.items():
            full = self.repo_root / path
            try:
                src = full.read_bytes()
            except OSError:
                continue
            if not TOP_CONST_RE.search(src):
                continue
            for cpath, value, line in _flatten_constants(src):
                group = cpath.split(".")[0]
                kind = GROUP_KIND.get(group, "constant")
                defs[cpath] = (value, file_id, line)
                self._add(
                    kind,
                    cpath,
                    value,
                    file_id,
                    line,
                    "definition",
                    self._enclosing_symbol(file_id, line),
                )
        # Usages via constant references, labelled emit/listen when they are an event argument.
        call_role: dict[tuple[int, int, str], str] = {}
        for r in self.conn.execute(
            "SELECT file_id, line, callee_name, first_arg FROM calls WHERE first_arg IS NOT NULL"
        ):
            if r["callee_name"] in EMIT_NAMES:
                call_role[(r["file_id"], r["line"], r["first_arg"])] = "emit"
            elif r["callee_name"] in LISTEN_NAMES:
                call_role[(r["file_id"], r["line"], r["first_arg"])] = "listen"
        for r in self.conn.execute("SELECT file_id, symbol_id, path, line FROM const_refs"):
            if r["path"] not in defs:
                continue
            value = defs[r["path"]][0]
            kind = GROUP_KIND.get(r["path"].split(".")[0], "constant")
            role = call_role.get((r["file_id"], r["line"], r["path"]), "reference")
            self._add(kind, r["path"], value, r["file_id"], r["line"], role, r["symbol_id"])
        # Literal usages of event / intent values (front-end websocket handlers, emit('x')).
        by_value = defaultdict(list)
        for cpath, (value, _, _) in defs.items():
            if GROUP_KIND.get(cpath.split(".")[0]) in {"event", "intent", "ws_message", "action"}:
                by_value[value].append(cpath)
        if by_value:
            for r in self.conn.execute(
                "SELECT file_id, symbol_id, value, line FROM strings WHERE context != 'property'"
            ):
                for cpath in by_value.get(r["value"], []):
                    kind = GROUP_KIND[cpath.split(".")[0]]
                    role = call_role.get((r["file_id"], r["line"], r["value"]), "literal")
                    self._add(
                        kind, cpath, r["value"], r["file_id"], r["line"], role, r["symbol_id"]
                    )

    # --------------------------------------------------------------- MCP tools
    def mcp_tools(self) -> None:
        rows = self.conn.execute(
            "SELECT s.file_id, s.symbol_id, s.value, s.line, f.path FROM strings s JOIN files f"
            " ON f.id = s.file_id WHERE s.context = 'property' AND s.detail = 'intent'"
            " AND f.path LIKE '%services/mcp/%'"
        ).fetchall()
        for r in rows:
            title = self.conn.execute(
                "SELECT value FROM strings WHERE file_id = ? AND detail = 'title' AND line"
                " BETWEEN ? AND ? ORDER BY line LIMIT 1",
                (r["file_id"], r["line"], r["line"] + 6),
            ).fetchone()
            self._add(
                "mcp_tool",
                r["value"],
                r["value"],
                r["file_id"],
                r["line"],
                "definition",
                r["symbol_id"],
                title["value"] if title else None,
            )

    # --------------------------------------------------------------- services
    def services(self) -> None:
        for path, file_id in self.file_ids.items():
            m = re.match(r"^server/services/([^/]+)/index\.js$", path)
            if not m:
                continue
            row = self.conn.execute(
                "SELECT s.id, s.start_line FROM exports e JOIN symbols s ON s.file_id = e.file_id"
                " AND s.name = e.local_name WHERE e.file_id = ? AND e.exported_name = 'default'"
                " LIMIT 1",
                (file_id,),
            ).fetchone()
            line = row["start_line"] if row else 1
            self._add(
                "service",
                m.group(1),
                m.group(1),
                file_id,
                line,
                "definition",
                row["id"] if row else None,
                f"server/services/{m.group(1)}/",
            )

    # --------------------------------------------------------------- routes / deeplinks
    def routes(self) -> None:
        api_defs: dict[str, tuple[str, str, re.Pattern]] = {}
        file_paths = {v: k for k, v in self.file_ids.items()}
        for r in self.conn.execute(
            "SELECT file_id, symbol_id, value, line FROM strings WHERE context = 'key'"
        ):
            m = API_KEY_RE.match(r["value"].strip())
            if not m:
                continue
            method, path = m.group(1).upper(), m.group(2)
            name = f"{method} {path}"
            # Front-end demo fixtures declare the same keys; they are mocks, not route handlers.
            role = "demo_mock" if file_paths[r["file_id"]].startswith("front/") else "definition"
            self._add("api_route", name, path, r["file_id"], r["line"], role, r["symbol_id"])
            if role == "definition":
                api_defs[name] = (method, path, _route_regex(path))
        front_defs: dict[str, re.Pattern] = {}
        rows = self.conn.execute(
            "SELECT s.file_id, s.symbol_id, s.value, s.line, f.path FROM strings s JOIN files f"
            " ON f.id = s.file_id WHERE s.context = 'jsx_attribute' AND s.detail = 'path'"
            " AND s.value LIKE '/%'"
        ).fetchall()
        for r in rows:
            component = None
            try:
                text = (self.repo_root / r["path"]).read_text(errors="replace").splitlines()
                m = re.search(r"<(\w+)[^<>]*path=", text[r["line"] - 1])
                component = m.group(1) if m else None
            except (OSError, IndexError):
                pass
            self._add(
                "front_route",
                r["value"],
                r["value"],
                r["file_id"],
                r["line"],
                "definition",
                r["symbol_id"],
                component,
            )
            front_defs[r["value"]] = _route_regex(r["value"])

        def specificity(pattern: str) -> tuple[int, int]:
            segs = [p for p in pattern.split("/") if p]
            return (sum(1 for p in segs if not p.startswith(":")), -len(segs))

        # Usages: any other string literal whose path matches a declared route pattern; the most
        # specific pattern (most static segments) wins.
        seen: set[tuple[str, int, int]] = set()
        for r in self.conn.execute(
            "SELECT file_id, symbol_id, value, line, context, detail FROM strings"
            " WHERE (value LIKE '/%' OR value LIKE '%/api/%') AND context != 'key'"
            " AND NOT (context = 'jsx_attribute' AND detail = 'path')"
        ):
            used = _usage_path(r["value"].strip())
            if "/api/" in used:
                api_path = used[used.index("/api/") :]
                method = (r["detail"] or "").rsplit(".", 1)[-1].upper()
                hits = [
                    (name, path)
                    for name, (m, path, rx) in api_defs.items()
                    if rx.match(api_path) and (method not in HTTP_METHODS or m == method)
                ]
                if hits:
                    name = max(hits, key=lambda h: specificity(h[1]))[0]
                    key = (name, r["file_id"], r["line"])
                    if key not in seen:
                        seen.add(key)
                        self._add(
                            "api_route",
                            name,
                            r["value"],
                            r["file_id"],
                            r["line"],
                            "usage",
                            r["symbol_id"],
                            r["detail"],
                        )
                continue
            hits = [p for p, rx in front_defs.items() if rx.match(used)]
            if hits:
                pattern = max(hits, key=specificity)
                self._add(
                    "front_route",
                    pattern,
                    r["value"],
                    r["file_id"],
                    r["line"],
                    "usage",
                    r["symbol_id"],
                    r["detail"],
                )

    # --------------------------------------------------------------- URIs
    def uris(self) -> None:
        for r in self.conn.execute(
            "SELECT file_id, symbol_id, value, line FROM strings WHERE value LIKE '%://%'"
        ):
            value = r["value"].strip()
            if URI_RE.match(value):
                scheme_host = re.match(r"^([a-z0-9+.-]+://[^/\s?#]*)", value, re.I)
                name = scheme_host.group(1) if scheme_host else value[:60]
                self._add("uri", name, value, r["file_id"], r["line"], "usage", r["symbol_id"])

    def run(self) -> dict:
        self.conn.execute("DELETE FROM catalog")
        self.constants()
        self.mcp_tools()
        self.services()
        self.routes()
        self.uris()
        self.conn.executemany(
            "INSERT INTO catalog(kind, name, value, file_id, line, role, symbol_id, detail)"
            " VALUES (?,?,?,?,?,?,?,?)",
            self.rows,
        )
        stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in self.rows:
            stats[row[0]][row[5]] += 1
        out = {k: dict(v) for k, v in sorted(stats.items())}
        logger.info("catalog: %s", out)
        return out


def build_catalog(conn: sqlite3.Connection, repo_root: Path) -> dict:
    return CatalogBuilder(conn, repo_root).run()
