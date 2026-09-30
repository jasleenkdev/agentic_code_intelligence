"""SQLite storage for the structural index (D-010). One file: index/code.db."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

from src.indexer.extractor import FileFacts

logger = logging.getLogger(__name__)

SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY, path TEXT UNIQUE, lines INTEGER, module_system TEXT,
    parse_errors INTEGER, sha1 TEXT
);
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY, file_id INTEGER, name TEXT, qualname TEXT, kind TEXT,
    start_line INTEGER, end_line INTEGER, parent_id INTEGER, params TEXT,
    is_async INTEGER, is_exported INTEGER, doc TEXT
);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY, file_id INTEGER, caller_id INTEGER, callee_text TEXT,
    callee_name TEXT, receiver TEXT, line INTEGER, end_line INTEGER, ordinal INTEGER,
    awaited INTEGER, in_branch INTEGER, in_callback INTEGER, in_loop INTEGER, is_new INTEGER,
    args_text TEXT, first_arg TEXT, callee_symbol_id INTEGER, resolution TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    from_id INTEGER, to_id INTEGER, call_id INTEGER, confidence TEXT
);
CREATE TABLE IF NOT EXISTS strings (
    id INTEGER PRIMARY KEY, file_id INTEGER, symbol_id INTEGER, value TEXT, normalized TEXT,
    line INTEGER, context TEXT, detail TEXT, is_template INTEGER
);
CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY, file_id INTEGER, local_name TEXT, imported_name TEXT, source TEXT,
    resolved_file_id INTEGER, line INTEGER
);
CREATE TABLE IF NOT EXISTS exports (
    id INTEGER PRIMARY KEY, file_id INTEGER, exported_name TEXT, local_name TEXT, line INTEGER
);
CREATE TABLE IF NOT EXISTS const_refs (
    id INTEGER PRIMARY KEY, file_id INTEGER, symbol_id INTEGER, path TEXT, line INTEGER
);
CREATE TABLE IF NOT EXISTS bindings (
    id INTEGER PRIMARY KEY, file_id INTEGER, owner TEXT, member TEXT, target TEXT, line INTEGER,
    target_symbol_id INTEGER
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, symbol_id INTEGER, file_id INTEGER, start_line INTEGER,
    end_line INTEGER, header TEXT, text TEXT
);
CREATE TABLE IF NOT EXISTS catalog (
    id INTEGER PRIMARY KEY, kind TEXT, name TEXT, value TEXT, file_id INTEGER, line INTEGER,
    role TEXT, symbol_id INTEGER, detail TEXT
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS ix_sym_name ON symbols(name);
CREATE INDEX IF NOT EXISTS ix_sym_qual ON symbols(qualname);
CREATE INDEX IF NOT EXISTS ix_sym_file ON symbols(file_id);
CREATE INDEX IF NOT EXISTS ix_calls_caller ON calls(caller_id);
CREATE INDEX IF NOT EXISTS ix_calls_name ON calls(callee_name);
CREATE INDEX IF NOT EXISTS ix_calls_target ON calls(callee_symbol_id);
CREATE INDEX IF NOT EXISTS ix_edges_from ON edges(from_id);
CREATE INDEX IF NOT EXISTS ix_edges_to ON edges(to_id);
CREATE INDEX IF NOT EXISTS ix_str_norm ON strings(normalized);
CREATE INDEX IF NOT EXISTS ix_imports_file ON imports(file_id, local_name);
CREATE INDEX IF NOT EXISTS ix_exports_file ON exports(file_id, exported_name);
CREATE INDEX IF NOT EXISTS ix_const_path ON const_refs(path);
CREATE INDEX IF NOT EXISTS ix_bind_member ON bindings(member);
CREATE INDEX IF NOT EXISTS ix_chunks_symbol ON chunks(symbol_id);
CREATE INDEX IF NOT EXISTS ix_catalog_kind ON catalog(kind, name);
"""

_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(value: str) -> str:
    """Lowercase and collapse punctuation — the `normalized` form used for fuzzy usage search."""
    return _NORM_RE.sub(" ", value.lower()).strip()


def connect(db_path: Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        if not db_path.exists():
            raise FileNotFoundError(f"Index not found at {db_path}. Run the indexer first.")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def create_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(INDEXES)


def insert_file(
    conn: sqlite3.Connection, rel_path: str, facts: FileFacts, parse_errors: int, sha1: str
) -> int:
    cur = conn.execute(
        "INSERT INTO files(path, lines, module_system, parse_errors, sha1) VALUES (?,?,?,?,?)",
        (rel_path, facts.line_count, facts.module_system, parse_errors, sha1),
    )
    file_id = cur.lastrowid
    local_to_id: dict[int, int] = {}
    for sym in facts.symbols:
        parent = local_to_id.get(sym["parent_local"]) if sym["parent_local"] is not None else None
        c = conn.execute(
            "INSERT INTO symbols(file_id, name, qualname, kind, start_line, end_line, parent_id,"
            " params, is_async, is_exported, doc) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                file_id,
                sym["name"],
                sym["qualname"],
                sym["kind"],
                sym["start_line"],
                sym["end_line"],
                parent,
                json.dumps(sym["params"]),
                int(sym["is_async"]),
                int(sym["is_exported"]),
                sym["doc"],
            ),
        )
        local_to_id[sym["local_id"]] = c.lastrowid
    conn.executemany(
        "INSERT INTO calls(file_id, caller_id, callee_text, callee_name, receiver, line, end_line,"
        " ordinal, awaited, in_branch, in_callback, in_loop, is_new, args_text, first_arg,"
        " callee_symbol_id, resolution) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,'unresolved')",
        [
            (
                file_id,
                local_to_id[c["caller_local"]],
                c["callee_text"],
                c["callee_name"],
                c["receiver"],
                c["line"],
                c["end_line"],
                c["ordinal"],
                int(c["awaited"]),
                int(c["in_branch"]),
                int(c["in_callback"]),
                int(c["in_loop"]),
                int(c["is_new"]),
                c["args_text"],
                c["first_arg"],
            )
            for c in facts.calls
        ],
    )
    conn.executemany(
        "INSERT INTO strings(file_id, symbol_id, value, normalized, line, context, detail,"
        " is_template) VALUES (?,?,?,?,?,?,?,?)",
        [
            (
                file_id,
                local_to_id.get(s["symbol_local"]),
                s["value"],
                normalize_text(s["value"]),
                s["line"],
                s["context"],
                s["detail"],
                int(s["is_template"]),
            )
            for s in facts.strings
        ],
    )
    conn.executemany(
        "INSERT INTO imports(file_id, local_name, imported_name, source, line) VALUES (?,?,?,?,?)",
        [
            (file_id, i["local_name"], i["imported_name"], i["source"], i["line"])
            for i in facts.imports
        ],
    )
    conn.executemany(
        "INSERT INTO exports(file_id, exported_name, local_name, line) VALUES (?,?,?,?)",
        [(file_id, e["exported_name"], e["local_name"], e["line"]) for e in facts.exports],
    )
    conn.executemany(
        "INSERT INTO const_refs(file_id, symbol_id, path, line) VALUES (?,?,?,?)",
        [
            (file_id, local_to_id.get(r["symbol_local"]), r["path"], r["line"])
            for r in facts.const_refs
        ],
    )
    conn.executemany(
        "INSERT INTO bindings(file_id, owner, member, target, line) VALUES (?,?,?,?,?)",
        [(file_id, b["owner"], b["member"], b["target"], b["line"]) for b in facts.bindings],
    )
    return file_id


def set_meta(conn: sqlite3.Connection, key: str, value: object) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, json.dumps(value)))


def get_meta(conn: sqlite3.Connection, key: str, default: object = None) -> object:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default
