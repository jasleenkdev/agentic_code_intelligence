"""The eight read-only agent tools (docs/approach.md §A.2). Each returns structured dicts, never
prose.

A module-level `ToolContext` holds the SQLite connection and retriever so tools can be called by
name from the answer pipeline; tests swap it with `set_context`.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src import config
from src.catalog.search import search_catalog
from src.config import RankingConfig
from src.graph import queries
from src.indexer import store
from src.indexer.store import normalize_text

logger = logging.getLogger(__name__)

MAX_READ_LINES = 300
PREVIEW_LINES = 8


@dataclass
class ToolContext:
    conn: sqlite3.Connection
    repo_root: Path
    ranking: RankingConfig = field(default_factory=lambda: config.RANKING)
    _retriever: object | None = None

    @property
    def retriever(self):
        if self._retriever is None:
            from src.retrieval.search import Retriever

            self._retriever = Retriever(self.conn)
        return self._retriever


_CTX: ToolContext | None = None


def get_context() -> ToolContext:
    global _CTX
    if _CTX is None:
        _CTX = ToolContext(store.connect(config.DB_PATH, readonly=True), config.REPO_ROOT)
    return _CTX


def set_context(ctx: ToolContext | None) -> None:
    global _CTX
    _CTX = ctx


def _preview(ctx: ToolContext, chunk_id: int) -> str:
    _, text = ctx.retriever.chunk_text(chunk_id)
    lines = text.split("\n")
    out = "\n".join(lines[:PREVIEW_LINES])
    return out + ("\n…" if len(lines) > PREVIEW_LINES else "")


def _hits_to_records(ctx: ToolContext, hits: list) -> list[dict]:
    return [
        {
            "file_path": h.file_path,
            "start_line": h.start_line,
            "end_line": h.end_line,
            "symbol": h.symbol,
            "kind": h.kind,
            "score": h.score,
            "ranks": h.ranks,
            "preview": _preview(ctx, h.chunk_id),
        }
        for h in hits
    ]


# ------------------------------------------------------------------------------------- tools


def semantic_search(query: str, k: int = 10, path_prefix: str | None = None) -> dict:
    """Hybrid retrieval (BM25 + embeddings, RRF, per the active RankingConfig)."""
    ctx = get_context()
    hits = ctx.retriever.search(query, k, ctx.ranking, path_prefix)
    return {"query": query, "results": _hits_to_records(ctx, hits)}


def keyword_search(query: str, k: int = 10, path_prefix: str | None = None) -> dict:
    """BM25 only, identifier-aware tokenisation."""
    ctx = get_context()
    cfg = RankingConfig(use_vectors=False, use_graph_boost=False, use_rerank=False)
    hits = ctx.retriever.search(query, k, cfg, path_prefix)
    return {"query": query, "results": _hits_to_records(ctx, hits)}


def find_symbol(name: str, kind: str | None = None) -> dict:
    ctx = get_context()
    return {"name": name, "results": queries.find_symbols(ctx.conn, name, kind)[:50]}


def get_callers(symbol: str, depth: int = 1) -> dict:
    ctx = get_context()
    out = queries.get_callers(ctx.conn, symbol, depth)
    out["callers"] = out["callers"][:100]
    return out


def get_callees(symbol: str, depth: int = 1) -> dict:
    ctx = get_context()
    out = queries.get_callees(ctx.conn, symbol, depth)
    out["callees"] = out["callees"][:150]
    return out


def find_call_sequence(first: str, second: str, same_function: bool = True) -> dict:
    ctx = get_context()
    return queries.find_call_sequence(ctx.conn, first, second, same_function)


def find_string_usages(text: str, exact: bool = False) -> dict:
    """String literals, constant references and domain-catalog entries matching `text`."""
    ctx = get_context()
    conn = ctx.conn
    text = text.strip().strip("'\"`")
    if not text:
        return {"text": text, "files": {}, "total": 0, "catalog": []}
    usages: dict[tuple[str, int], dict] = {}

    def add(path: str, line: int, rec: dict) -> None:
        """Keep the best-ranked record per file:line (lower rank = more relevant)."""
        key = (path, line)
        if key not in usages or rec["rank"] < usages[key]["rank"]:
            usages[key] = {"file_path": path, "line": line, **rec}

    def catalog_rank(role: str) -> int:
        return 0 if role == "definition" else 2

    if exact:
        rows = conn.execute(
            "SELECT s.value, s.line, s.context, s.detail, f.path, sy.qualname FROM strings s"
            " JOIN files f ON f.id = s.file_id LEFT JOIN symbols sy ON sy.id = s.symbol_id"
            " WHERE s.value = ? LIMIT 500",
            (text,),
        )
    else:
        norm = normalize_text(text)
        rows = conn.execute(
            "SELECT s.value, s.line, s.context, s.detail, f.path, sy.qualname FROM strings s"
            " JOIN files f ON f.id = s.file_id LEFT JOIN symbols sy ON sy.id = s.symbol_id"
            " WHERE s.value LIKE ? OR (? != '' AND s.normalized LIKE ?) LIMIT 500",
            (f"%{text}%", norm, f"%{norm}%"),
        )
    for r in rows:
        add(
            r["path"],
            r["line"],
            {
                "kind": "literal",
                "rank": 1 if r["value"] == text else 3,
                "value": r["value"][:160],
                "context": r["context"],
                "detail": r["detail"],
                "symbol": r["qualname"],
            },
        )
    const_rows = conn.execute(
        "SELECT c.path AS cpath, c.line, f.path, sy.qualname FROM const_refs c JOIN files f ON"
        " f.id = c.file_id LEFT JOIN symbols sy ON sy.id = c.symbol_id WHERE "
        + ("c.path = ?" if exact else "c.path LIKE ?")
        + " LIMIT 500",
        (text if exact else f"%{text}%",),
    )
    for r in const_rows:
        add(
            r["path"],
            r["line"],
            {
                "kind": "constant",
                "rank": 1 if r["cpath"] == text else 3,
                "value": r["cpath"],
                "context": "reference",
                "detail": None,
                "symbol": r["qualname"],
            },
        )
    catalog_hits = []
    if exact:
        for row in conn.execute(
            "SELECT c.kind, c.name, c.value, c.role, c.line, c.detail, f.path FROM catalog c"
            " JOIN files f ON f.id = c.file_id WHERE c.name = ? OR c.value = ? LIMIT 500",
            (text, text),
        ):
            add(
                row["path"],
                row["line"],
                {
                    "kind": f"{row['kind']}:{row['role']}",
                    "rank": catalog_rank(row["role"]),
                    "value": row["value"][:160],
                    "context": row["role"],
                    "detail": row["detail"],
                    "symbol": None,
                },
            )
    else:
        hits = [h for h in search_catalog(conn, text, limit=3) if h.score >= 1.0]
        # Only the best entry (and exact ties): mixing near-misses such as GET/PATCH variants of
        # a DELETE route buries the real answer.
        hits = [h for h in hits if h.score >= hits[0].score - 1e-6] if hits else []
        for hit in hits:
            catalog_hits.append(
                {"kind": hit.kind, "name": hit.name, "score": hit.score, "detail": hit.detail}
            )
            for row in hit.definitions + hit.usages:
                add(
                    row["file_path"],
                    row["line"],
                    {
                        "kind": f"{hit.kind}:{row['role']}",
                        "rank": catalog_rank(row["role"]),
                        "value": row["value"][:160],
                        "context": row["role"],
                        "detail": row["detail"],
                        "symbol": None,
                    },
                )
    files: dict[str, list[dict]] = defaultdict(list)
    for u in sorted(usages.values(), key=lambda u: (u["rank"], u["file_path"], u["line"])):
        files[u["file_path"]].append(u)
    return {
        "text": text,
        "exact": exact,
        "catalog": catalog_hits,
        "files": dict(files),
        "total": len(usages),
    }


def read_file_range(file_path: str, start_line: int, end_line: int) -> dict:
    """Exact source for a line range (1-based, inclusive), confined to the target repo."""
    ctx = get_context()
    root = ctx.repo_root.resolve()
    target = (root / file_path.lstrip("/")).resolve()
    if root not in target.parents or not target.is_file():
        return {"error": f"file not found in repo: {file_path}"}
    lines = target.read_text(encoding="utf-8", errors="replace").split("\n")
    total = len(lines)
    start = max(1, int(start_line))
    end = min(total, int(end_line), start + MAX_READ_LINES - 1)
    if start > total or end < start:
        return {"error": f"range {start_line}-{end_line} outside file ({total} lines)"}
    return {
        "file_path": target.relative_to(root).as_posix(),
        "start_line": start,
        "end_line": end,
        "total_lines": total,
        "text": "\n".join(lines[start - 1 : end]),
    }


TOOLS = {
    "semantic_search": semantic_search,
    "keyword_search": keyword_search,
    "find_symbol": find_symbol,
    "get_callers": get_callers,
    "get_callees": get_callees,
    "find_call_sequence": find_call_sequence,
    "find_string_usages": find_string_usages,
    "read_file_range": read_file_range,
}
