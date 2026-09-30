"""Function-level chunks with a context header (D-004).

- One chunk per function/arrow symbol whose parent is the module or a class, and one per method
  anywhere (object-literal methods inside factories, e.g. front-end `actions`); other nested
  helpers stay inside their parent's chunk.
- Functions longer than MAX_CHUNK_LINES are split into overlapping windows (same symbol_id).
- Lines not covered by any function chunk (module-level constants, route tables, class fields)
  become "module" chunks attributed to the innermost enclosing class/module symbol, so that
  e.g. `constants.js` and `app.jsx` route declarations are retrievable too.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

CHUNK_KINDS = frozenset({"function", "arrow", "method"})
MIN_MODULE_RUN_LINES = 3  # non-blank, non-import lines needed to keep a module-level run


def _windows(start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if end - start + 1 <= size:
        return [(start, end)]
    out, s = [], start
    while s <= end:
        e = min(s + size - 1, end)
        out.append((s, e))
        if e == end:
            break
        s = e - overlap + 1
    return out


def _is_trivial(line: str) -> bool:
    t = line.strip()
    return (
        not t
        or t.startswith(("import ", "//", "/*", "*", "*/"))
        or t in {"}", "};", ")", ");", "]", "];", "{"}
        or ("require(" in t and t.startswith(("const ", "let ", "var ")))
    )


def header_for(path: str, qualname: str, kind: str, start: int, end: int, doc: str | None) -> str:
    head = f"// file: {path}\n// symbol: {qualname} ({kind}) lines {start}-{end}\n"
    if doc:
        head += f"// doc: {doc[:300]}\n"
    return head


def build_chunks(conn: sqlite3.Connection, repo_root: Path) -> dict:
    conn.execute("DELETE FROM chunks")
    files = conn.execute("SELECT id, path FROM files ORDER BY id").fetchall()
    size, overlap = config.MAX_CHUNK_LINES, config.CHUNK_WINDOW_OVERLAP
    rows: list[tuple] = []
    n_func = n_window = n_module = 0
    for f in files:
        try:
            lines = (
                (repo_root / f["path"]).read_text(encoding="utf-8", errors="replace").split("\n")
            )
        except OSError as exc:
            logger.warning("chunker: cannot read %s (%s)", f["path"], exc)
            continue
        syms = conn.execute(
            "SELECT s.id, s.name, s.qualname, s.kind, s.start_line, s.end_line, s.doc,"
            " p.kind AS parent_kind FROM symbols s LEFT JOIN symbols p ON p.id = s.parent_id"
            " WHERE s.file_id = ? ORDER BY s.start_line",
            (f["id"],),
        ).fetchall()
        covered = [False] * (len(lines) + 2)
        for s in syms:
            if s["kind"] not in CHUNK_KINDS:
                continue
            if s["parent_kind"] not in {"module", "class"} and s["kind"] != "method":
                continue
            spans = _windows(s["start_line"], s["end_line"], size, overlap)
            for a, b in spans:
                body = "\n".join(lines[a - 1 : b])
                head = header_for(f["path"], s["qualname"], s["kind"], a, b, s["doc"])
                rows.append((s["id"], f["id"], a, b, head, body))
            n_func += 1
            n_window += len(spans) - 1
            for i in range(s["start_line"], min(s["end_line"], len(lines)) + 1):
                covered[i] = True
        # Module-level / class-level remainder.
        containers = [s for s in syms if s["kind"] in {"module", "class"}]
        run: list[int] = []

        def flush(
            run: list[int], lines: list[str] = lines, containers: list = containers, f=f
        ) -> None:
            nonlocal n_module
            meaningful = [i for i in run if not _is_trivial(lines[i - 1])]
            if len(meaningful) < MIN_MODULE_RUN_LINES:
                return
            a0, b0 = meaningful[0], meaningful[-1]
            for a, b in _windows(a0, b0, size, overlap):
                owner = min(
                    (c for c in containers if c["start_line"] <= a and c["end_line"] >= b),
                    key=lambda c: c["end_line"] - c["start_line"],
                    default=containers[0] if containers else None,
                )
                if owner is None:
                    continue
                body = "\n".join(lines[a - 1 : b])
                qual = owner["qualname"] if owner["kind"] == "class" else "<module>"
                head = header_for(f["path"], qual, owner["kind"], a, b, owner["doc"])
                rows.append((owner["id"], f["id"], a, b, head, body))
                n_module += 1

        for i in range(1, len(lines) + 1):
            if covered[i]:
                if run:
                    flush(run)
                    run = []
            else:
                run.append(i)
        if run:
            flush(run)
    conn.executemany(
        "INSERT INTO chunks(symbol_id, file_id, start_line, end_line, header, text)"
        " VALUES (?,?,?,?,?,?)",
        rows,
    )
    stats = {
        "chunks": len(rows),
        "function_chunks": n_func,
        "extra_windows": n_window,
        "module_chunks": n_module,
    }
    logger.info("chunker: %s", stats)
    return stats
