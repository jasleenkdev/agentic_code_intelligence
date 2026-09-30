"""Best-effort static call resolution → call-graph edges.

Resolution labels (stored on each call row):
- ``static``     — resolved through same-file definitions or import/export chains.
- ``heuristic``  — resolved by member name + owner/directory matching (dynamic dispatch guess).
- ``external``   — a builtin, a global, or a method on a well-known external receiver.
- ``unresolved`` — no confident target (ambiguous or dynamic); the call site is still kept.
"""

from __future__ import annotations

import logging
import posixpath
import re
import sqlite3
from collections import defaultdict

logger = logging.getLogger(__name__)

JS_GLOBALS = frozenset(
    """require setTimeout setInterval clearTimeout clearInterval setImmediate parseInt parseFloat
    isNaN isFinite encodeURIComponent decodeURIComponent encodeURI decodeURI String Number Boolean
    Array Object Promise Date Error TypeError RegExp Symbol Map Set WeakMap JSON Math fetch
    Buffer URL URLSearchParams structuredClone queueMicrotask describe it expect super import
    Proxy Reflect BigInt eval alert confirm""".split()
)
EXTERNAL_RECEIVERS = frozenset(
    """console logger JSON Math Object Array Promise Number String Date Buffer process path fs
    window document localStorage sessionStorage navigator crypto os util axios dayjs moment
    Sequelize Op db res req route router app express Joi cron schedule http https
    child_process zlib stream events EventEmitter Reflect Symbol URL""".split()
)
BUILTIN_METHODS = frozenset(
    """push pop shift unshift map filter forEach reduce reduceRight flatMap flat then catch finally
    join split replace replaceAll toString includes indexOf lastIndexOf slice splice keys values
    entries find findIndex some every sort concat trim trimStart trimEnd toLowerCase toUpperCase
    startsWith endsWith padStart padEnd log info warn error debug trace stringify parse resolve
    reject all allSettled race assign freeze hasOwnProperty apply call bind toFixed charAt
    charCodeAt substring substr match matchAll test exec getTime toISOString valueOf fill has
    add delete clear preventDefault stopPropagation setState forceUpdate isArray from""".split()
)
_EXTS = ("", ".js", ".jsx", ".mjs", ".cjs", "/index.js", "/index.jsx", "/index.mjs")
_NORM = re.compile(r"[^a-z0-9]")


def _norm(name: str) -> str:
    return _NORM.sub("", name.lower())


def _owner_match_level(receiver: str | None, owner: str) -> int:
    """2 = exact owner match, 1 = substring match, 0 = no match."""
    if not receiver:
        return 0
    last = receiver.split(".")[-1]
    r, o = _norm(last), _norm(owner.split(".")[-1])
    if not r or not o or r in {"this", "self", "that"}:
        return 0
    o_stripped = re.sub(r"(manager|service|handler|controller|client)$", "", o)
    if r == o or r == o_stripped:
        return 2
    if len(r) >= 4 and (r in o or (o_stripped and o_stripped in r)):
        return 1
    return 0


class Resolver:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.path_by_id: dict[int, str] = {}
        self.id_by_path: dict[str, int] = {}
        self.syms_by_file_name: dict[tuple[int, str], list[int]] = defaultdict(list)
        self.sym_file: dict[int, int] = {}
        self.sym_kind: dict[int, str] = {}
        self.sym_parent_kind: dict[int, str] = {}
        self.global_funcs: dict[str, list[int]] = defaultdict(list)
        self.exports: dict[tuple[int, str], str] = {}
        self.imports: dict[tuple[int, str], tuple[int | None, str]] = {}
        self.member_index: dict[str, list[tuple[int, int, str]]] = defaultdict(list)

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        c = self.conn
        for row in c.execute("SELECT id, path FROM files"):
            self.path_by_id[row["id"]] = row["path"]
            self.id_by_path[row["path"]] = row["id"]
        kinds = {r["id"]: r["kind"] for r in c.execute("SELECT id, kind FROM symbols")}
        for row in c.execute("SELECT id, file_id, name, kind, parent_id, qualname FROM symbols"):
            sid = row["id"]
            self.sym_file[sid] = row["file_id"]
            self.sym_kind[sid] = row["kind"]
            parent_kind = kinds.get(row["parent_id"], "")
            self.sym_parent_kind[sid] = parent_kind
            if row["kind"] == "module":
                continue
            key = (row["file_id"], row["name"])
            # Top-level definitions first so they win over nested helpers of the same name.
            if parent_kind == "module":
                self.syms_by_file_name[key].insert(0, sid)
            else:
                self.syms_by_file_name[key].append(sid)
            if row["kind"] in {"function", "arrow", "class"} and parent_kind == "module":
                self.global_funcs[row["name"]].append(sid)
            if row["kind"] == "method":
                owner = row["qualname"].rsplit(".", 1)[0] if "." in row["qualname"] else ""
                self.member_index[row["name"]].append((sid, row["file_id"], owner))
        for row in c.execute("SELECT file_id, exported_name, local_name FROM exports"):
            self.exports[(row["file_id"], row["exported_name"])] = row["local_name"]

    def _resolve_import_paths(self) -> int:
        resolved = 0
        rows = self.conn.execute("SELECT id, file_id, source FROM imports").fetchall()
        updates = []
        for row in rows:
            source = row["source"]
            target = None
            if source.startswith("."):
                base = posixpath.dirname(self.path_by_id[row["file_id"]])
                joined = posixpath.normpath(posixpath.join(base, source))
                for ext in _EXTS:
                    if joined + ext in self.id_by_path:
                        target = self.id_by_path[joined + ext]
                        break
            if target is not None:
                resolved += 1
            updates.append((target, row["id"]))
        self.conn.executemany("UPDATE imports SET resolved_file_id = ? WHERE id = ?", updates)
        for row in self.conn.execute(
            "SELECT file_id, local_name, imported_name, resolved_file_id FROM imports"
        ):
            self.imports[(row["file_id"], row["local_name"])] = (
                row["resolved_file_id"],
                row["imported_name"],
            )
        return resolved

    # ------------------------------------------------------------------ resolution
    def resolve_export(self, file_id: int, name: str, depth: int = 0) -> int | tuple | None:
        """Symbol id exported as `name` from file, or ('module', file_id) for object exports."""
        local = self.exports.get((file_id, name))
        if local is None:
            if name == "default":
                return ("module", file_id)
            return None
        syms = self.syms_by_file_name.get((file_id, local))
        if syms:
            return syms[0]
        if depth < 2 and (file_id, local) in self.imports:
            return self.resolve_local(file_id, local, depth + 1)
        return None

    def resolve_local(self, file_id: int, name: str, depth: int = 0) -> int | tuple | None:
        """Resolve an identifier as seen from `file_id` (same-file definition, then import)."""
        syms = self.syms_by_file_name.get((file_id, name))
        if syms:
            return syms[0]
        imp = self.imports.get((file_id, name))
        if imp is None:
            return None
        target_file, imported = imp
        if target_file is None:
            return None
        if imported == "*":
            return ("module", target_file)
        return self.resolve_export(target_file, imported, depth)

    def _index_bindings(self) -> None:
        rows = self.conn.execute(
            "SELECT id, file_id, owner, member, target FROM bindings"
        ).fetchall()
        updates = []
        for row in rows:
            target = self.resolve_local(row["file_id"], row["target"])
            if isinstance(target, int) and self.sym_kind.get(target) in {
                "function",
                "arrow",
                "method",
            }:
                updates.append((target, row["id"]))
                owner = row["owner"]
                if owner in {"this", "<object>"}:
                    owner = ""
                self.member_index[row["member"]].append((target, row["file_id"], owner))
        self.conn.executemany("UPDATE bindings SET target_symbol_id = ? WHERE id = ?", updates)

    def _pick_member(
        self, name: str, receiver: str | None, caller_file: int
    ) -> tuple[int | None, str]:
        cands = self.member_index.get(name)
        if not cands:
            return None, "unresolved"
        uniq: dict[int, tuple[int, int, str]] = {}
        for c in cands:
            uniq.setdefault(c[0], c)
        cands = list(uniq.values())
        caller_dir = posixpath.dirname(self.path_by_id[caller_file])
        if receiver in {"this", "self", "that"}:
            same = [c for c in cands if c[1] == caller_file]
            if len(same) == 1:
                return same[0][0], "heuristic"
            near = [c for c in cands if posixpath.dirname(self.path_by_id[c[1]]) == caller_dir]
            if len(near) == 1:
                return near[0][0], "heuristic"
            # Method bound in the parent directory's index.js (Gladys `lib/x/index.js` pattern).
            parent_dir = posixpath.dirname(caller_dir)
            up = [c for c in cands if posixpath.dirname(self.path_by_id[c[1]]) == parent_dir]
            if len(up) == 1:
                return up[0][0], "heuristic"
        for level in (2, 1):
            owner_hits = [c for c in cands if c[2] and _owner_match_level(receiver, c[2]) == level]
            if len(owner_hits) == 1:
                return owner_hits[0][0], "heuristic"
            if owner_hits:
                break
        if len(cands) == 1 and receiver not in EXTERNAL_RECEIVERS:
            return cands[0][0], "heuristic"
        return None, "unresolved"

    def resolve_call(self, row: sqlite3.Row) -> tuple[int | None, str]:
        name = row["callee_name"]
        text = row["callee_text"]
        receiver = row["receiver"]
        file_id = row["file_id"]
        if receiver is None and text == name:  # bare identifier call
            if name in JS_GLOBALS:
                return None, "external"
            target = self.resolve_local(file_id, name)
            if isinstance(target, int):
                return target, "static"
            if isinstance(target, tuple):
                return None, "unresolved"
            if (file_id, name) in self.imports:
                return None, "external"  # imported from an npm package
            funcs = self.global_funcs.get(name, [])
            if len(funcs) == 1:
                return funcs[0], "heuristic"
            return None, "unresolved"
        if receiver is None:
            return None, "unresolved"
        root = receiver.split(".")[0].split("(")[0].split("[")[0]
        if root in EXTERNAL_RECEIVERS and root not in {"this"}:
            return None, "external"
        if receiver.isidentifier():
            target = self.resolve_local(file_id, receiver)
            if isinstance(target, tuple):
                exported = self.resolve_export(target[1], name)
                if isinstance(exported, int):
                    return exported, "static"
            elif isinstance(target, int) and self.sym_kind.get(target) in {"class", "function"}:
                # `Foo.bar()` on a class/constructor defined or imported here.
                tfile = self.sym_file[target]
                same = [c for c in self.member_index.get(name, []) if c[1] == tfile]
                if len(same) == 1:
                    return same[0][0], "heuristic"
            if (file_id, receiver) in self.imports and self.imports[(file_id, receiver)][0] is None:
                return None, "external"
        if name in BUILTIN_METHODS:
            return None, "external"
        return self._pick_member(name, receiver, file_id)

    def run(self) -> dict:
        self._load()
        imports_resolved = self._resolve_import_paths()
        self._index_bindings()
        counts: dict[str, int] = defaultdict(int)
        updates = []
        edges = []
        rows = self.conn.execute(
            "SELECT id, file_id, caller_id, callee_text, callee_name, receiver FROM calls"
        ).fetchall()
        for row in rows:
            target, resolution = self.resolve_call(row)
            counts[resolution] += 1
            updates.append((target, resolution, row["id"]))
            if target is not None:
                edges.append((row["caller_id"], target, row["id"], resolution))
        self.conn.executemany(
            "UPDATE calls SET callee_symbol_id = ?, resolution = ? WHERE id = ?", updates
        )
        self.conn.execute("DELETE FROM edges")
        self.conn.executemany(
            "INSERT INTO edges(from_id, to_id, call_id, confidence) VALUES (?,?,?,?)", edges
        )
        stats = {
            "imports_resolved": imports_resolved,
            "calls": len(rows),
            "edges": len(edges),
            **{f"calls_{k}": v for k, v in counts.items()},
        }
        logger.info("resolver: %s", stats)
        return stats


def resolve_calls(conn: sqlite3.Connection) -> dict:
    return Resolver(conn).run()
