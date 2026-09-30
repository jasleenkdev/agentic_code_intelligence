"""Extract symbols, call sites, string literals, imports/exports and bindings from one JS file.

Uses tree-sitter (grammar pinned to tree-sitter-javascript 0.23.1, see D-006). The traversal is
iterative so deeply nested JSX cannot blow the Python stack. Everything returned is plain dicts with
1-based line numbers; `local_id`s are file-local and turned into database ids by the store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tree_sitter_javascript as tsjs
from tree_sitter import Language, Node, Parser

JS_LANGUAGE = Language(tsjs.language())

FUNCTION_TYPES = frozenset(
    {
        "function_declaration",
        "generator_function_declaration",
        "function_expression",
        "function",
        "generator_function",
        "arrow_function",
        "method_definition",
    }
)
CLASS_TYPES = frozenset({"class_declaration", "class"})
LOOP_TYPES = frozenset(
    {"for_statement", "for_in_statement", "while_statement", "do_statement", "for_of_statement"}
)
CONST_ROOT_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
WS_RE = re.compile(r"\s+")
MAX_STRING_LEN = 500
MAX_ARGS_TEXT = 200
LOOP_METHODS = frozenset({"forEach", "map", "filter", "reduce", "flatMap", "some", "every", "find"})


@dataclass
class FileFacts:
    """Everything extracted from a single file."""

    symbols: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    strings: list[dict] = field(default_factory=list)
    imports: list[dict] = field(default_factory=list)
    exports: list[dict] = field(default_factory=list)
    const_refs: list[dict] = field(default_factory=list)
    bindings: list[dict] = field(default_factory=list)
    module_system: str = "unknown"
    line_count: int = 0


@dataclass(frozen=True)
class _Ctx:
    symbol: int  # local id of the enclosing named symbol
    branch: bool = False  # inside a conditional arm
    callback: bool = False  # inside an anonymous function nested in the symbol
    loop: bool = False  # inside a loop body / iteration callback
    owner: str | None = None  # object/class name for qualifying pair/method names


def make_parser() -> Parser:
    return Parser(JS_LANGUAGE)


def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _flat(text: str, limit: int = 300) -> str:
    text = WS_RE.sub(" ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _string_value(node: Node, src: bytes) -> str:
    raw = _text(node, src)
    if node.type == "string" and len(raw) >= 2:
        return raw[1:-1]
    if node.type == "template_string" and len(raw) >= 2:
        return raw[1:-1]
    return raw


def _doc_comment(stmt: Node, src: bytes) -> str | None:
    prev = stmt.prev_named_sibling
    if prev is not None and prev.type == "comment" and stmt.start_point[0] - prev.end_point[0] <= 1:
        text = _text(prev, src)
        if text.startswith("/**") or text.startswith("//"):
            lines = [ln.strip().lstrip("/*").strip() for ln in text.splitlines()]
            cleaned = " ".join(ln for ln in lines if ln and not ln.startswith("@example"))
            return cleaned[:400] or None
    return None


def _params(fn: Node, src: bytes) -> list[str]:
    params = fn.child_by_field_name("parameters")
    if params is None:
        single = fn.child_by_field_name("parameter")
        return [_text(single, src)] if single is not None else []
    return [_flat(_text(p, src), 60) for p in params.named_children if p.type != "comment"]


def _statement_of(node: Node) -> Node:
    """The statement node whose preceding comment documents `node`."""
    cur = node
    while cur.parent is not None and cur.parent.type in {
        "variable_declarator",
        "lexical_declaration",
        "variable_declaration",
        "assignment_expression",
        "expression_statement",
        "export_statement",
        "pair",
    }:
        cur = cur.parent
    return cur


def _member_qualname(left_text: str) -> str:
    q = left_text.replace(".prototype.", ".").replace("module.exports.", "").replace("exports.", "")
    return _flat(q, 120)


def _function_name(fn: Node, src: bytes, ctx: _Ctx) -> tuple[str, str] | None:
    """Return (name, qualname) for a function node if it is a *named* symbol, else None."""
    parent = fn.parent
    own = fn.child_by_field_name("name")
    if fn.type == "method_definition" and own is not None:
        name = _text(own, src)
        return name, f"{ctx.owner}.{name}" if ctx.owner else name
    if fn.type in {"function_declaration", "generator_function_declaration"} and own is not None:
        name = _text(own, src)
        return name, name
    if parent is None:
        return None
    if parent.type == "variable_declarator":
        target = parent.child_by_field_name("name")
        if target is not None and target.type == "identifier":
            name = _text(target, src)
            return name, name
    if parent.type == "assignment_expression" and parent.child_by_field_name("right") == fn:
        left = parent.child_by_field_name("left")
        if left is not None:
            left_text = _text(left, src)
            if left.type == "member_expression":
                prop = left.child_by_field_name("property")
                prop_name = _text(prop, src) if prop is not None else left_text
                if left_text in {"module.exports", "exports"}:
                    if own is not None:
                        n = _text(own, src)
                        return n, n
                    return "module.exports", "module.exports"
                return prop_name, _member_qualname(left_text)
            if left.type == "identifier":
                return left_text, left_text
    if parent.type == "pair" and parent.child_by_field_name("value") == fn:
        key = parent.child_by_field_name("key")
        if key is not None:
            name = _text(key, src).strip("'\"")
            return name, f"{ctx.owner}.{name}" if ctx.owner else name
    if parent.type == "field_definition":
        prop = parent.child_by_field_name("property")
        if prop is not None:
            name = _text(prop, src)
            return name, f"{ctx.owner}.{name}" if ctx.owner else name
    if own is not None and fn.type in {"function_expression", "function"}:
        # Named function expression used as a value (e.g. `module.exports = function Foo()`).
        if parent.type not in {"arguments"}:
            name = _text(own, src)
            return name, name
    return None


def _owner_for_object(obj: Node, src: bytes) -> str | None:
    parent = obj.parent
    while parent is not None and parent.type in {"parenthesized_expression", "arguments"}:
        parent = parent.parent
    if parent is None:
        return None
    if parent.type == "variable_declarator":
        name = parent.child_by_field_name("name")
        return _text(name, src) if name is not None and name.type == "identifier" else None
    if parent.type == "assignment_expression":
        left = parent.child_by_field_name("left")
        return _member_qualname(_text(left, src)) if left is not None else None
    if parent.type == "pair":
        key = parent.child_by_field_name("key")
        return _text(key, src) if key is not None else None
    return None


def _string_context(node: Node, src: bytes) -> tuple[str, str | None]:
    parent = node.parent
    if parent is None:
        return "other", None
    ptype = parent.type
    if ptype == "arguments":
        call = parent.parent
        fn = call.child_by_field_name("function") if call is not None else None
        callee = _flat(_text(fn, src), 80) if fn is not None else None
        if callee in {"require"}:
            return "import", callee
        return "argument", callee
    if ptype in {"import_statement", "export_statement"}:
        return "import", None
    if ptype in {"variable_declarator", "assignment_expression"}:
        return "assignment", None
    if ptype == "pair":
        key = parent.child_by_field_name("key")
        if key is not None and key == node:
            return "key", None
        return "property", _text(key, src).strip("'\"") if key is not None else None
    if ptype == "return_statement":
        return "return", None
    if ptype in {"jsx_attribute", "jsx_expression"}:
        attr = parent if ptype == "jsx_attribute" else parent.parent
        if attr is not None and attr.type == "jsx_attribute" and attr.named_child_count:
            return "jsx_attribute", _text(attr.named_children[0], src)
        return "jsx_attribute", None
    if ptype == "array":
        return "array", None
    if ptype in {"binary_expression", "switch_case"}:
        return "comparison", None
    return "other", None


def _require_source(node: Node, src: bytes) -> str | None:
    """If `node` is `require('x')`, return 'x'."""
    if node.type != "call_expression":
        return None
    fn = node.child_by_field_name("function")
    if fn is None or fn.type != "identifier" or _text(fn, src) != "require":
        return None
    args = node.child_by_field_name("arguments")
    if args is None or not args.named_children:
        return None
    first = args.named_children[0]
    if first.type == "string":
        return _string_value(first, src)
    return None


class _Extractor:
    def __init__(self, src: bytes) -> None:
        self.src = src
        self.facts = FileFacts()

    # ---------------------------------------------------------------- symbols
    def _add_symbol(
        self,
        name: str,
        qualname: str,
        kind: str,
        node: Node,
        parent: int | None,
        params: list[str] | None = None,
        is_async: bool = False,
        doc: str | None = None,
    ) -> int:
        local_id = len(self.facts.symbols)
        self.facts.symbols.append(
            {
                "local_id": local_id,
                "name": name,
                "qualname": qualname,
                "kind": kind,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "parent_local": parent,
                "params": params or [],
                "is_async": is_async,
                "is_exported": False,
                "doc": doc,
            }
        )
        return local_id

    # ---------------------------------------------------------------- imports/exports
    def _handle_declarator(self, node: Node) -> None:
        src = self.src
        value = node.child_by_field_name("value")
        target = node.child_by_field_name("name")
        if value is None or target is None:
            return
        # `require('x').foo` or `require('x')`
        member = None
        call = value
        if value.type == "member_expression":
            obj = value.child_by_field_name("object")
            prop = value.child_by_field_name("property")
            if obj is not None and _require_source(obj, src) is not None:
                call, member = obj, _text(prop, src) if prop is not None else None
        if value.type == "await_expression" and value.named_children:
            call = value.named_children[0]
        source = _require_source(call, src)
        if source is None:
            return
        line = node.start_point[0] + 1
        self.facts.module_system = "cjs" if self.facts.module_system != "esm" else "mixed"
        if target.type == "identifier":
            self.facts.imports.append(
                {
                    "local_name": _text(target, src),
                    "imported_name": member or "default",
                    "source": source,
                    "line": line,
                }
            )
        elif target.type == "object_pattern":
            for prop in target.named_children:
                if prop.type == "shorthand_property_identifier_pattern":
                    n = _text(prop, src)
                    self.facts.imports.append(
                        {"local_name": n, "imported_name": n, "source": source, "line": line}
                    )
                elif prop.type == "pair_pattern":
                    key = prop.child_by_field_name("key")
                    val = prop.child_by_field_name("value")
                    if key is not None and val is not None and val.type == "identifier":
                        self.facts.imports.append(
                            {
                                "local_name": _text(val, src),
                                "imported_name": _text(key, src),
                                "source": source,
                                "line": line,
                            }
                        )

    def _handle_import_statement(self, node: Node) -> None:
        src = self.src
        source_node = node.child_by_field_name("source")
        if source_node is None:
            return
        source = _string_value(source_node, src)
        line = node.start_point[0] + 1
        self.facts.module_system = "esm" if self.facts.module_system != "cjs" else "mixed"
        clause = next((c for c in node.named_children if c.type == "import_clause"), None)
        if clause is None:
            return
        for part in clause.named_children:
            if part.type == "identifier":
                self.facts.imports.append(
                    {
                        "local_name": _text(part, src),
                        "imported_name": "default",
                        "source": source,
                        "line": line,
                    }
                )
            elif part.type == "namespace_import":
                ident = next((c for c in part.named_children if c.type == "identifier"), None)
                if ident is not None:
                    self.facts.imports.append(
                        {
                            "local_name": _text(ident, src),
                            "imported_name": "*",
                            "source": source,
                            "line": line,
                        }
                    )
            elif part.type == "named_imports":
                for spec in part.named_children:
                    if spec.type != "import_specifier":
                        continue
                    name = spec.child_by_field_name("name")
                    alias = spec.child_by_field_name("alias")
                    if name is None:
                        continue
                    self.facts.imports.append(
                        {
                            "local_name": _text(alias or name, src),
                            "imported_name": _text(name, src),
                            "source": source,
                            "line": line,
                        }
                    )

    def _handle_export_statement(self, node: Node) -> None:
        src = self.src
        line = node.start_point[0] + 1
        self.facts.module_system = "esm" if self.facts.module_system != "cjs" else "mixed"
        is_default = any(c.type == "default" for c in node.children)
        decl = node.child_by_field_name("declaration")
        value = node.child_by_field_name("value")
        if decl is not None:
            names: list[str] = []
            if decl.type in {"function_declaration", "class_declaration"} | {
                "generator_function_declaration"
            }:
                n = decl.child_by_field_name("name")
                if n is not None:
                    names.append(_text(n, src))
            elif decl.type in {"lexical_declaration", "variable_declaration"}:
                for d in decl.named_children:
                    n = d.child_by_field_name("name") if d.type == "variable_declarator" else None
                    if n is not None and n.type == "identifier":
                        names.append(_text(n, src))
            for n in names:
                self.facts.exports.append(
                    {"exported_name": "default" if is_default else n, "local_name": n, "line": line}
                )
        elif value is not None:
            local = _text(value, src) if value.type == "identifier" else None
            if local is None:
                own = value.child_by_field_name("name")
                local = _text(own, src) if own is not None else "default"
            self.facts.exports.append(
                {"exported_name": "default", "local_name": local, "line": line}
            )
        for clause in node.named_children:
            if clause.type != "export_clause":
                continue
            for spec in clause.named_children:
                name = spec.child_by_field_name("name")
                alias = spec.child_by_field_name("alias")
                if name is not None:
                    self.facts.exports.append(
                        {
                            "exported_name": _text(alias or name, src),
                            "local_name": _text(name, src),
                            "line": line,
                        }
                    )

    def _handle_assignment(self, node: Node, ctx: _Ctx) -> None:
        """CommonJS exports and prototype/member bindings (`X.prototype.m = m`)."""
        src = self.src
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is None or right is None:
            return
        left_text = _text(left, src)
        line = node.start_point[0] + 1
        if left_text in {"module.exports", "exports"}:
            self.facts.module_system = "cjs" if self.facts.module_system != "esm" else "mixed"
            if right.type == "object":
                for prop in right.named_children:
                    if prop.type == "shorthand_property_identifier":
                        n = _text(prop, src)
                        self.facts.exports.append(
                            {"exported_name": n, "local_name": n, "line": line}
                        )
                    elif prop.type == "pair":
                        key = prop.child_by_field_name("key")
                        val = prop.child_by_field_name("value")
                        if key is None or val is None:
                            continue
                        local = _text(val, src) if val.type == "identifier" else _text(key, src)
                        self.facts.exports.append(
                            {
                                "exported_name": _text(key, src).strip("'\""),
                                "local_name": local,
                                "line": line,
                            }
                        )
            else:
                local = None
                if right.type == "identifier":
                    local = _text(right, src)
                else:
                    own = right.child_by_field_name("name")
                    local = _text(own, src) if own is not None else "module.exports"
                self.facts.exports.append(
                    {"exported_name": "default", "local_name": local, "line": line}
                )
            return
        if left.type == "member_expression":
            prop = left.child_by_field_name("property")
            obj = left.child_by_field_name("object")
            if prop is None or obj is None:
                return
            member = _text(prop, src)
            obj_text = _text(obj, src)
            if obj_text in {"module.exports", "exports"}:
                local = _text(right, src) if right.type == "identifier" else member
                self.facts.exports.append(
                    {"exported_name": member, "local_name": local, "line": line}
                )
            if right.type == "identifier":
                owner = obj_text.replace(".prototype", "")
                self.facts.bindings.append(
                    {
                        "owner": _flat(owner, 80),
                        "member": member,
                        "target": _text(right, src),
                        "line": line,
                    }
                )

    def _handle_object_bindings(self, obj: Node, owner: str | None) -> None:
        """`{ start, stop, device: manager }` — shorthand properties bind members to locals."""
        src = self.src
        for prop in obj.named_children:
            if prop.type == "shorthand_property_identifier":
                n = _text(prop, src)
                self.facts.bindings.append(
                    {
                        "owner": owner or "<object>",
                        "member": n,
                        "target": n,
                        "line": prop.start_point[0] + 1,
                    }
                )
            elif prop.type == "pair":
                key = prop.child_by_field_name("key")
                val = prop.child_by_field_name("value")
                if key is not None and val is not None and val.type == "identifier":
                    self.facts.bindings.append(
                        {
                            "owner": owner or "<object>",
                            "member": _text(key, src).strip("'\""),
                            "target": _text(val, src),
                            "line": prop.start_point[0] + 1,
                        }
                    )

    # ---------------------------------------------------------------- calls/strings
    def _add_call(self, node: Node, ctx: _Ctx, is_new: bool = False) -> None:
        src = self.src
        fn = node.child_by_field_name("constructor" if is_new else "function")
        if fn is None:
            return
        callee_text = _flat(_text(fn, src), 160)
        callee_name = callee_text
        receiver = None
        if fn.type == "member_expression":
            prop = fn.child_by_field_name("property")
            obj = fn.child_by_field_name("object")
            callee_name = _text(prop, src) if prop is not None else callee_text
            receiver = _flat(_text(obj, src), 120) if obj is not None else None
        elif fn.type != "identifier":
            callee_name = callee_text.split("(")[0].split(".")[-1] or callee_text
        args = node.child_by_field_name("arguments")
        args_text = _flat(_text(args, src), MAX_ARGS_TEXT) if args is not None else ""
        first_arg = None
        if args is not None and args.named_children:
            a0 = args.named_children[0]
            if a0.type in {"string", "template_string"}:
                first_arg = _string_value(a0, src)[:200]
            elif a0.type in {"member_expression", "identifier"}:
                first_arg = _flat(_text(a0, src), 120)
        parent = node.parent
        while parent is not None and parent.type == "parenthesized_expression":
            parent = parent.parent
        awaited = parent is not None and parent.type == "await_expression"
        self.facts.calls.append(
            {
                "caller_local": ctx.symbol,
                "callee_text": callee_text,
                "callee_name": callee_name,
                "receiver": receiver,
                "line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "byte": node.start_byte,
                "awaited": awaited,
                "in_branch": ctx.branch,
                "in_callback": ctx.callback,
                "in_loop": ctx.loop,
                "is_new": is_new,
                "args_text": args_text,
                "first_arg": first_arg,
            }
        )

    def _add_string(self, node: Node, ctx: _Ctx) -> None:
        value = _string_value(node, self.src)
        if not value.strip():
            return
        context, detail = _string_context(node, self.src)
        self.facts.strings.append(
            {
                "value": value[:MAX_STRING_LEN],
                "line": node.start_point[0] + 1,
                "symbol_local": ctx.symbol,
                "context": context,
                "detail": detail,
                "is_template": node.type == "template_string",
            }
        )

    # ---------------------------------------------------------------- traversal
    def run(self, tree_root: Node) -> FileFacts:
        src = self.src
        self.facts.line_count = tree_root.end_point[0] + 1
        module_sym = self._add_symbol("<module>", "<module>", "module", tree_root, None)
        stack: list[tuple[Node, _Ctx]] = [(tree_root, _Ctx(symbol=module_sym))]
        while stack:
            node, ctx = stack.pop()
            ntype = node.type
            child_ctx = ctx

            if ntype in FUNCTION_TYPES:
                named = _function_name(node, src, ctx)
                if named is not None:
                    name, qualname = named
                    is_async = any(c.type == "async" for c in node.children)
                    is_member = ntype == "method_definition" or ctx.owner or "." in qualname
                    kind = "method" if is_member else "function"
                    if ntype == "arrow_function":
                        kind = "arrow" if kind == "function" else kind
                    sym = self._add_symbol(
                        name,
                        qualname,
                        kind,
                        node,
                        ctx.symbol,
                        _params(node, src),
                        is_async,
                        _doc_comment(_statement_of(node), src),
                    )
                    child_ctx = _Ctx(symbol=sym)
                else:
                    # Anonymous function: calls inside may run later / zero times.
                    loop = ctx.loop
                    p = node.parent
                    if p is not None and p.type == "arguments":
                        call = p.parent
                        fnode = call.child_by_field_name("function") if call is not None else None
                        if fnode is not None and fnode.type == "member_expression":
                            prop = fnode.child_by_field_name("property")
                            if prop is not None and _text(prop, src) in LOOP_METHODS:
                                loop = True
                    child_ctx = _Ctx(
                        symbol=ctx.symbol, branch=ctx.branch, callback=True, loop=loop, owner=None
                    )
            elif ntype in CLASS_TYPES:
                name_node = node.child_by_field_name("name")
                name = _text(name_node, src) if name_node is not None else None
                if (
                    name is None
                    and node.parent is not None
                    and node.parent.type == "variable_declarator"
                ):
                    t = node.parent.child_by_field_name("name")
                    name = _text(t, src) if t is not None else None
                name = name or "<anonymous class>"
                sym = self._add_symbol(
                    name,
                    name,
                    "class",
                    node,
                    ctx.symbol,
                    doc=_doc_comment(_statement_of(node), src),
                )
                child_ctx = _Ctx(symbol=sym, owner=name)
            elif ntype == "object":
                owner = _owner_for_object(node, src)
                self._handle_object_bindings(node, owner)
                if owner:
                    child_ctx = _Ctx(
                        symbol=ctx.symbol,
                        branch=ctx.branch,
                        callback=ctx.callback,
                        loop=ctx.loop,
                        owner=owner,
                    )
            elif ntype == "variable_declarator":
                self._handle_declarator(node)
            elif ntype == "import_statement":
                self._handle_import_statement(node)
            elif ntype == "export_statement":
                self._handle_export_statement(node)
            elif ntype == "assignment_expression":
                self._handle_assignment(node, ctx)
            elif ntype == "call_expression":
                self._add_call(node, ctx)
            elif ntype == "new_expression":
                self._add_call(node, ctx, is_new=True)
            elif ntype in {"string", "template_string"}:
                self._add_string(node, ctx)
            elif ntype == "member_expression":
                parent = node.parent
                is_outer = not (
                    parent is not None
                    and parent.type == "member_expression"
                    and parent.child_by_field_name("object") == node
                )
                if is_outer:
                    root = node
                    while root.type == "member_expression":
                        obj = root.child_by_field_name("object")
                        if obj is None:
                            break
                        root = obj
                    if root.type == "identifier" and CONST_ROOT_RE.match(_text(root, src)):
                        self.facts.const_refs.append(
                            {
                                "path": _flat(_text(node, src), 160),
                                "line": node.start_point[0] + 1,
                                "symbol_local": ctx.symbol,
                            }
                        )

            # Push children in reverse so they are visited in source order.
            children = node.named_children
            for child in reversed(children):
                cctx = child_ctx
                if ntype == "if_statement" and child != node.child_by_field_name("condition"):
                    cctx = _replace(child_ctx, branch=True)
                elif ntype == "ternary_expression" and child != node.child_by_field_name(
                    "condition"
                ):
                    cctx = _replace(child_ctx, branch=True)
                elif ntype in {"switch_case", "switch_default", "catch_clause"}:
                    cctx = _replace(child_ctx, branch=True)
                elif ntype == "binary_expression" and child == node.child_by_field_name("right"):
                    op = node.child_by_field_name("operator")
                    if op is not None and op.type in {"&&", "||", "??"}:
                        cctx = _replace(child_ctx, branch=True)
                elif ntype in LOOP_TYPES and child == node.child_by_field_name("body"):
                    cctx = _replace(child_ctx, loop=True, branch=True)
                elif ntype == "optional_chain":
                    cctx = _replace(child_ctx, branch=True)
                stack.append((child, cctx))

        # Ordinals: source order of call sites within each enclosing symbol.
        self.facts.calls.sort(key=lambda c: (c["caller_local"], c["byte"]))
        last_symbol, ordinal = None, 0
        for call in self.facts.calls:
            if call["caller_local"] != last_symbol:
                last_symbol, ordinal = call["caller_local"], 0
            call["ordinal"] = ordinal
            ordinal += 1
        # Mark exported symbols.
        exported_locals = {e["local_name"] for e in self.facts.exports}
        for sym in self.facts.symbols:
            if sym["kind"] != "module" and sym["parent_local"] == module_sym:
                if sym["name"] in exported_locals:
                    sym["is_exported"] = True
        return self.facts


def _replace(ctx: _Ctx, **changes: bool) -> _Ctx:
    return _Ctx(
        symbol=ctx.symbol,
        branch=changes.get("branch", ctx.branch),
        callback=changes.get("callback", ctx.callback),
        loop=changes.get("loop", ctx.loop),
        owner=ctx.owner,
    )


def extract(src: bytes, parser: Parser | None = None) -> tuple[FileFacts, int]:
    """Parse `src` and extract facts. Returns (facts, number_of_error_nodes)."""
    parser = parser or make_parser()
    tree = parser.parse(src)
    errors = 0
    if tree.root_node.has_error:
        stack = [tree.root_node]
        while stack:
            n = stack.pop()
            if n.type == "ERROR" or n.is_missing:
                errors += 1
            elif n.has_error:
                stack.extend(n.children)
    facts = _Extractor(src).run(tree.root_node)
    return facts, errors
