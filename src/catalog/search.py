"""Deterministic search over the domain catalog: plain-English phrase → catalog entries.

"where is the Bluetooth-settings deeplink used?" → kind hint ``front_route`` (from "deeplink"),
content tokens {bluetooth, settings→config} → ``/dashboard/integration/device/bluetooth/config``
(component ``BluetoothSettingsPage``) with its definition and every usage.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

KIND_HINTS: dict[str, str] = {
    "deeplink": "front_route",
    "deeplinks": "front_route",
    "deep": "front_route",
    "link": "front_route",
    "route": "front_route",
    "page": "front_route",
    "screen": "front_route",
    "endpoint": "api_route",
    "api": "api_route",
    "http": "api_route",
    "rest": "api_route",
    "event": "event",
    "events": "event",
    "emitted": "event",
    "intent": "intent",
    "intents": "intent",
    "tool": "mcp_tool",
    "tools": "mcp_tool",
    "mcp": "mcp_tool",
    "service": "service",
    "integration": "service",
    "action": "action",
    "websocket": "ws_message",
    "ws": "ws_message",
    "job": "job",
    "uri": "uri",
    "url": "uri",
    "scheme": "uri",
}
SYNONYMS: dict[str, tuple[str, ...]] = {
    "settings": ("config", "settings"),
    "setting": ("config", "settings"),
    "configuration": ("config",),
    "preferences": ("config", "settings"),
    "bt": ("bluetooth",),
    "lights": ("light",),
    "devices": ("device",),
    "scenes": ("scene",),
    "switches": ("switch",),
    "turn": ("turn",),
}
STOPWORDS = frozenset(
    """a an the is are was be of in to for from with by at where which who what how when do does
    did we our this that it its used use uses using usage usages find show list all any me get
    defined definition declared called there code file files place places handled handle""".split()
)
_SPLIT_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def _stem(tok: str) -> str:
    """Tiny plural stemmer so 'events' ~ 'event', 'devices' ~ 'device' (not 'status')."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    out = []
    for word in re.split(r"[^A-Za-z0-9]+", text):
        if not word:
            continue
        parts = _SPLIT_RE.findall(word) or [word]
        out.extend(_stem(p.lower()) for p in parts)
    return out


@dataclass
class CatalogHit:
    kind: str
    name: str
    score: float
    definitions: list[dict] = field(default_factory=list)
    usages: list[dict] = field(default_factory=list)
    detail: str | None = None


def parse_query(query: str) -> tuple[set[str], list[set[str]]]:
    """Return (kind hints, content-token groups). Each group is a set of acceptable synonyms."""
    kinds: set[str] = set()
    groups: list[set[str]] = []
    for tok in tokenize(query):
        if tok in KIND_HINTS:
            kinds.add(KIND_HINTS[tok])
            if tok not in {"page", "service", "integration", "event", "intent", "action", "job"}:
                continue
        if tok in STOPWORDS or len(tok) < 2:
            continue
        groups.append({_stem(t) for t in SYNONYMS.get(tok, (tok,))})
    return kinds, groups


def _rows(conn: sqlite3.Connection, kind: str, name: str) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT c.kind, c.name, c.value, c.role, c.line, c.detail, f.path AS file_path,"
            " c.symbol_id FROM catalog c JOIN files f ON f.id = c.file_id"
            " WHERE c.kind = ? AND c.name = ? ORDER BY c.role != 'definition', f.path, c.line",
            (kind, name),
        )
    ]


def search_catalog(
    conn: sqlite3.Connection, query: str, kind: str | None = None, limit: int = 5
) -> list[CatalogHit]:
    """Rank catalog entries (grouped by kind+name) against a phrase; include all their rows."""
    kinds, groups = parse_query(query)
    if kind:
        kinds = {kind}
    if not groups:
        return []
    params: list = []
    where = "role IN ('definition')"
    if kinds:
        where += f" AND kind IN ({','.join('?' * len(kinds))})"
        params.extend(sorted(kinds))
    entries = conn.execute(
        f"SELECT DISTINCT kind, name, value, detail FROM catalog WHERE {where}",  # noqa: S608
        params,
    ).fetchall()
    if not entries and kinds:  # hinted kind absent from this repo: search all kinds
        entries = conn.execute(
            "SELECT DISTINCT kind, name, value, detail FROM catalog WHERE role = 'definition'"
        ).fetchall()
    scored: dict[tuple[str, str], tuple[float, str | None]] = {}
    for e in entries:
        etoks = set(tokenize(f"{e['name']} {e['value']} {e['detail'] or ''}"))
        matched = sum(1 for g in groups if g & etoks)
        if matched == 0:
            continue
        coverage = matched / len(groups)
        # Prefer short, specific entries (fewer unmatched tokens) and kind agreement.
        precision = matched / max(len(etoks), 1)
        score = coverage + 0.2 * precision + (0.3 if e["kind"] in kinds else 0.0)
        key = (e["kind"], e["name"])
        if key not in scored or scored[key][0] < score:
            scored[key] = (score, e["detail"])
    ranked = sorted(scored.items(), key=lambda kv: -kv[1][0])[:limit]
    hits = []
    for (k, name), (score, detail) in ranked:
        rows = _rows(conn, k, name)
        hits.append(
            CatalogHit(
                kind=k,
                name=name,
                score=round(score, 3),
                definitions=[r for r in rows if r["role"] == "definition"],
                usages=[r for r in rows if r["role"] != "definition"],
                detail=detail,
            )
        )
    return hits
