"""Confidence assessment and grounded refinement round 2: the constrained rewrite (D-028).

`assess` combines deterministic signals of a ranked list: fused-score margin between the top two
hits, BM25/embedding agreement on the top hit, whether the top domain-catalog entry lands in the
top 3, and how many of the top candidates passed verification.

`constrained_rewrite` never lets the LLM invent a name: it builds a numbered list of *real*
index terms — identifiers (qualnames) of the current candidates and their call-graph neighbours,
doc-comment phrases of those symbols, and domain-catalog names — asks the LLM to pick the ones
relevant to the question, and searches again with the picked terms only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass

from src.agent.llm import LLMClient, LLMUnavailable, parse_json_object

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefineConfig:
    """Refinement policy. Round 1 = graph/catalog expansion (deterministic, before the first
    results); round 2 = constrained rewrite (LLM, after the first results)."""

    gate: str = "low"  # "low" = refine only when assess() says low confidence; "always"
    expand: bool = True
    rewrite: bool = True
    max_rounds: int = 2
    margin_threshold: float = 0.2  # margin at or above this counts as a confident top hit
    round2_weight: float = 1.0  # RRF weight of the rewrite's list when merging
    max_terms: int = 40
    max_picks: int = 5
    rrf_k: int = 60

    def to_dict(self) -> dict:
        return asdict(self)


def assess(
    hits: list,
    catalog_ids: list[int] | None = None,
    verified: int | None = None,
    attempted: int | None = None,
    margin_threshold: float = 0.2,
) -> dict:
    """Deterministic confidence signals and a label."""
    if not hits:
        return {
            "label": "low",
            "margin": 0.0,
            "agreement": False,
            "catalog_in_top3": False,
            "verified_ratio": 0.0,
        }
    s1 = hits[0].score or 0.0
    s2 = hits[1].score if len(hits) > 1 else 0.0
    margin = (s1 - s2) / s1 if s1 > 0 else 0.0
    r = hits[0].ranks or {}
    agreement = r.get("bm25", 99) <= 3 and r.get("vector", 99) <= 3
    top3 = {h.chunk_id for h in hits[:3]}
    catalog_in_top3 = bool(catalog_ids and top3 & set(catalog_ids))
    ratio = (verified / attempted) if attempted else 1.0
    confident = (agreement or margin >= margin_threshold or catalog_in_top3) and ratio >= 0.5
    return {
        "label": "high" if confident else "low",
        "margin": round(margin, 3),
        "agreement": agreement,
        "catalog_in_top3": catalog_in_top3,
        "verified_ratio": round(ratio, 2),
    }


_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def humanise(term: str) -> str:
    """`DeviceManager.purgeStates` → "device manager purge states"."""
    words = []
    for part in re.split(r"[^A-Za-z0-9]+", term):
        words.extend(w.lower() for w in _CAMEL.findall(part))
    return " ".join(words)


def candidate_terms(
    retriever, query: str, hits: list, neighbours: dict[int, list], max_terms: int = 40
) -> list[dict]:
    """Real index vocabulary around the current candidates (no free-form text)."""
    from src.catalog.search import search_catalog

    conn = retriever.conn
    terms: list[dict] = []
    seen: set[str] = set()

    def add(text: str, kind: str) -> None:
        key = text.strip().lower()
        if key and key not in seen and len(terms) < max_terms:
            seen.add(key)
            terms.append({"text": text.strip(), "kind": kind})

    sym_ids = [h.symbol_id for h in hits[:12] if h.symbol_id is not None]
    rows = {}
    if sym_ids:
        marks = ",".join("?" * len(sym_ids))
        for r in conn.execute(
            f"SELECT id, qualname, kind, doc FROM symbols WHERE id IN ({marks})", sym_ids
        ):
            rows[r["id"]] = r
    for h in hits[:12]:
        r = rows.get(h.symbol_id)
        if r is None or r["kind"] == "module":
            continue
        add(r["qualname"], "identifier")
        doc = re.sub(r"@\w+.*", "", r["doc"] or "").strip()
        if doc:
            add(" ".join(doc.split()[:12]), "doc")
    for lst in neighbours.values():
        for _cid, _rel, label in lst:
            add(label, "identifier")
    for c in search_catalog(conn, query, limit=5):
        add(c.name + (f" ({c.detail})" if c.detail else ""), "catalog")
    return terms


PICK_PROMPT = """You help search a JavaScript codebase (Gladys Assistant, home automation).
A developer asked: "{question}"
Below are REAL names and phrases found in the code. Pick the ones (at most {max_picks}) that are
most relevant to what the developer is looking for. Do not invent anything: answer only with
their numbers. Reply as JSON {{"picks": [numbers]}}; use [] if none fit.
{items}"""


def pick_terms(llm: LLMClient, question: str, terms: list[dict], max_picks: int = 5) -> list[dict]:
    items = "\n".join(f"{i}. {t['text']}" for i, t in enumerate(terms, 1))
    try:
        resp = llm.chat(
            [
                {
                    "role": "user",
                    "content": PICK_PROMPT.format(
                        question=question, max_picks=max_picks, items=items
                    ),
                }
            ],
            max_tokens=40,
        )
        picks = parse_json_object(resp.text).get("picks", [])
    except (LLMUnavailable, ValueError) as exc:
        logger.info("term picking skipped: %s", exc)
        return []
    out = []
    for p in picks if isinstance(picks, list) else []:
        try:
            i = int(p)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= len(terms) and terms[i - 1] not in out:  # only listed terms survive
            out.append(terms[i - 1])
    return out[:max_picks]


def rewrite_query(picked: list[dict]) -> str:
    """Search text built only from the picked real terms (identifiers humanised)."""
    parts = [humanise(t["text"]) if t["kind"] == "identifier" else t["text"] for t in picked]
    return " ".join(p for p in parts if p)


def merge_rounds(first: list, second: list, cfg: RefineConfig, k: int) -> list:
    """Deterministic RRF of two ranked Hit lists (first round weight 1)."""
    score: dict[int, float] = {}
    keep: dict[int, object] = {}
    for weight, hits in ((1.0, first), (cfg.round2_weight, second)):
        for rank, h in enumerate(hits, 1):
            score[h.chunk_id] = score.get(h.chunk_id, 0.0) + weight / (cfg.rrf_k + rank)
            keep.setdefault(h.chunk_id, h)
    order = sorted(score, key=lambda c: -score[c])[:k]
    out = []
    for cid in order:
        h = keep[cid]
        h.score = round(score[cid], 5)
        out.append(h)
    return out
