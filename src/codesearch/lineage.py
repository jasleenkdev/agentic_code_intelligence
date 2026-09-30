"""Evolutionary retrieval (bonus): search every version at once, handling near-duplicates.

1. **Lineages.** Snippets of different versions that are "the same thing" are grouped:
   first by a stable key when the source provides one (codebases: `path::symbol#window`),
   then by content — identical text is one lineage, and near-identical snippets (embedding
   cosine ≥ `cos_min` to a nearest neighbour in another version AND token-shape Jaccard ≥
   `jaccard_min` after identifier normalisation) are merged with union-find. The APPS
   benchmark gives the system no keys, so lineages there are inferred purely from content.
2. **Version hints.** "latest", "newest", "current" → latest; "original", "first",
   "initial", "oldest" → earliest; an explicit label ("v2", "5.1.0") → that version. The hint
   phrase is removed from the text before encoding.
3. **Ranking.** Scores are computed for every (snippet, version); each lineage takes its best
   member score (collapse), so near-duplicate versions no longer crowd the top 10; the version
   shown for a lineage is the hinted one if present, otherwise the best-scoring member, with ties
   within `tie_eps` resolved toward the most recent version.
"""

from __future__ import annotations

import io
import keyword
import re
import tokenize
from dataclasses import dataclass, field

import numpy as np

_HINT_LATEST = r"(?:latest|newest|most recent|current|last)"
_HINT_EARLIEST = r"(?:original|first|initial|oldest|earliest)"


@dataclass
class VersionHint:
    kind: str | None  # "latest" | "earliest" | "label" | None
    label: str | None
    phrase: str
    cleaned_query: str


def parse_version_hint(query: str, versions: list[str]) -> VersionHint:
    """Find a version hint in the query and strip it."""
    q = query
    for v in sorted(versions, key=len, reverse=True):
        rx = re.compile(
            rf"\s*[\(\[]?\b(?:in|at|from|of)?\s*(?:version|release|tag|commit)?\s*"
            rf"{re.escape(v)}\b[\)\]]?",
            re.IGNORECASE,
        )
        m = rx.search(q)
        if m:
            return VersionHint(
                "label", v, m.group(0).strip(), (q[: m.start()] + q[m.end() :]).strip()
            )
    for kind, word in (("latest", _HINT_LATEST), ("earliest", _HINT_EARLIEST)):
        rx = re.compile(
            rf"\s*[\(\[]?\b(?:in |from |of )?(?:the )?{word} (?:version|release|revision|code)"
            rf"(?: of the code)?[\)\]]?",
            re.IGNORECASE,
        )
        m = rx.search(q)
        if m:
            return VersionHint(
                kind, None, m.group(0).strip(), (q[: m.start()] + q[m.end() :]).strip()
            )
    return VersionHint(None, None, "", query)


def shape_tokens(code: str) -> set[str]:
    """Token 3-grams with identifiers/literals abstracted (robust to renames and reformatting)."""
    try:
        toks = []
        for tk in tokenize.generate_tokens(io.StringIO(code).readline):
            if tk.type in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            ):
                continue
            if tk.type == tokenize.NAME and not keyword.iskeyword(tk.string):
                toks.append("ID")
            elif tk.type == tokenize.NUMBER:
                toks.append("NUM")
            elif tk.type == tokenize.STRING:
                toks.append("STR")
            else:
                toks.append(tk.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        toks = re.findall(r"\w+|[^\s\w]", code)
    return {" ".join(toks[i : i + 3]) for i in range(max(len(toks) - 2, 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def infer_lineages(
    texts: list[str],
    versions: list[str],
    embeddings: np.ndarray,
    keys: list[str] | None = None,
    cos_min: float = 0.9,
    jaccard_min: float = 0.3,
    neighbours: int = 5,
) -> list[int]:
    """Lineage id per snippet (index of the lineage's first member)."""
    n = len(texts)
    uf = _UnionFind(n)
    first_by: dict[str, int] = {}
    for i in range(n):  # stable keys, then identical text
        for k in ([f"key:{keys[i]}"] if keys and keys[i] else []) + [f"text:{hash(texts[i])}"]:
            if k in first_by:
                uf.union(first_by[k], i)
            else:
                first_by[k] = i
    if n > 1 and cos_min < 1.0:
        shapes: dict[int, set[str]] = {}
        vers = np.array(versions)
        block = 2048
        for s in range(0, n, block):
            sims = embeddings[s : s + block] @ embeddings.T
            for r in range(sims.shape[0]):
                i = s + r
                row = sims[r]
                row[vers == vers[i]] = -1  # only link across versions
                k = min(neighbours, n - 1)
                for j in np.argpartition(-row, k - 1)[:k]:
                    if row[j] < cos_min or uf.find(i) == uf.find(int(j)):
                        continue
                    a = shapes.setdefault(i, shape_tokens(texts[i]))
                    b = shapes.setdefault(int(j), shape_tokens(texts[int(j)]))
                    if jaccard(a, b) >= jaccard_min:
                        uf.union(i, int(j))
    return [uf.find(i) for i in range(n)]


@dataclass
class LineageHit:
    lineage: int
    index: int  # chosen member (snippet index)
    score: float
    members: list[tuple[int, float]] = field(default_factory=list)  # (index, score), by version


def collapse(
    scores: np.ndarray,
    lineages: list[int] | np.ndarray,
    version_rank: np.ndarray,
    hint: VersionHint,
    version_labels: list[str],
    top_k: int = 10,
    tie_eps: float = 0.01,
    pool: int = 200,
    members_of: dict[int, list[int]] | None = None,
) -> list[LineageHit]:
    """Collapse one query's per-snippet scores to per-lineage hits (best first).

    `version_rank[i]` = chronological position of snippet i's version (0 = oldest);
    `members_of` = lineage id → all its snippet indices (see `members_by_lineage`).
    """
    lin = np.asarray(lineages)
    order = np.argsort(-scores)[: max(pool, top_k * 20)]
    groups: dict[int, list[int]] = {}
    for i in order:
        groups.setdefault(int(lin[i]), []).append(int(i))
    hits = []
    for g, members in groups.items():
        if members_of is not None:  # every version of the lineage, not only those in the pool
            members = members_of.get(g, members)
        best = max(scores[m] for m in members)
        chosen = _choose_member(members, scores, version_rank, hint, version_labels, tie_eps)
        members_sorted = sorted(members, key=lambda m: version_rank[m])
        hits.append(
            LineageHit(g, chosen, float(best), [(m, float(scores[m])) for m in members_sorted])
        )
    hits.sort(key=lambda h: -h.score)
    return hits[:top_k]


def _choose_member(members, scores, version_rank, hint, version_labels, tie_eps) -> int:
    if hint.kind == "label" and hint.label is not None:
        exact = [m for m in members if version_labels[m] == hint.label]
        if exact:
            return max(exact, key=lambda m: scores[m])
    if hint.kind == "earliest":
        return min(members, key=lambda m: (version_rank[m], -scores[m]))
    if hint.kind == "latest":
        return max(members, key=lambda m: (version_rank[m], scores[m]))
    best = max(scores[m] for m in members)
    near = [m for m in members if scores[m] >= best - tie_eps]
    return max(near, key=lambda m: (version_rank[m], scores[m]))


def members_by_lineage(lineages: list[int] | np.ndarray) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i, g in enumerate(lineages):
        out.setdefault(int(g), []).append(i)
    return out
