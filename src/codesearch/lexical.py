"""Lexical signals between a problem statement and a code snippet.

1. **Literal overlap** — string and numeric literals in the code (`"IMPOSSIBLE"`, `"Yes"`,
   `998244353`, `10**9+7`) that also occur in the problem text. Solutions print the exact output
   tokens the statement asks for, so a shared rare literal is strong evidence. Scored as the sum
   of IDF weights (over the snippet library) of matched literals.
2. **BM25** over identifier-split tokens of code vs words of the query (weak on APPS because
   stdin solutions use short variable names, kept as an ablation).

Both are computed as sparse matrices so a whole query set is scored at once.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_STR_RE = re.compile(r"""(?:[rbuRBU]{0,2})("([^"\\\n]|\\.){1,40}"|'([^'\\\n]|\\.){1,40}')""")
_NUM_RE = re.compile(r"(?<![\w.])(\d{3,}|10\s*\*\*\s*\d+(?:\s*\+\s*\d+)?|1e\d+)(?![\w.])")
_TRIVIAL = {
    "",
    " ",
    "\n",
    ",",
    ".",
    ":",
    ";",
    "-",
    "+",
    "*",
    "/",
    "0",
    "1",
    "2",
    "r",
    "w",
    "%d",
    "%s",
    "{}",
    "%",
    "#",
    "|",
    "(",
    ")",
    "[",
    "]",
    "\t",
    "..",
    "__main__",
}


MAX_EXPONENT = 30  # 10**k with k above this stays symbolic (no giant integers, see test)


def _norm_num(tok: str) -> str:
    """`10**9+7` / `1e9` → decimal digits (as statements write them); other numbers unchanged."""
    t = re.sub(r"\s+", "", tok)
    m = re.fullmatch(r"10\*\*(\d+)(?:\+(\d+))?", t) or re.fullmatch(r"1e(\d+)()", t)
    if m:
        exp = int(m.group(1)) if len(m.group(1)) < 4 else MAX_EXPONENT + 1
        if exp > MAX_EXPONENT:
            return t
        return str(10**exp + int(m.group(2) or 0))
    return t


def code_literals(code: str) -> set[str]:
    """Distinctive literals: strings (≥2 non-space chars, not format-only), numbers ≥ 100."""
    out: set[str] = set()
    for m in _STR_RE.finditer(code):
        s = m.group(1)[1:-1].strip()
        if s in _TRIVIAL or len(s) < 2 or re.fullmatch(r"[\s%{}\d.:,\\n-]*", s):
            continue
        out.add(s)
    for m in _NUM_RE.finditer(code):
        n = _norm_num(m.group(1))
        if len(n) >= 3:
            out.add(n)
    return out


def _pow10(m: re.Match) -> str:
    exp = int(m.group(1))
    return str(10**exp) if exp <= MAX_EXPONENT else m.group(0)


def _query_text_forms(q: str) -> tuple[str, set[str]]:
    t = q.replace("$", "")
    t = re.sub(r"10\^\{?(\d{1,2})\}?", _pow10, t)  # 10^9 → 1000000000; huge exponents untouched
    nums = {n.replace(",", "") for n in re.findall(r"\d[\d,]{2,}", t)}
    return t, nums


class LiteralMatcher:
    """IDF-weighted literal overlap between queries and a fixed snippet library."""

    def __init__(self, snippets: list[str]) -> None:
        self.lits = [code_literals(s) for s in snippets]
        df = Counter(lit for ls in self.lits for lit in ls)
        n = max(len(snippets), 1)
        self.idf = {lit: math.log(1 + n / c) for lit, c in df.items()}
        self.by_lit: dict[str, list[int]] = {}
        for i, ls in enumerate(self.lits):
            for lit in ls:
                self.by_lit.setdefault(lit, []).append(i)
        self._word_lits = {x for x in self.by_lit if not x.isdigit() and re.fullmatch(r"\S+", x)}
        self._phrase_lits = [x for x in self.by_lit if not x.isdigit() and x not in self._word_lits]

    def score(self, queries: list[str]) -> np.ndarray:
        out = np.zeros((len(queries), len(self.lits)), dtype=np.float32)
        for qi, q in enumerate(queries):
            for lit in self.matching_literals(q):
                out[qi, self.by_lit[lit]] += self.idf[lit]
        return out

    def matching_literals(self, query: str) -> list[str]:
        """Library literals present in the query: numbers by value, single words as whole
        tokens (case-sensitive), anything longer as a substring."""
        text, nums = _query_text_forms(query)
        words = set(re.findall(r"[^\s]+", text)) | set(re.findall(r"\w+", text))
        hits = [n for n in nums if n in self.by_lit]
        hits += [w for w in words if w in self._word_lits]
        hits += [lit for lit in self._phrase_lits if lit in text]
        return list(dict.fromkeys(hits))

    def matched(self, query: str, doc_index: int) -> list[str]:
        return sorted(set(self.matching_literals(query)) & self.lits[doc_index])


_WORD_RE = re.compile(r"[A-Za-z]+")
_STOP = frozenset(
    "the a an of to in and or is are be for on with that this it as by from at you your each "
    "which print input output line lines first second integer integers number numbers given "
    "contains test case cases".split()
)


def _words(text: str) -> list[str]:
    toks = []
    for w in _WORD_RE.findall(text):
        for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", w) or [w]:
            p = part.lower()
            if len(p) > 1 and p not in _STOP:
                toks.append(p)
    return toks


class BM25Matrix:
    """Plain BM25 (k1=1.2, b=0.75) producing a dense [n_queries, n_docs] score matrix."""

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        toks = [_words(d) for d in docs]
        self.n = len(docs)
        self.dl = np.array([len(t) for t in toks], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if self.n else 1.0
        self.tf: dict[str, dict[int, int]] = {}
        for i, t in enumerate(toks):
            for w, c in Counter(t).items():
                self.tf.setdefault(w, {})[i] = c
        self.idf = {
            w: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.tf.items()
        }

    def score(self, queries: list[str]) -> np.ndarray:
        out = np.zeros((len(queries), self.n), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.dl / self.avgdl)
        for qi, q in enumerate(queries):
            for w in set(_words(q)):
                posting = self.tf.get(w)
                if not posting:
                    continue
                idx = np.fromiter(posting.keys(), dtype=np.int64)
                tf = np.fromiter(posting.values(), dtype=np.float32)
                out[qi, idx] += self.idf[w] * tf * (self.k1 + 1) / (tf + norm[idx])
        return out
