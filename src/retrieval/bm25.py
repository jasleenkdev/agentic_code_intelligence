"""BM25 lexical index over chunks with identifier splitting (D-007)."""

from __future__ import annotations

import logging
import pickle
import re
import sqlite3

import numpy as np
from rank_bm25 import BM25Okapi

from src import config

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*|\d+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
JS_NOISE = frozenset(
    """const let var function return await async this if else for while do new of in typeof
    instanceof null undefined true false try catch finally throw class extends super import export
    from default module exports require switch case break continue yield void delete static get
    set""".split()
)
EN_STOP = frozenset(
    """a an the is are was be to of and or on at by with where which who what how when does do did
    we our us it its that this there here code find show me please can could would should""".split()
)


def split_identifier(word: str) -> list[str]:
    parts = [p.lower() for chunk in word.split("_") for p in _CAMEL_RE.findall(chunk)]
    return [p for p in parts if len(p) > 1]


def tokenize_code(text: str) -> list[str]:
    """Identifier-aware tokens: full identifier (lowercased) + its camel/snake parts."""
    out: list[str] = []
    for word in _WORD_RE.findall(text):
        low = word.lower()
        if low in JS_NOISE:
            continue
        parts = split_identifier(word)
        if len(parts) > 1 or (parts and parts[0] != low):
            out.append(low)
        out.extend(parts or ([low] if len(low) > 1 else []))
    return out


def tokenize_query(text: str) -> list[str]:
    return [t for t in tokenize_code(text) if t not in EN_STOP]


class BM25Index:
    def __init__(self, bm25: BM25Okapi, chunk_ids: np.ndarray) -> None:
        self.bm25 = bm25
        self.chunk_ids = chunk_ids
        self.index_of = {int(c): i for i, c in enumerate(chunk_ids)}

    def search(self, query: str, n: int = 50) -> list[tuple[int, float]]:
        toks = tokenize_query(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [(int(self.chunk_ids[i]), float(scores[i])) for i in top if scores[i] > 0]


def build_bm25(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id, header, text FROM chunks ORDER BY id").fetchall()
    corpus = [tokenize_code(r["header"] + "\n" + r["text"]) for r in rows]
    bm25 = BM25Okapi(corpus)
    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    with open(config.BM25_PATH, "wb") as fh:
        pickle.dump({"bm25": bm25, "chunk_ids": ids}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("bm25: %d chunks, avg %.0f tokens", len(corpus), bm25.avgdl)


def load_bm25() -> BM25Index:
    if not config.BM25_PATH.exists():
        raise FileNotFoundError(f"BM25 index missing at {config.BM25_PATH}; run the indexer.")
    with open(config.BM25_PATH, "rb") as fh:
        data = pickle.load(fh)  # noqa: S301 — our own file, written by build_bm25
    return BM25Index(data["bm25"], data["chunk_ids"])
