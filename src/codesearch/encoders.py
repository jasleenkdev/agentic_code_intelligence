"""CPU-only embedding encoders with a content-hash embedding cache.

Every model is loaded with `device="cpu"` (CUDA hidden, MPS never selected): the hackathon
requires CPU inference. Embeddings are cached by `sha1(model, max_seq, prefixed text)`, so an
unchanged snippet is never re-encoded — this is also the P1 incremental-rebuild mechanism
(a new version re-encodes only snippets whose text changed, D-038).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src import config

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU in every model below
logger = logging.getLogger(__name__)

EMB_CACHE_DIR = Path(
    os.environ.get("CODESEARCH_EMB_CACHE", "") or config.INDEX_DIR / "emb_cache"
).resolve()


@dataclass(frozen=True)
class EncoderProfile:
    """How one model is called: prefixes, sequence cap, loading flags, licence."""

    key: str
    model_id: str
    query_prefix: str = ""
    doc_prefix: str = ""
    max_seq: int = 512
    trust_remote_code: bool = False
    batch_size: int = 16
    params_m: int = 0
    licence: str = ""


APPS_INSTRUCT = "Given a programming problem description, retrieve the code solution that solves it"

PROFILES: dict[str, EncoderProfile] = {
    p.key: p
    for p in [
        EncoderProfile(
            "bge-small",
            "BAAI/bge-small-en-v1.5",
            query_prefix="Represent this sentence for searching relevant passages: ",
            batch_size=32,
            params_m=33,
            licence="MIT",
        ),
        EncoderProfile(
            "coderankembed",
            "nomic-ai/CodeRankEmbed",
            query_prefix="Represent this query for searching relevant code: ",
            trust_remote_code=True,
            params_m=137,
            licence="MIT",
        ),
        EncoderProfile(
            "granite-small-r2",
            "ibm-granite/granite-embedding-small-english-r2",
            max_seq=1024,
            params_m=47,
            licence="Apache-2.0",
        ),
        EncoderProfile(
            "gte-modernbert",
            "Alibaba-NLP/gte-modernbert-base",
            max_seq=1024,
            params_m=149,
            licence="Apache-2.0",
        ),
        EncoderProfile(
            "sfr-code-400m",
            "Salesforce/SFR-Embedding-Code-400M_R",
            query_prefix="Instruct: Given Code or Text, retrieval relevant content\nQuery: ",
            trust_remote_code=True,
            max_seq=1024,
            params_m=434,
            licence="CC-BY-NC-4.0",
        ),
        EncoderProfile(
            "qwen3-emb-0.6b",
            "Qwen/Qwen3-Embedding-0.6B",
            query_prefix=f"Instruct: {APPS_INSTRUCT}\nQuery:",
            max_seq=1024,
            batch_size=8,
            params_m=596,
            licence="Apache-2.0",
        ),
        EncoderProfile(
            "jina-code-0.5b",
            "jinaai/jina-code-embeddings-0.5b",
            query_prefix="Find the most relevant code snippet given the following query:\n",
            doc_prefix="Candidate code snippet:\n",
            max_seq=1024,
            batch_size=8,
            params_m=494,
            licence="CC-BY-NC-4.0",
        ),
    ]
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class EmbeddingCache:
    """Append-only on-disk store `key -> float32 vector`, one directory per (model, max_seq).

    Shards are `.npz` files (`keys`, `vecs`) written after every encoded batch group, so a long
    run can be interrupted and resumed without losing work.
    """

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, int] = {}
        self._mats: list[np.ndarray] = []
        self._rows = 0
        for shard in sorted(self.dir.glob("shard_*.npz")):
            try:
                z = np.load(shard)
                self._add(list(z["keys"]), z["vecs"])
            except Exception as e:  # a torn shard from a killed run is skipped, not fatal
                logger.warning("skipping unreadable cache shard %s: %s", shard.name, e)
        self._matrix: np.ndarray | None = None

    def _add(self, keys: list[str], vecs: np.ndarray) -> None:
        for i, k in enumerate(keys):
            self._index[str(k)] = self._rows + i
        self._mats.append(vecs.astype(np.float32))
        self._rows += len(keys)
        self._matrix = None

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def get(self, keys: list[str]) -> np.ndarray:
        if self._matrix is None:
            self._matrix = np.concatenate(self._mats) if self._mats else np.zeros((0, 1))
        return self._matrix[[self._index[k] for k in keys]]

    def put(self, keys: list[str], vecs: np.ndarray) -> None:
        n = len(list(self.dir.glob("shard_*.npz")))
        tmp = self.dir / f".tmp_{n:06d}.npz"
        np.savez(tmp, keys=np.array(keys), vecs=vecs.astype(np.float32))
        tmp.rename(self.dir / f"shard_{n:06d}.npz")
        self._add(keys, vecs)


class Encoder:
    """One embedding model on the CPU, with cached, length-sorted, resumable encoding."""

    def __init__(self, profile: EncoderProfile | str, max_seq: int | None = None) -> None:
        self.profile = PROFILES[profile] if isinstance(profile, str) else profile
        self.max_seq = max_seq or self.profile.max_seq
        self._model = None
        self.cache = EmbeddingCache(
            EMB_CACHE_DIR / f"{slug(self.profile.model_id)}__{self.max_seq}"
        )
        self.last_stats: dict = {}

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.profile.model_id, self.profile.trust_remote_code)
            self._model.max_seq_length = self.max_seq
        return self._model

    def _key(self, text: str) -> str:
        h = hashlib.sha1(f"{self.profile.model_id}\x00{self.max_seq}\x00{text}".encode())
        return h.hexdigest()

    def encode(self, texts: list[str], kind: str = "doc", group: int = 512) -> np.ndarray:
        """L2-normalised embeddings for `texts` (`kind` = "query" or "doc" picks the prefix)."""
        prefix = self.profile.query_prefix if kind == "query" else self.profile.doc_prefix
        full = [prefix + t for t in texts]
        keys = [self._key(t) for t in full]
        todo = sorted(
            {k: t for k, t in zip(keys, full, strict=True) if k not in self.cache}.items()
        )
        t0 = time.perf_counter()
        if todo:
            # Longest first inside each group keeps padding low; sentence-transformers also
            # sorts within a batch.
            todo.sort(key=lambda kv: -len(kv[1]))
            logger.info("%s: encoding %d/%d %ss", self.profile.key, len(todo), len(full), kind)
            for s in range(0, len(todo), group):
                part = todo[s : s + group]
                vecs = self.model.encode(
                    [t for _, t in part],
                    batch_size=self.profile.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                self.cache.put([k for k, _ in part], vecs)
                done = s + len(part)
                rate = done / (time.perf_counter() - t0)
                logger.info(
                    "%s: %d/%d %ss (%.1f/s, eta %.0f s)",
                    self.profile.key,
                    done,
                    len(todo),
                    kind,
                    rate,
                    (len(todo) - done) / rate,
                )
        self.last_stats = {
            "encoded": len(todo),
            "cached": len(full) - len(todo),
            "seconds": time.perf_counter() - t0,
        }
        return self.cache.get(keys)


@lru_cache(maxsize=4)
def _load_model(model_id: str, trust_remote_code: bool):
    import torch
    from sentence_transformers import SentenceTransformer

    from src.retrieval.embeddings import _patch_remote_model, detach_from_mmap

    torch.set_num_threads(config.TORCH_THREADS)
    # transformers 5 loads weights in their checkpoint dtype (fp16/bf16 for most recent models);
    # half precision on this CPU is ~12x slower than float32 (D-036), so force float32.
    kwargs = {
        "device": "cpu",
        "trust_remote_code": trust_remote_code,
        "model_kwargs": {"dtype": torch.float32},
    }
    try:
        model = SentenceTransformer(model_id, local_files_only=True, **kwargs)
    except Exception:  # not cached yet — download once into .hf_cache
        model = SentenceTransformer(model_id, **kwargs)
    detach_from_mmap(model)  # D-021: mmapped safetensors weights can SIGBUS on this Mac
    _patch_remote_model(model)
    model.eval()
    logger.info("loaded %s on %s", model_id, model.device)
    return model
