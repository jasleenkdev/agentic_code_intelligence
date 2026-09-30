"""CPU code embeddings (sentence-transformers) + exact inner-product search in numpy (D-008).

The vector matrix is small (7 758 × 384 float32 ≈ 11 MB), so brute-force search is ~1 ms and
needs no ANN library; faiss was dropped because it segfaults alongside torch on macOS (D-016)."""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from functools import lru_cache

import numpy as np

from src import config

logger = logging.getLogger(__name__)

MAX_EMBED_CHARS = 2000  # ~512 tokens of code; the model truncates beyond EMBED_MAX_SEQ anyway


def detach_from_mmap(module) -> None:
    """Copy weights out of the memory-mapped safetensors file into ordinary tensors.

    With transformers 5 on this macOS machine the cross-encoder's mmapped weights raise SIGBUS /
    produce NaN on first use (D-021); owning the memory avoids it for a few MB of RAM.
    """
    import torch

    with torch.no_grad():
        for p in module.parameters():
            p.data = p.data.clone()
        for b in module.buffers():
            b.data = b.data.clone()


# Per-model settings. "custom" models are not sentence-transformers checkpoints.
MODEL_PROFILES: dict[str, dict] = {
    "BAAI/bge-small-en-v1.5": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "nomic-ai/CodeRankEmbed": {
        "query_prefix": "Represent this query for searching relevant code: ",
    },
    "flax-sentence-embeddings/st-codesearch-distilroberta-base": {"query_prefix": ""},
    "Salesforce/codet5p-110m-embedding": {"query_prefix": "", "custom": "codet5p"},
}


def _profile(model_id: str) -> dict:
    return MODEL_PROFILES.get(model_id, {"query_prefix": ""})


def _extended_attention_mask(self, attention_mask, input_shape=None, device=None, dtype=None):
    """transformers-4 `get_extended_attention_mask`, removed in transformers 5 but still called
    by CodeRankEmbed's remote NomicBERT code (D-026)."""
    import torch

    dtype = dtype or torch.float32
    ext = (
        attention_mask[:, None, None, :]
        if attention_mask.dim() == 2
        else attention_mask[:, None, :, :]
    )
    ext = ext.to(dtype=dtype)
    return (1.0 - ext) * torch.finfo(dtype).min


def _patch_remote_model(model) -> None:
    """Add transformers-4 helpers that remote-code models still expect."""
    for mod in model.modules():
        cls = type(mod)
        if cls.__name__.startswith("NomicBert") and not hasattr(cls, "get_extended_attention_mask"):
            cls.get_extended_attention_mask = _extended_attention_mask


class _CodeT5pEncoder:
    """CodeT5+ 110M embedding model: its remote code returns normalised 256-d embeddings."""

    def __init__(self, model_id: str) -> None:
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        kw = {"trust_remote_code": True}
        try:
            cfg = AutoConfig.from_pretrained(model_id, local_files_only=True, **kw)
            local = True
        except Exception:  # noqa: BLE001 — not cached yet
            cfg = AutoConfig.from_pretrained(model_id, **kw)
            local = False
        # transformers 5 no longer defaults these on custom configs (D-026).
        for attr, val in (
            ("is_decoder", False),
            ("use_cache", False),
            ("is_encoder_decoder", False),
        ):
            if not hasattr(cfg, attr):
                setattr(cfg, attr, val)
        self.tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local, **kw)
        self.model = AutoModel.from_pretrained(model_id, config=cfg, local_files_only=local, **kw)
        self.model.eval()
        detach_from_mmap(self.model)

    def encode(self, texts, batch_size=32, show_progress_bar=False, **_):
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(
                    texts[i : i + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=config.EMBED_MAX_SEQ,
                    return_tensors="pt",
                )
                emb = self.model(**enc)
                emb = emb[0] if isinstance(emb, tuple | list) else emb
                emb = torch.nn.functional.normalize(emb, dim=-1)
                out.append(emb.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 256), dtype=np.float32)


@lru_cache(maxsize=2)
def get_model(model_id: str = ""):
    import torch

    torch.set_num_threads(config.TORCH_THREADS)
    model_id = model_id or config.EMBED_MODEL
    if _profile(model_id).get("custom") == "codet5p":
        return _CodeT5pEncoder(model_id)
    from sentence_transformers import SentenceTransformer

    try:  # prefer the local cache: no network at query time (NFR-4)
        model = SentenceTransformer(
            model_id, device="cpu", trust_remote_code=True, local_files_only=True
        )
    except Exception:  # noqa: BLE001 — not cached yet → download once
        model = SentenceTransformer(model_id, device="cpu", trust_remote_code=True)
    detach_from_mmap(model)
    _patch_remote_model(model)
    model.max_seq_length = config.EMBED_MAX_SEQ
    return model


def embed_texts(texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    model = get_model(config.EMBED_MODEL)
    if isinstance(model, _CodeT5pEncoder):
        vecs = model.encode(texts, batch_size=batch_size)
    else:
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
    return np.asarray(vecs, dtype=np.float32)


@lru_cache(maxsize=256)
def embed_query(query: str) -> np.ndarray:
    prefix = config.EMBED_QUERY_PREFIX or _profile(config.EMBED_MODEL)["query_prefix"]
    return embed_texts([prefix + query])[0]


def chunk_embedding_text(header: str, text: str, enrichment: str = "") -> str:
    return ((enrichment or header) + text)[:MAX_EMBED_CHARS]


_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def _words(ident: str) -> str:
    return " ".join(
        p.lower() for part in re.split(r"[_.$/\-]+", ident) for p in _CAMEL.findall(part)
    )


def enrichment_header(conn: sqlite3.Connection, chunk: sqlite3.Row) -> str:
    """Natural-language header for the embedder: split symbol name, path words, JSDoc and the
    names of the functions this chunk's symbol calls (D-025)."""
    sym = conn.execute(
        "SELECT s.name, s.qualname, s.kind, s.doc, f.path FROM symbols s JOIN files f ON"
        " f.id = s.file_id WHERE s.id = ?",
        (chunk["symbol_id"],),
    ).fetchone()
    if sym is None:
        return chunk["header"]
    calls = conn.execute(
        "SELECT DISTINCT callee_name FROM calls WHERE caller_id = ? AND resolution != 'external'"
        " AND line BETWEEN ? AND ? LIMIT 20",
        (chunk["symbol_id"], chunk["start_line"], chunk["end_line"]),
    ).fetchall()
    path = sym["path"]
    folder = _words("/".join(path.split("/")[:-1]))
    if sym["kind"] == "module":
        head = (
            f"Module-level code in {_words(path.rsplit('/', 1)[-1].rsplit('.', 1)[0])} ({folder})."
        )
    else:
        head = (
            f"{sym['kind'].capitalize()} {_words(sym['qualname'])} ({sym['qualname']}) in {folder}."
        )
    doc = re.sub(r"@(param|returns?|example)\b.*", "", sym["doc"] or "").strip()
    doc = re.sub(r"@description\s*", "", doc)
    parts = [head]
    if doc:
        parts.append(f"Purpose: {doc[:300]}")
    if calls:
        parts.append("Calls: " + ", ".join(_words(c["callee_name"]) for c in calls) + ".")
    return "\n".join(parts) + f"\nFile: {path}\nCode:\n"


def build_vectors(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT id, symbol_id, start_line, end_line, header, text FROM chunks ORDER BY id"
    ).fetchall()
    if config.EMBED_TEXT == "enriched":
        texts = [
            chunk_embedding_text(r["header"], r["text"], enrichment_header(conn, r)) for r in rows
        ]
    else:
        texts = [chunk_embedding_text(r["header"], r["text"]) for r in rows]
    # Sort by length so batches pad less (big CPU win); restore order afterwards.
    order = np.argsort([len(t) for t in texts])
    t0 = time.perf_counter()
    get_model(config.EMBED_MODEL)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    vecs_sorted = embed_texts([texts[i] for i in order], show_progress=True)
    encode_s = time.perf_counter() - t0
    vecs = np.empty_like(vecs_sorted)
    vecs[order] = vecs_sorted
    np.save(config.VECTORS_PATH, vecs)
    np.save(config.VECTOR_IDS_PATH, np.array([r["id"] for r in rows], dtype=np.int64))
    stats = {
        "model": config.EMBED_MODEL,
        "text_variant": config.EMBED_TEXT,
        "dim": int(vecs.shape[1]),
        "chunks": len(rows),
        "model_load_sec": round(load_s, 2),
        "encode_sec": round(encode_s, 2),
        "chunks_per_sec": round(len(rows) / max(encode_s, 1e-9), 1),
    }
    logger.info("vectors: %s", stats)
    return stats


class VectorIndex:
    def __init__(self) -> None:
        if not config.VECTORS_PATH.exists():
            raise FileNotFoundError(f"Vector index missing at {config.VECTORS_PATH}; run indexer.")
        self.matrix = np.load(config.VECTORS_PATH)
        self.chunk_ids = np.load(config.VECTOR_IDS_PATH)
        self.index_of = {int(c): i for i, c in enumerate(self.chunk_ids)}

    def search(self, query: str, n: int = 50) -> list[tuple[int, float]]:
        scores = self.matrix @ embed_query(query)
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [(int(self.chunk_ids[i]), float(scores[i])) for i in top]
