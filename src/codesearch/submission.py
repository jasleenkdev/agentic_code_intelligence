"""The submitted pipeline: `PrePostPipelineEncoder` for MTEB AppsRetrieval (R-P0-1).

The class subclasses `mteb.models.abs_encoder.AbsEncoder` exactly as the guidelines show and
implements both interfaces MTEB understands (docs/approach.md §2.1):

* `encode()` — single-vector path: pre-processed text → weighted concatenation of the
  L2-normalised embeddings of every model in the pipeline (one cosine = the weighted ensemble).
* `index()` + `search()` — MTEB's SearchProtocol. Because the class implements it, MTEB calls
  these instead of `encode()`, which lets the official run use the full multi-pass pipeline
  (pre-process → dense recall → rerank with lexical / program-style / hubness signals).

`FINAL_CONFIG` and the ablation `VARIANTS` were chosen on dev-tune only (D-035).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from src import config
from src.codesearch.encoders import PROFILES
from src.codesearch.pipeline import (
    PipelineConfig,
    build_library,
    encode_queries,
    get_encoder,
    rank,
)
from src.codesearch.preprocess import snippet_view

os.environ.setdefault("MTEB_CACHE", str(config.PROJECT_ROOT / ".cache" / "mteb"))

from mteb.models.abs_encoder import AbsEncoder  # noqa: E402
from mteb.models.model_meta import ModelMeta, ScoringFunction  # noqa: E402
from mteb.types import PromptType  # noqa: E402

logger = logging.getLogger(__name__)

# Chosen without the test split (D-035, D-040, D-041; results/apps_ablation.md):
# * base model jina-code-embeddings-0.5b (best APPS model that runs on this CPU);
# * + Qwen3-Embedding-0.6B at weight 0.4 (best ensemble weight on jina's dev-tune, where the
#   contamination favours jina alone, so the gain is a conservative estimate);
# * cleaned full query view (sample I/O kept: dropping it hurt);
# * rerank knobs tuned on the uncontaminated Qwen3 proxy, used at half weight because the best
#   weight shrinks as the base model gets stronger (proxy 0.76 vs ensemble ~0.85+ expected).
_KNOBS = {"w_style": 1.0, "w_literal": 0.2, "w_hub": 0.25, "w_bm25": 0.05}  # proxy optimum
_HALF = {k: v / 2 for k, v in _KNOBS.items()}
_FULL = (("full", 1.0),)
_RAW = (("raw", 1.0),)
_J = ("jina-code-0.5b", 1.0)
_Q = ("qwen3-emb-0.6b", 1.0)

FINAL_CONFIG = PipelineConfig(models=(_J, ("qwen3-emb-0.6b", 0.4)), query_views=_FULL, **_HALF)

# (name, what changes, config) — the ablation ladder reported in results/apps_ablation.md.
VARIANTS: list[tuple[str, str, PipelineConfig]] = [
    (
        "bge-small raw",
        "small general embedder, raw text",
        PipelineConfig(models=(("bge-small", 1.0),), query_views=_RAW),
    ),
    (
        "gte-modernbert raw",
        "149M general embedder, raw text",
        PipelineConfig(models=(("gte-modernbert", 1.0),), query_views=_RAW),
    ),
    (
        "q0 qwen3 raw",
        "proxy model (uncontaminated), raw text, cosine only",
        PipelineConfig(models=(_Q,), query_views=_RAW),
    ),
    (
        "q1 +clean",
        "q0 + LaTeX/boilerplate-cleaned query",
        PipelineConfig(models=(_Q,), query_views=_FULL),
    ),
    (
        "q1b +2 views",
        "q1 + second view without examples (not kept: +0.0001 tune, −0.004 holdout)",
        PipelineConfig(models=(_Q,), query_views=(("full", 1.0), ("no_examples", 1.0))),
    ),
    (
        "q1c no examples",
        "q1 but examples dropped from the query (hurts)",
        PipelineConfig(models=(_Q,), query_views=(("no_examples", 1.0),)),
    ),
    (
        "q1d normalised code",
        "q1 with comment/docstring-free snippets (mixed: −0.0015 with two views in the "
        "sweep; within noise, not adopted)",
        PipelineConfig(models=(_Q,), query_views=_FULL, snippet_view="normalised"),
    ),
    (
        "q2 +style",
        "q1 + program-style agreement (w 0.5)",
        PipelineConfig(models=(_Q,), query_views=_FULL, w_style=0.5),
    ),
    (
        "q3 +literal",
        "q2 + shared-literal overlap (w 0.1)",
        PipelineConfig(models=(_Q,), query_views=_FULL, w_style=0.5, w_literal=0.1),
    ),
    (
        "q4 +hubness",
        "q3 + snippet-side hubness penalty (w 0.125)",
        PipelineConfig(models=(_Q,), query_views=_FULL, w_style=0.5, w_literal=0.1, w_hub=0.125),
    ),
    (
        "q5 +bm25 = half knobs",
        "q4 + BM25 (w 0.025)",
        PipelineConfig(models=(_Q,), query_views=_FULL, **_HALF),
    ),
    (
        "q6 full knobs",
        "proxy sweep optimum (knobs ×2)",
        PipelineConfig(models=(_Q,), query_views=_FULL, **_KNOBS),
    ),
    (
        "j0 jina raw",
        "base model, raw text, cosine only (dev contaminated, D-040)",
        PipelineConfig(models=(_J,), query_views=_RAW),
    ),
    (
        "j1 jina +clean +half knobs",
        "j0 + pre/post-processing",
        PipelineConfig(models=(_J,), query_views=_FULL, **_HALF),
    ),
    (
        "e0 jina+qwen raw",
        "j0 + Qwen3 at 0.4 (ensemble via weighted concatenation)",
        PipelineConfig(models=(_J, ("qwen3-emb-0.6b", 0.4)), query_views=_RAW),
    ),
]


def _meta(cfg: PipelineConfig) -> ModelMeta:
    profiles = [PROFILES[k] for k, _ in cfg.models]
    return ModelMeta(
        loader=None,
        name="prism-hackathon/prepost-pipeline-" + "+".join(p.key for p in profiles),
        revision="1",
        release_date="2026-09-27",
        languages=["eng-Latn", "python-Code"],
        n_parameters=sum(p.params_m for p in profiles) * 1_000_000,
        memory_usage_mb=None,
        max_tokens=max(p.max_seq for p in profiles),
        embed_dim=None,
        license="apache-2.0",
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        similarity_fn_name=ScoringFunction.COSINE,
        framework=["PyTorch", "Sentence Transformers"],
        reference="https://huggingface.co/" + profiles[0].model_id,
        use_instructions=True,
        training_datasets=None,
        adapted_from=",".join(p.model_id for p in profiles),
    )


class PrePostPipelineEncoder(AbsEncoder):
    """Pre-process → multi-model dense recall → rerank; CPU only, free local models."""

    def __init__(self, cfg: PipelineConfig = FINAL_CONFIG) -> None:
        self.cfg = cfg
        self.mteb_model_meta = _meta(cfg)
        self.lib = None
        self.timings: dict[str, float] = {}

    # ---- single-vector path (used only if MTEB bypasses search, e.g. other task types) ----
    def encode(
        self,
        inputs,
        *,
        task_metadata=None,
        hf_split: str = "",
        hf_subset: str = "",
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        texts = [t for batch in inputs for t in batch["text"]]
        parts = []
        for key, w in self.cfg.models:
            if prompt_type == PromptType.query:
                v = encode_queries(texts, key, self.cfg)
            else:
                views = [snippet_view(t, self.cfg.snippet_view) for t in texts]
                v = get_encoder(key).encode(views, "doc")
            parts.append(np.sqrt(w) * v)
        out = np.concatenate(parts, axis=1)
        return out / np.linalg.norm(out, axis=1, keepdims=True).clip(1e-9)

    # ---- SearchProtocol: the multi-pass pipeline used by the official run ----
    def index(
        self,
        corpus,
        *,
        task_metadata=None,
        hf_split: str = "",
        hf_subset: str = "",
        encode_kwargs: dict | None = None,
        num_proc: int | None = None,
    ) -> None:
        import time

        t = time.perf_counter()
        ids = [str(x) for x in corpus["id"]]
        titles = corpus["title"] if "title" in corpus.column_names else [""] * len(ids)
        texts = [
            (f"{ti}\n{tx}" if ti else tx) for ti, tx in zip(titles, corpus["text"], strict=True)
        ]
        self.lib = build_library(ids, texts, self.cfg)
        self.timings["index_s"] = time.perf_counter() - t
        logger.info("indexed %d snippets in %.1f s", len(ids), self.timings["index_s"])

    def search(
        self,
        queries,
        *,
        task_metadata=None,
        hf_split: str = "",
        hf_subset: str = "",
        top_k: int = 1000,
        encode_kwargs: dict | None = None,
        top_ranked: dict | None = None,
        num_proc: int | None = None,
    ) -> dict[str, dict[str, float]]:
        import time

        if self.lib is None:
            raise RuntimeError("search() called before index()")
        t = time.perf_counter()
        qids = [str(x) for x in queries["id"]]
        texts = list(queries["text"])
        idx, sc = rank(texts, self.lib, self.cfg)
        out: dict[str, dict[str, float]] = {}
        for i, q in enumerate(qids):
            out[q] = {
                self.lib.ids[j]: float(s)
                for j, s in zip(idx[i][:top_k], sc[i][:top_k], strict=True)
            }
        self.timings["search_s"] = time.perf_counter() - t
        logger.info("searched %d queries in %.1f s", len(qids), self.timings["search_s"])
        return out
