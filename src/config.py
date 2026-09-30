"""Central configuration. Every tunable lives here and is read from the environment (.env).

Ranking and budget parameters are recorded with every evaluation run (FR-6), so keep them here
rather than scattered as literals.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Keep every model download inside the project (never write to ~/.cache).
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "")
    return value if value.strip() else default


REPO_ROOT = Path(_env("CODEINTEL_REPO", str(PROJECT_ROOT / "sample_repo"))).resolve()
INDEX_DIR = Path(_env("CODEINTEL_INDEX_DIR", str(PROJECT_ROOT / "index"))).resolve()
DB_PATH = INDEX_DIR / "code.db"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
BM25_PATH = INDEX_DIR / "bm25.pkl"
# One vector file per (embedding model, chunk-text variant) so alternatives can coexist.
_VEC_SLUG = (
    re.sub(r"[^a-z0-9]+", "-", _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5").split("/")[-1].lower())
    + "__"
    + _env("EMBED_TEXT", "enriched")
)
VECTORS_PATH = INDEX_DIR / f"vectors__{_VEC_SLUG}.npy"
VECTOR_IDS_PATH = INDEX_DIR / f"vector_ids__{_VEC_SLUG}.npy"
LLM_CACHE_DIR = PROJECT_ROOT / ".cache" / "llm"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---- Indexing scope (relative to REPO_ROOT) ----
INCLUDE_DIRS: tuple[str, ...] = tuple(
    p.strip() for p in _env("CODEINTEL_INCLUDE", "server,front/src").split(",") if p.strip()
)
EXCLUDE_DIR_NAMES: frozenset[str] = frozenset(
    {
        "node_modules",
        "dist",
        "build",
        "vendor",
        "coverage",
        ".git",
        "test",
        "tests",
        "__tests__",
        "cypress",
        "migrations",
        "seeders",
    }
)
JS_EXTENSIONS: frozenset[str] = frozenset({".js", ".jsx", ".mjs", ".cjs"})
MAX_FILE_BYTES = 400_000
MINIFIED_AVG_LINE_LEN = 300  # average line length above this => treated as minified
MAX_CHUNK_LINES = 120  # longer functions are split into windows for retrieval
CHUNK_WINDOW_OVERLAP = 10

# ---- Models ----
EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Empty = use the model's own query prefix from retrieval.embeddings.MODEL_PROFILES.
EMBED_QUERY_PREFIX = os.environ.get("EMBED_QUERY_PREFIX", "")
# Chunk text fed to the embedder: "plain" = header + code; "enriched" = natural-language header
# (split symbol name, path words, JSDoc, called function names) + code (D-025, default).
EMBED_TEXT = _env("EMBED_TEXT", "enriched")
EMBED_MAX_SEQ = int(_env("EMBED_MAX_SEQ", "512"))
# Embedding model for versioned codebase libraries (Gladys), see src/codesearch/versions.py.
CODESEARCH_CODEBASE_MODEL = _env("CODESEARCH_CODEBASE_MODEL", "gte-modernbert")
RERANK_MODEL = _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
WHISPER_MODEL = _env("WHISPER_MODEL", "base.en")
TORCH_THREADS = int(_env("TORCH_THREADS", str(os.cpu_count() or 4)))

# ---- LLM (OpenAI-compatible; local Ollama by default, no key needed) ----
LLM_BASE_URL = _env("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = _env("LLM_API_KEY", "ollama")
LLM_MODEL = _env("LLM_MODEL", "qwen2.5-coder:3b")
LLM_FALLBACK_MODEL = _env("LLM_FALLBACK_MODEL", "qwen2.5-coder:1.5b")
# Ollama-only option: 0 forces CPU inference (the hackathon constraint). -1 = let Ollama decide.
LLM_NUM_GPU = int(_env("LLM_NUM_GPU", "0"))
LLM_TIMEOUT_SEC = float(_env("LLM_TIMEOUT_SEC", "120"))
LLM_MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "400"))
LLM_NUM_CTX = int(_env("LLM_NUM_CTX", "8192"))
# Generation is out of scope under the v2 guidelines (D-033): the streamed LLM explanation of
# Gladys answers is kept but OFF unless CODEINTEL_EXPLAIN=1 (or the UI/CLI/API asks for it).
EXPLAIN_DEFAULT = _env("CODEINTEL_EXPLAIN", "0").lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RankingConfig:
    """All ranking knobs in one place (FR-6). Recorded with every evaluation run."""

    candidates_per_retriever: int = 50
    rrf_k: int = 60
    use_bm25: bool = True
    use_vectors: bool = True
    use_rerank: bool = False
    rerank_top_n: int = 30
    use_graph_boost: bool = False
    graph_boost: float = 0.3  # added RRF mass for symbol-name / structural hits
    default_k: int = 10
    # Fusion of the BM25 and vector lists (D-024): "rrf" = weighted reciprocal rank fusion,
    # "score" = weighted sum of min-max normalised scores.
    fusion: str = "rrf"
    w_bm25: float = 1.0
    w_vector: float = 1.0
    # Vector weight used instead of w_vector when the query contains code identifiers
    # (camelCase, snake_case, dotted, `name()`); None = no query adaptation.
    ident_w_vector: float | None = None
    # BM25 weight used instead of w_bm25 when the query has NO code identifiers (plain English);
    # None = no adaptation.
    nl_w_bm25: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AgentBudget:
    """Limits for the answer pipeline (docs/approach.md, D-027)."""

    refine_rounds: int = 2  # max refinement rounds (1 = graph/catalog expansion, 2 = rewrite)
    explanation_max_tokens: int = 220

    def to_dict(self) -> dict:
        return asdict(self)


# Hybrid default (D-024): min-max score fusion; BM25 weight 0.5 for plain-English queries,
# vector weight 0.25 for queries containing code identifiers. Chosen on original + half of the
# paraphrased set; the other half was held out (results/fusion_sweep_*.md, docs/approach.md §A.4).
RANKING = RankingConfig(fusion="score", ident_w_vector=0.25, nl_w_bm25=0.5)
# The agent's retriever: the hybrid above + the graph/catalog list.
AGENT_RANKING = RankingConfig(
    fusion="score", ident_w_vector=0.25, nl_w_bm25=0.5, use_graph_boost=True
)
BUDGET = AgentBudget()


def _agent_defaults():
    from src.agent.expand import ExpandConfig
    from src.agent.refine import RefineConfig

    # Calibrated on original + reworded-tuning by eval/calibrate_agent.py (D-028, D-029):
    # graph/catalog expansion with λ = 0.1 from the top 3 hits is MRR-neutral and stays on
    # (agentic refinement round 1); every constrained-rewrite variant lowered MRR, so round 2
    # is implemented but off (rewrite=False).
    return RefineConfig(gate="low", expand=True, rewrite=False), ExpandConfig(lam=0.1, seeds=3)


REFINE, EXPAND = _agent_defaults()
