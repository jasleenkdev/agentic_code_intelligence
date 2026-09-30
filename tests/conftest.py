"""Shared fixtures: a tiny JS repo indexed into a temporary directory."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# The API warms the code-search library at startup; unit tests must not load models for that.
os.environ.setdefault("CODESEARCH_WARM", "0")

from src import config

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "mini_repo"


@pytest.fixture(scope="session")
def mini_index(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Index the fixture repo (structure + catalog + chunks + bm25) into a temp dir."""
    from src.indexer.pipeline import build_index

    index_dir = tmp_path_factory.mktemp("index")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "INDEX_DIR", index_dir)
    mp.setattr(config, "DB_PATH", index_dir / "code.db")
    mp.setattr(config, "MANIFEST_PATH", index_dir / "manifest.json")
    mp.setattr(config, "BM25_PATH", index_dir / "bm25.pkl")
    mp.setattr(config, "VECTORS_PATH", index_dir / "vectors.npy")
    mp.setattr(config, "VECTOR_IDS_PATH", index_dir / "vector_ids.npy")
    mp.setattr(config, "REPO_ROOT", FIXTURE_REPO)
    stages = tuple(s for s in ("structure", "catalog", "chunks", "bm25") if s in _available())
    build_index(stages, repo_root=FIXTURE_REPO)
    yield index_dir
    mp.undo()


def _available() -> set[str]:
    import importlib.util

    ok = {"structure"}
    for stage, mod in (
        ("catalog", "src.catalog.builder"),
        ("chunks", "src.retrieval.chunker"),
        ("bm25", "src.retrieval.bm25"),
    ):
        if importlib.util.find_spec(mod) is not None:
            ok.add(stage)
    return ok


@pytest.fixture()
def mini_db(mini_index: Path):
    from src.indexer import store

    conn = store.connect(mini_index / "code.db", readonly=True)
    yield conn
    conn.close()
