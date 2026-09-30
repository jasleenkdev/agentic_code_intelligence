"""HTTP routes for the snippet-library search engine (`/api/cs/...`, R-D-1). Retrieval only."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src import config

router = APIRouter(prefix="/api/cs", tags=["code search"])
_lock = threading.Lock()  # CPU-bound: one search at a time
RELEASE_DIR = config.PROJECT_ROOT / "release"


def warm_default_library() -> None:
    """Background warm-up at API start: load the newest APPS (or first) library and its encoders."""
    import logging

    from src.codesearch.engine import get_engine

    try:
        eng = get_engine()
        libs = eng.libraries_map()
        name = "apps" if "apps" in libs else next(iter(libs), None)
        if name:
            with _lock:
                eng.warm(name, libs[name][-1])
    except Exception as e:  # warm-up is best effort; searches still work (just slower first)
        logging.getLogger(__name__).warning("code-search warm-up failed: %s", e)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=20_000)
    library: str = "apps"
    version: str | None = None  # a version label, "all", or None = latest
    k: int = Field(10, ge=1, le=50)


@router.get("/libraries", summary="Indexed snippet libraries, their versions and models")
def libraries() -> list[dict]:
    from src.codesearch.engine import get_engine

    return get_engine().libraries()


@router.get("/examples", summary="Example queries per library")
def examples() -> dict:
    from src.codesearch.examples import EXAMPLES

    return EXAMPLES


@router.post("/search", summary="Rank a library's snippets for a query (multi-pass)")
def search(req: SearchRequest) -> dict:
    from src.codesearch.engine import get_engine

    eng = get_engine()
    known = eng.libraries_map()
    if req.library not in known:
        raise HTTPException(404, detail=f"Library '{req.library}' is not indexed")
    if req.version not in (None, "all", *known[req.library]):
        raise HTTPException(404, detail=f"Version '{req.version}' of '{req.library}' not indexed")
    with _lock:
        return eng.search(req.query, req.library, req.version, k=req.k)


def _read(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


@router.get(
    "/eval", summary="APPS, P1 and evolutionary-retrieval results from results/ and release/"
)
def eval_results() -> dict:
    r = config.RESULTS_DIR
    official = _read(RELEASE_DIR / "appsretrieval_results.json")
    return {
        "official": _official_summary(official),
        "benchmark": _read(r / "apps_model_benchmark.json"),
        "ablation": _read(r / "apps_ablation.json"),
        "baselines": _read(r / "apps_test_baselines.json"),
        "p1": _read(r / "p1_rebuild.json"),
        "evolution": _read(r / "evolution_bench.json"),
        "run_info": _read(RELEASE_DIR / "run_info.json"),
    }


def _official_summary(res: dict | None) -> dict | None:
    if not res:
        return None
    try:
        scores = res["scores"]["test"][0]
    except (KeyError, IndexError, TypeError):
        return {"raw": res}
    keep = ("ndcg_at_10", "mrr_at_10", "ndcg_at_1", "recall_at_10", "recall_at_100", "map_at_10")
    return {k: scores.get(k) for k in keep} | {"main_score": scores.get("main_score")}
