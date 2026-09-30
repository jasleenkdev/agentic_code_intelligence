"""FastAPI layer over the existing engine (D-030). No engine logic lives here: every endpoint calls
the same functions the CLI and Streamlit app use.

Run: .venv/bin/uvicorn src.api.app:app --port 8000   (OpenAPI docs at /docs)

Ask flow (`POST /api/ask`, Server-Sent Events):
    step* (phase A: plan/search/read/assess/refine/verify/stop) → results {phase: "first"}
    step* (phase B, LLM)                                         → results {phase: "final"}
    explanation* (streamed text pieces)                          → done
An `error` event replaces the rest if something fails. The app degrades without Ollama: results
still arrive, phase B and the explanation are skipped.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, replace
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src import config

logger = logging.getLogger(__name__)

FRONTEND_DIST = config.PROJECT_ROOT / "frontend" / "dist"
MAX_CACHED_RESULTS = 64
MAX_FILE_WINDOW = 400

app = FastAPI(
    title="Code Intelligence API",
    version="1.0.0",
    description="Agentic code search over the indexed JavaScript repository: planner-driven "
    "retrieval with a streamed step trace, local speech-to-text, code viewer, domain catalog, "
    "and evaluation results. Everything runs locally on CPU.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

_results: OrderedDict[str, Any] = OrderedDict()
_engine_lock = threading.Lock()  # one question at a time: the demo is single-user, CPU-bound


# ------------------------------------------------------------------------------ helpers


_local = threading.local()


def _conn():
    """Per-thread read-only connection for the API's own queries.

    The engine's shared connection is only ever used by the worker holding `_engine_lock`;
    sharing one sqlite3 connection across FastAPI's threadpool crashed the process under the
    Explore page's parallel requests.
    """
    from src.indexer import store

    key = str(config.DB_PATH)
    if getattr(_local, "key", None) != key:
        _local.conn, _local.key = store.connect(config.DB_PATH, readonly=True), key
    return _local.conn


def _remember(res) -> str:
    rid = uuid.uuid4().hex[:12]
    _results[rid] = res
    while len(_results) > MAX_CACHED_RESULTS:
        _results.popitem(last=False)
    return rid


def _get_result(rid: str):
    res = _results.get(rid)
    if res is None:
        raise HTTPException(404, detail="Unknown or expired result id; ask the question again.")
    return res


def _result_payload(res, rid: str, phase: str, changed: bool = False) -> dict:
    d = res.to_dict()
    d.pop("steps", None)  # steps travel as their own events
    d["id"] = rid
    d["phase"] = phase
    d["changed"] = changed
    return d


@app.exception_handler(Exception)
async def _friendly_errors(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled API error")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Something went wrong on the server.",
            "detail": f"{type(exc).__name__}: {exc}",
        },
    )


# ------------------------------------------------------------------------------ status


@app.get("/api/health", tags=["status"])
def health() -> dict:
    return {"ok": True}


@app.get("/api/stats", tags=["status"], summary="Indexed repo, counts and LLM availability")
def stats() -> dict:
    from src.agent.llm import get_llm

    manifest = {}
    if config.MANIFEST_PATH.exists():
        manifest = json.loads(config.MANIFEST_PATH.read_text())
    conn = _conn()
    functions = conn.execute(
        "SELECT COUNT(*) FROM symbols WHERE kind IN ('function','method','arrow')"
    ).fetchone()[0]
    llm = get_llm()
    return {
        "repo": {
            "name": "Gladys Assistant",
            "url": "https://github.com/GladysAssistant/Gladys",
            "commit": str(manifest.get("repo_commit", ""))[:7],
        },
        "files": manifest.get("counts", {}).get("files"),
        "functions": functions,
        "chunks": manifest.get("counts", {}).get("chunks"),
        "call_edges": manifest.get("counts", {}).get("edges"),
        "catalog_rows": manifest.get("counts", {}).get("catalog"),
        "embedding": {"model": config.EMBED_MODEL, "text": config.EMBED_TEXT},
        "built_at": manifest.get("built_at"),
        "llm": {
            "model": llm.model,
            "available": llm.available(),
            "cpu_only": config.LLM_NUM_GPU == 0,
        },
    }


# ------------------------------------------------------------------------------ ask (SSE)


class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        examples=["Where is the Bluetooth-settings deeplink used?"],
    )
    use_llm: bool = True
    explain: bool | None = None  # None = config.EXPLAIN_DEFAULT (off, D-033)
    optimize: bool = False
    refine: bool = True


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _run_ask(req: AskRequest, out: queue.Queue) -> None:
    from src.agent.agent import answer, continue_with_llm, stream_explanation
    from src.agent.llm import get_llm

    def push_step(step) -> None:
        out.put(("step", asdict(step)))

    try:
        with _engine_lock:
            refine = (
                config.REFINE if req.refine else replace(config.REFINE, expand=False, rewrite=False)
            )
            res = answer(
                req.query,
                use_llm=req.use_llm,
                optimize=req.optimize,
                refine=refine,
                on_step=push_step,
            )
            rid = _remember(res)
            out.put(("results", _result_payload(res, rid, "first")))
            llm = get_llm()
            llm_up = req.use_llm and llm.available()
            if llm_up:
                changed = continue_with_llm(res, llm, on_step=push_step)
                out.put(("results", _result_payload(res, rid, "final", changed)))
            explain = config.EXPLAIN_DEFAULT if req.explain is None else req.explain
            if llm_up and explain and res.snippets:
                t0 = time.perf_counter()
                first = None
                for piece in stream_explanation(res, llm):
                    first = first if first is not None else (time.perf_counter() - t0) * 1000
                    out.put(("explanation", {"text": piece}))
                res.explanation = ""  # not needed server-side
                out.put(
                    (
                        "explanation_done",
                        {
                            "first_token_ms": round(first or 0, 1),
                            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                        },
                    )
                )
            out.put(
                (
                    "done",
                    {
                        "id": rid,
                        "llm_available": llm_up,
                        "latency_ms": res.latency_ms,
                        "final_latency_ms": res.final_latency_ms,
                    },
                )
            )
    except Exception as exc:  # noqa: BLE001 — becomes an `error` event, never a stack trace
        logger.exception("ask failed")
        out.put(
            (
                "error",
                {
                    "error": "The question could not be answered.",
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            )
        )
    finally:
        out.put(None)


@app.post(
    "/api/ask",
    tags=["ask"],
    summary="Ask a question; trace, results and explanation stream as Server-Sent Events",
    response_description="text/event-stream: step, results, explanation, "
    "explanation_done, done, error",
)
def ask(req: AskRequest) -> StreamingResponse:
    out: queue.Queue = queue.Queue()
    threading.Thread(target=_run_ask, args=(req, out), daemon=True).start()

    def events():
        yield ": stream open\n\n"
        while True:
            item = out.get()
            if item is None:
                break
            yield _sse(*item)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------------------ result extras


@app.get(
    "/api/results/{rid}/callpaths",
    tags=["results"],
    summary="Call-path graph (nodes/edges) for a result",
)
def callpaths(rid: str) -> dict:
    res = _get_result(rid)
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = {}
    for path in res.call_paths[:40]:
        labels = [p.split(" → ")[0] for p in path]
        targets = [p.split(" → ")[1] if " → " in p else None for p in path]
        for i, label in enumerate(labels):
            node = nodes.setdefault(
                label,
                {
                    "id": label,
                    "label": label,
                    "location": None,
                    "role": "root" if i == 0 else "call",
                },
            )
            if targets[i]:
                node["location"] = targets[i]
            if i:
                key = (labels[i - 1], label)
                edges.setdefault(key, {"source": key[0], "target": key[1]})
    snippet_locs = {sn.symbol: sn.location for sn in res.snippets if sn.symbol}
    for n in nodes.values():
        n["location"] = n["location"] or snippet_locs.get(n["label"])
    return {
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "route": res.route.get("type"),
    }


@app.get(
    "/api/results/{rid}/optimizations",
    tags=["results"],
    summary="Static optimisation findings for a result, optionally explained by the LLM",
)
def optimizations(rid: str, explain: bool = True) -> dict:
    from src.agent.llm import get_llm
    from src.optimize.checks import explain_findings, static_findings

    res = _get_result(rid)
    if not res.optimizations:
        res.optimizations = static_findings(res.snippets, _conn())
    explained = sum(1 for o in res.optimizations if o.get("explanation"))
    llm = get_llm()
    if explain and res.optimizations and not explained and llm.available():
        with _engine_lock:
            explained = explain_findings(res.optimizations, llm)
    return {"findings": res.optimizations, "explained": explained, "llm_available": llm.available()}


# ------------------------------------------------------------------------------ code viewer


@app.get("/api/file", tags=["code"], summary="Lines of a repo file around a range")
def file_range(
    path: str,
    start: int = Query(1, ge=1),
    end: int | None = None,
    context: int = Query(20, ge=0, le=200),
) -> dict:
    from src.agent.verifier import file_lines

    lines = file_lines(path)
    if lines is None:
        raise HTTPException(404, detail=f"File not found in the indexed repository: {path}")
    total = len(lines)
    end = end or start
    if start > total:
        raise HTTPException(416, detail=f"{path} has only {total} lines.")
    lo = max(1, start - context)
    hi = min(total, max(end, start) + context, lo + MAX_FILE_WINDOW - 1)
    return {
        "path": path,
        "total_lines": total,
        "start_line": lo,
        "end_line": hi,
        "highlight": [start, min(end, total)],
        "lines": list(lines[lo - 1 : hi]),
    }


# ------------------------------------------------------------------------------ speech


@app.post(
    "/api/transcribe",
    tags=["speech"],
    summary="Transcribe recorded audio locally (faster-whisper, CPU)",
)
async def transcribe(audio: UploadFile = File(...)) -> dict:
    data = await audio.read()
    if not data:
        raise HTTPException(422, detail="The recording was empty.")
    if len(data) > 15_000_000:
        raise HTTPException(413, detail="Recording too long (15 MB max).")
    try:
        from src.speech.stt import transcribe as stt

        text, ms = stt(data)
    except Exception as exc:  # noqa: BLE001
        logger.exception("transcription failed")
        raise HTTPException(503, detail=f"Speech-to-text failed: {exc}") from exc
    return {"text": text, "elapsed_ms": ms}


# ------------------------------------------------------------------------------ catalog


@app.get("/api/catalog/kinds", tags=["catalog"], summary="Catalog kinds with counts")
def catalog_kinds() -> list[dict]:
    rows = (
        _conn()
        .execute(
            "SELECT kind, COUNT(DISTINCT name) AS entries, SUM(role != 'definition') AS usages"
            " FROM catalog GROUP BY kind ORDER BY entries DESC"
        )
        .fetchall()
    )
    return [dict(r) for r in rows]


@app.get("/api/catalog", tags=["catalog"], summary="Browse or search catalog entries")
def catalog(
    kind: str | None = None, q: str | None = None, limit: int = Query(100, ge=1, le=500)
) -> dict:
    conn = _conn()
    params: list = []
    where = "c.role = 'definition'"
    if kind:
        where += " AND c.kind = ?"
        params.append(kind)
    if q:
        like = f"%{q.strip()}%"
        where += " AND (c.name LIKE ? OR c.value LIKE ? OR IFNULL(c.detail, '') LIKE ?)"
        params += [like, like, like]
    rows = conn.execute(
        "SELECT c.kind, c.name, MIN(c.value) AS value, MAX(c.detail) AS detail,"
        " MIN(f.path) AS file_path, MIN(c.line) AS line,"
        " (SELECT COUNT(*) FROM catalog u WHERE u.kind = c.kind AND u.name = c.name"
        "  AND u.role != 'definition') AS usages"
        f" FROM catalog c JOIN files f ON f.id = c.file_id WHERE {where}"  # noqa: S608
        " GROUP BY c.kind, c.name ORDER BY usages DESC, c.name LIMIT ?",
        [*params, limit],
    ).fetchall()
    items = [dict(r) for r in rows]
    ranked_by = "substring"
    if q and not items:  # phrase search ("bluetooth settings deeplink") via the catalog ranker
        from src.catalog.search import search_catalog

        ranked_by = "phrase"
        for h in search_catalog(conn, q, kind=kind, limit=min(limit, 20)):
            d = h.definitions[0] if h.definitions else {}
            items.append(
                {
                    "kind": h.kind,
                    "name": h.name,
                    "value": d.get("value", h.name),
                    "detail": h.detail,
                    "file_path": d.get("file_path"),
                    "line": d.get("line"),
                    "usages": len(h.usages),
                }
            )
    return {"items": items, "count": len(items), "ranked_by": ranked_by}


@app.get("/api/catalog/entry", tags=["catalog"], summary="Definition and every usage of an entry")
def catalog_entry(kind: str, name: str) -> dict:
    rows = (
        _conn()
        .execute(
            "SELECT c.role, c.value, c.detail, c.line, f.path AS file_path, s.qualname AS symbol"
            " FROM catalog c JOIN files f ON f.id = c.file_id LEFT JOIN symbols s ON"
            " s.id = c.symbol_id WHERE c.kind = ? AND c.name = ?"
            " ORDER BY c.role != 'definition', f.path, c.line",
            (kind, name),
        )
        .fetchall()
    )
    if not rows:
        raise HTTPException(404, detail=f"No catalog entry {kind} {name}")
    items = [dict(r) for r in rows]
    return {
        "kind": kind,
        "name": name,
        "definitions": [r for r in items if r["role"] == "definition"],
        "usages": [r for r in items if r["role"] != "definition"],
    }


from src.api.codesearch_routes import router as _codesearch_router  # noqa: E402
from src.api.codesearch_routes import warm_default_library  # noqa: E402

app.include_router(_codesearch_router)


@app.on_event("startup")
def _warm_codesearch() -> None:
    # Load the demo library and query encoders in the background so the first live query is fast.
    if os.environ.get("CODESEARCH_WARM", "1") != "0":
        threading.Thread(target=warm_default_library, daemon=True).start()


# ------------------------------------------------------------------------------ frontend


if FRONTEND_DIST.exists():

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        """Serve the built React app; unknown paths fall back to index.html (client routing)."""
        target = (FRONTEND_DIST / full_path).resolve()
        if full_path and FRONTEND_DIST.resolve() in target.parents and target.is_file():
            return FileResponse(target)
        if full_path.startswith("api/"):
            raise HTTPException(404, detail="Unknown API endpoint")
        return FileResponse(FRONTEND_DIST / "index.html")
