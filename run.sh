#!/usr/bin/env bash
# One-command entry points. Usage: ./run.sh {setup|submit|index-apps|index-version|search|index|app|dev|streamlit|test|e2e|eval|apps-eval|gold|ask|lint} [args]
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
GLADYS_URL="https://github.com/GladysAssistant/Gladys.git"
GLADYS_SHA="8574722f2ffb1fafc10868396cfefd38a1880586"
MODELS=("qwen2.5-coder:3b" "qwen2.5-coder:1.5b")

PORT="${PORT:-8000}"
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/ms-playwright"

need_venv() { [ -x "$PY" ] || { echo "No .venv — run ./run.sh setup first." >&2; exit 1; }; }

# Build frontend/dist when it is missing or older than any frontend source file.
build_frontend() {
  if ! command -v npm >/dev/null 2>&1; then
    [ -f frontend/dist/index.html ] && return 0
    echo "npm (Node >= 20) is required to build the web UI; ./run.sh streamlit is the fallback." >&2
    exit 1
  fi
  [ -d frontend/node_modules ] || (cd frontend && npm ci --no-audit --no-fund)
  if [ ! -f frontend/dist/index.html ] || [ -n "$(find frontend/src frontend/index.html \
      frontend/package.json frontend/vite.config.ts frontend/public -newer frontend/dist/index.html \
      -print -quit)" ]; then
    echo "Building the web UI…"
    (cd frontend && npm run build >/dev/null)
  fi
}

case "${1:-help}" in
  setup)
    PYBIN="$(command -v python3.10 || command -v python3.11 || command -v python3.12 || command -v python3 || true)"
    [ -n "$PYBIN" ] || { echo "Python >= 3.10 is required." >&2; exit 1; }
    "$PYBIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
      || { echo "Python >= 3.10 is required (found $("$PYBIN" --version))." >&2; exit 1; }
    [ -d .venv ] || "$PYBIN" -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -r requirements.txt
    if [ ! -d sample_repo/.git ]; then
      echo "Fetching target repo Gladys @ ${GLADYS_SHA:0:7} into ./sample_repo (read-only input)…"
      git init -q sample_repo
      git -C sample_repo remote add origin "$GLADYS_URL"
      git -C sample_repo fetch -q --depth 1 origin "$GLADYS_SHA"
      git -C sample_repo checkout -q FETCH_HEAD
    fi
    if command -v ollama >/dev/null 2>&1; then
      for m in "${MODELS[@]}"; do ollama list | grep -q "^$m" || ollama pull "$m"; done
    else
      echo "WARNING: Ollama not found — retrieval and structural answers work, the LLM agent" \
           "falls back to ranked snippets. Install from https://ollama.com and re-run setup."
    fi
    $PY -m scripts.prefetch_models
    if command -v npm >/dev/null 2>&1; then
      (cd frontend && npm ci --no-audit --no-fund)
      build_frontend
    else
      echo "WARNING: npm not found — the React UI cannot be built; ./run.sh streamlit still works."
    fi
    echo "Setup complete. Next: ./run.sh index"
    ;;
  submit)
    need_venv; shift; $PY -m scripts.submit "$@"
    # Supporting results for the GitHub Release (tuning-split ablations, model benchmark, …).
    for f in results/apps_ablation.md results/apps_ablation.json results/apps_model_benchmark.md \
             results/apps_model_benchmark.json results/apps_test_baselines.md; do
      [ -f "$f" ] && cp "$f" release/
    done
    ;;
  index-apps)
    # Demo library: the APPS corpus as v1 plus two synthetic evolutions (v2, v3) for P1/bonus.
    need_venv; shift
    $PY -m scripts.index_version apps --apps-corpus --version v1 "$@"
    $PY -m scripts.index_version apps --evolve-from v1 --version v2 --rate 0.10 "$@"
    $PY -m scripts.index_version apps --evolve-from v2 --version v3 --rate 0.05 "$@"
    ;;
  index-version)
    # ./run.sh index-version gladys --ref v5.1.1   (codebase from data/versions/gladys.git)
    # ./run.sh index-version <library> --jsonl file.jsonl --version <label>
    need_venv; shift
    lib="${1:?library name}"; shift
    if [ "$lib" = "gladys" ] && [[ " $* " != *" --git "* ]]; then
      [ -d data/versions/gladys.git ] || git clone -q --bare "$GLADYS_URL" data/versions/gladys.git
      $PY -m scripts.index_version gladys --git data/versions/gladys.git "$@"
    else
      $PY -m scripts.index_version "$lib" "$@"
    fi
    ;;
  search) need_venv; shift; $PY -m scripts.search "$@" ;;
  apps-eval)
    # Tuning-split evaluations (never the test split): model benchmark, ablations,
    # evolutionary-retrieval benchmark, P1 rebuild timings.
    need_venv; shift
    $PY -m eval.apps_ablation "$@"
    $PY -m eval.evolution_bench
    $PY -m eval.p1_rebuild
    ;;
  index) need_venv; shift; $PY -m scripts.index "$@" ;;
  app)
    need_venv; shift; build_frontend
    echo "Code Intelligence → http://localhost:$PORT  (API docs: http://localhost:$PORT/docs)"
    exec .venv/bin/uvicorn src.api.app:app --host 127.0.0.1 --port "$PORT" "$@"
    ;;
  dev)
    need_venv; shift
    [ -d frontend/node_modules ] || (cd frontend && npm ci --no-audit --no-fund)
    .venv/bin/uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload --reload-dir src &
    API_PID=$!
    trap 'kill $API_PID 2>/dev/null' EXIT INT TERM
    (cd frontend && npm run dev)
    ;;
  streamlit) need_venv; shift; exec .venv/bin/streamlit run ui/app.py --server.headless true "$@" ;;
  test)  need_venv; shift; .venv/bin/pytest "$@" ;;
  e2e)
    need_venv; shift; build_frontend
    [ -d "$PLAYWRIGHT_BROWSERS_PATH" ] && ls "$PLAYWRIGHT_BROWSERS_PATH" | grep -q chromium \
      || .venv/bin/playwright install chromium
    .venv/bin/pytest tests/e2e -m e2e "$@"
    ;;
  eval)
    need_venv; shift
    [ -f eval/gold.json ] || $PY -m eval.generate_gold
    $PY -m eval.run_eval "$@"
    ;;
  gold)  need_venv; shift; $PY -m eval.generate_gold "$@" ;;
  ask)   need_venv; shift; $PY -m scripts.ask "$@" ;;
  lint)
    need_venv
    .venv/bin/ruff format src scripts tests eval ui
    .venv/bin/ruff check src scripts tests eval ui
    [ ! -d frontend/node_modules ] || (cd frontend && npx oxlint src && npx tsc -b)
    ;;
  *)
    echo "Usage: ./run.sh {setup|submit|index-apps|index-version|search|index|app|dev|streamlit|test|e2e|eval|apps-eval|gold|ask|lint} [args]"
    echo "  setup      create .venv, install deps, fetch sample_repo, pull Ollama models, prefetch"
    echo "             models, install and build the web UI"
    echo "  submit     official MTEB AppsRetrieval run (test split) → release/appsretrieval_results.json"
    echo "  index-apps index the APPS corpus as library 'apps' (v1 + synthetic v2, v3)"
    echo "  index-version LIB …  index one version, e.g. gladys --ref v5.1.1 (incremental)"
    echo "  search Q   rank snippets from the terminal (--library apps|gladys --version LABEL|all)"
    echo "  apps-eval  tuning-split ablations, evolutionary-retrieval benchmark, P1 timings"
    echo "  index      build the index (structure, catalog, chunks, BM25, vectors)"
    echo "  app        API + web UI on http://localhost:$PORT (set PORT to change; rebuilds the UI if stale)"
    echo "  dev        API with reload on :8000 + Vite dev server on http://localhost:5173"
    echo "  streamlit  fallback Streamlit UI on http://localhost:8501"
    echo "  test       run the test suite (unit + API; excludes e2e)"
    echo "  e2e        Playwright browser test of the three demo questions (real index)"
    echo "  eval       run the benchmark (add --systems / --explain / --use-llm-cache / --merge)"
    echo "  ask        ask one question from the terminal, e.g. ./run.sh ask \"who calls purgeStates?\""
    ;;
esac
