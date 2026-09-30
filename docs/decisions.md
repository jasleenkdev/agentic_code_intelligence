# Design decisions

Every significant design choice in this project, numbered in the order it was made. Each entry
states the decision, the alternatives considered and the reason. Code comments, docstrings and
the files in `results/` cite these numbers (`D-0xx`). Where a later decision replaced an earlier
one, the heading says so; the earlier entry is kept so the history of the design stays readable.

- **D-001 … D-032** cover the first-round prototype, a code-intelligence agent over the Gladys
  Assistant JavaScript codebase, which remains in the repository as a secondary demo.
- **D-033 … D-042** cover the snippet-library retrieval engine evaluated on CoIR APPS, versioned
  libraries (P1) and all-versions retrieval (bonus).

The overall design is described in [`approach.md`](approach.md).

---

## D-001 · Documentation-first design
*2026-09-23*

- **Decision:** requirements and architecture were written down before any application code, and
  every change to a component's design, schema, tool signature, library or data flow was
  documented alongside the change, together with a numbered entry in this log.
- **Alternatives:** write code first and document at the end; a single README.
- **Reason:** in a fixed-length hackathon, keeping the design and its rationale written down as it
  changes keeps everyone working on the project aligned and makes the submission self-explaining.
  The design documents are consolidated for the submission into `docs/approach.md` and this file.

## D-002 · Structural queries answered deterministically, not by the LLM
*2026-09-23*

- **Decision:** call-order, caller, callee and usage queries are computed from the AST and call
  graph. The LLM may phrase the answer; it does not supply the facts.
- **Alternatives:** let the LLM infer structure from retrieved snippets.
- **Reason:** correctness is checkable and demoable; LLM-inferred structure is not.

## D-003 · Hybrid retrieval fused with RRF (default superseded by D-024)
*2026-09-23*

- **Decision:** BM25 and vector search fused by Reciprocal Rank Fusion; cross-encoder rerank
  optional.
- **Alternatives:** vector-only; BM25-only; learned weighted fusion.
- **Reason:** identifier-heavy code queries favour lexical retrieval and intent-heavy ones favour
  semantic retrieval; RRF needs no score normalisation. The alternatives serve as ablation
  baselines.

## D-004 · Function-level chunking with a context header
*2026-09-23*

- **Decision:** chunks are whole functions/methods prefixed with file path, enclosing class and
  JSDoc; functions longer than 120 lines are split into overlapping windows.
- **Alternatives:** fixed-size sliding windows; file-level chunks.
- **Reason:** functions are the natural unit of retrieval and citation and map 1:1 to symbols.

## D-005 · Hard citation verification before answering
*2026-09-23*

- **Decision:** every cited `file:line` is re-read from disk and checked; failures are dropped.
- **Alternatives:** trust the model's citations; verify a sample.
- **Reason:** citation accuracy was a hard requirement (every cited range must exist and contain
  what is claimed), not a target.

## D-006 · JavaScript parser: tree-sitter, grammar pinned to 0.23.1
*2026-09-23*

- **Decision:** `tree-sitter` 0.26 Python bindings with `tree-sitter-javascript==0.23.1`.
- **Alternatives:** tree-sitter-javascript 0.25.0 (latest); Babel/Acorn via a Node subprocess;
  esprima-python.
- **Reason:** in-process, fast (2 186 files parsed and extracted in 2.7 s) and error-tolerant. The
  latest grammar 0.25.0 mis-parses Preact's `class="…"` JSX attribute (593 of 2 293 candidate files
  had ERROR nodes); 0.23.1 parses all of them with zero errors. A Node parser would add a process
  hop for no fidelity gain on this repository.

## D-007 · Lexical index: rank-bm25 with identifier splitting
*2026-09-23*

- **Decision:** `rank_bm25.BM25Okapi` over chunk tokens, pickled to `index/bm25.pkl`; tokens are
  camelCase/snake_case-split identifiers plus the original identifier.
- **Alternatives:** Tantivy; SQLite FTS5.
- **Reason:** ~15k chunks fit trivially in memory; identifier splitting is easiest to control in
  Python; no extra native dependencies.

## D-008 · Embedding model and vector store (store superseded by D-018)
*2026-09-23*

- **Decision:** `BAAI/bge-small-en-v1.5` (33M params, 384-d) on CPU via sentence-transformers;
  FAISS `IndexFlatIP` over L2-normalised vectors.
- **Alternatives (model):** `jinaai/jina-embeddings-v2-base-code` (fails to load with
  transformers 5.x: `find_pruneable_heads_and_indices` removed); `nomic-ai/CodeRankEmbed` (137M,
  code-specific; benchmarked in D-016). **(storage):** Chroma, LanceDB, sqlite-vec.
- **Reason:** exact search over ~15k × 384 floats takes < 5 ms, so approximate-nearest-neighbour
  structures buy nothing; FAISS flat is one file and one dependency.

## D-009 · Local LLM via an OpenAI-compatible client (hand-written loop superseded by D-027/D-028)
*2026-09-23*

- **Decision:** no paid APIs anywhere. The LLM layer is a small OpenAI-compatible chat client
  (`/v1/chat/completions`) pointed at local Ollama by default, with no key. Default model
  `qwen2.5-coder:3b`, fallback `qwen2.5-coder:1.5b`; any OpenAI-compatible free tier (Groq,
  Gemini's OpenAI endpoint) can be enabled via `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` in
  `.env`. No agent framework: a hand-written loop gives full control of budgets, JSON validation
  and the trace.
- **Alternatives:** a paid hosted LLM API (ruled out by the free/local-only constraint);
  `llama3.2:3b` (9.5 tok/s on CPU, weaker at code); `qwen2.5-coder:7b` (better, but ~2× slower on
  CPU); LangChain/LlamaIndex agents.
- **Reason:** measured on the demo machine with `num_gpu=0`: qwen2.5-coder:3b produced valid JSON
  tool calls at 12 tok/s; llama3.2:3b 9.5 tok/s. The deterministic paths carry the facts (D-002),
  so the LLM only plans and phrases; a 3B coder model is sufficient and keeps latency tolerable.

## D-010 · Index storage: SQLite
*2026-09-23*

- **Decision:** one SQLite file `index/code.db` with secondary indexes on names, callers and
  targets.
- **Alternatives:** JSONL; DuckDB.
- **Reason:** `get_callers` / `find_call_sequence` need indexed point lookups and joins; SQLite is
  in the standard library, single-file, and builds in < 4 s for this repository.

## D-011 · Target codebase undecided (superseded by D-012)
*2026-09-23*

- **Decision:** the target-codebase profile was left open until a repository was selected.
- **Alternatives:** —
- **Reason:** no sample repository was supplied by the organisers; resolved by D-012.

## D-012 · Target codebase: Gladys Assistant
*2026-09-23*

- **Decision:** clone `GladysAssistant/Gladys` @ `8574722` into `./sample_repo`; index `server/`
  and `front/src/` excluding tests, migrations and seeders.
- **Alternatives (all cloned and profiled):**

  | Repo | JS files / LOC | TS files / LOC | Last commit | Verdict |
  |---|---|---|---|---|
  | livekit/agents-js | 4 / 159 | 707 / 196k | 2026-09-22 | ~100 % TypeScript — rejected |
  | HeyHeyChicken/NOVA-NodeJS | 1 / 356 | 0 | 2024-01-02 | far too small — rejected |
  | leon-ai/leon | 106 / 15k | 616 / 111k | 2026-09-22 | mostly TypeScript — rejected |
  | axa-group/nlp.js | 905 / 231k | 0 | 2024-03-21 | NLP library, not an assistant; stale |
  | **GladysAssistant/Gladys** | **3 286 / 392k** (2 186 / 213k indexed) | **0** | **2026-09-22** | **chosen** |

- **Reason:** 100 % JavaScript, large, actively maintained, and it has exactly the structures the
  brief asked about: registered intents, an event bus, 39 service integrations (incl. Bluetooth),
  MCP tools for LLMs, and front-end routes that act as deeplinks
  (`/dashboard/integration/device/bluetooth`). Tests are excluded from the default scope because
  they duplicate every symbol name and would pollute "who calls X" answers.

## D-013 · Everything local, CPU-only, models inside the project
*2026-09-23*

- **Decision:** `HF_HOME=./.hf_cache`, Ollama with `num_gpu=0`, all Python dependencies in
  `./.venv`.
- **Alternatives:** the default `~/.cache`; letting Ollama use Metal (28 tok/s vs 12 tok/s).
- **Reason:** the hackathon constraint is CPU-only and the project must not write outside its
  folder. `LLM_NUM_GPU=-1` re-enables the GPU for anyone who prefers speed over the constraint.

## D-014 · Extra call-site facts: branch / callback / loop / await flags
*2026-09-23*

- **Decision:** each call row records `in_branch`, `in_callback`, `in_loop`, `awaited` and
  `first_arg`.
- **Alternatives:** ordinal only (the original schema).
- **Reason:** `find_call_sequence` must label pairs `certain` vs `possible`; the optimiser needs
  awaits and loops; event linking needs the first argument. All are cheap to capture during the
  single AST pass.

## D-015 · Domain catalog as a first-class, deterministic index
*2026-09-23*

- **Decision:** a `catalog` table of domain entities (intents, events, MCP tools, services, API
  routes, front routes/deeplinks, URIs) with definition and usage rows, and a rule-based phrase
  search.
- **Alternatives:** rely on `find_string_usages` plus semantic search to find e.g. "the Bluetooth
  settings deeplink"; ask the LLM to map the phrase to a string.
- **Reason:** usage questions are about *domain* names that never appear verbatim in the code
  ("Bluetooth settings" → `/dashboard/…/bluetooth/config` rendered by `BluetoothSettingsPage`).
  A catalog answers them exactly, with no LLM, in milliseconds; route-pattern matching links
  templated usages (`/…/${id}`) to their definition, which grep cannot.

## D-016 · bge-small on CPU; import faiss after torch (import-order fix superseded by D-018)
*2026-09-23*

- **Decision:** keep `BAAI/bge-small-en-v1.5` as the default embedder (7 758 chunks in 144 s,
  54 chunks/s, 384-d, 11 MB of vectors). `faiss` is imported only after `torch`
  (`embeddings._faiss()`); models load with `local_files_only=True` first so query time needs no
  network.
- **Alternatives:** `nomic-ai/CodeRankEmbed` (code-specific, 137M params, ~4× slower on CPU;
  crashed with the same OpenMP issue before this fix; remains selectable via `EMBED_MODEL`);
  jina-v2-base-code (incompatible with transformers 5.x).
- **Reason:** importing faiss before torch loads two OpenMP runtimes on macOS arm64 and the
  process segfaults (exit 139) at the first encode — reproduced and fixed by import order.
  bge-small is fast enough to re-index the whole repository in under 3 minutes on CPU and ranked
  the right Bluetooth / purge functions first in smoke tests.

## D-017 · Agent shaped around a weak, slow CPU model (superseded by D-027)
*2026-09-23*

- **Decision:** deterministic routing and handlers carry all structural/usage facts; the semantic
  loop starts from a deterministic seed search; the LLM speaks a constrained JSON action format
  validated per turn with bounded retries; hard step/read/wall-clock budgets; a non-LLM fallback;
  citations are snapped to tool-returned evidence and re-read from disk.
- **Alternatives:** the model's native tool-calling API; a ReAct text format; letting the model
  answer structural questions from snippets.
- **Reason:** measured on the demo machine, qwen2.5-coder:3b on CPU evaluates prompts at
  ~60 tok/s (a 2 000-token prompt took 33 s) and generates ~10 tok/s; 7B timed out at 60 s on the
  same prompt. Small prompts, few steps and a zero-LLM path are the only way to stay interactive.
  JSON mode (`format: json`) made every observed reply parseable; validation still guards the
  schema (unknown tools, missing arguments).

## D-018 · Replace FAISS with exact numpy search
*2026-09-23*

- **Decision:** store vectors as a numpy file (since D-025: `index/vectors__<model>__<text>.npy`);
  search is one matrix–vector product.
- **Alternatives:** keep FAISS with torch-first import order (D-016); LanceDB; sqlite-vec.
- **Reason:** the import-order fix was not sufficient — the process still segfaulted (exit 139)
  when BM25 ran before the first vector search. At 7 758 × 384 brute-force search takes ~1 ms, so
  an ANN library adds risk and no speed. Removes the `faiss-cpu` dependency.

## D-019 · Streamlit UI with local faster-whisper voice input (API part superseded by D-030; Streamlit is now the fallback UI)
*2026-09-23*

- **Decision:** Streamlit (`st.audio_input` for the microphone, `st.status` for the live trace,
  `st.graphviz_chart` for call paths); speech via faster-whisper `base.en` int8 on CPU.
- **Alternatives:** Gradio; a FastAPI + custom JS front end; the browser Web Speech API (sends
  audio to a cloud service in Chrome); whisper `tiny.en` (faster, less accurate) or `small.en`
  (slower).
- **Reason:** Streamlit gives microphone capture, streaming status and graph rendering with no
  front-end build; faster-whisper keeps audio on the machine and `base.en` int8 transcribes a
  short question in well under real time on the M1 CPU. The separate HTTP API planned earlier was
  dropped at this point: the UI imports the pipeline directly, and `scripts/ask.py` covers
  scripting.

## D-020 · Evaluation metric definitions (Gladys benchmark)
*2026-09-23*

- **Decision:** relevance = same file + overlapping line range; precision@k divides by
  min(k, returned); recall@k over gold locations; MRR over the first relevant result.
- **Alternatives:** exact line equality; precision@k ÷ k.
- **Reason:** systems return function-level ranges while gold may be a single call line or a whole
  function, so overlap is the fair unit. Dividing by k would penalise deterministic answers for
  returning a short, exact list; recall@k already captures completeness.

## D-021 · Copy model weights out of the safetensors mmap
*2026-09-23*

- **Decision:** after loading, every sentence-transformers / cross-encoder module's parameters and
  buffers are cloned into ordinary tensors (`embeddings.detach_from_mmap`).
- **Alternatives:** pin an older transformers release; convert the checkpoint to `.bin`.
- **Reason:** with transformers 5.17 on the demo machine, the cross-encoder's memory-mapped
  weights produced NaN scores inside sentence-transformers and a SIGBUS (exit 138) in a plain
  transformers forward pass, although the file's SHA-256 matched. The first evaluation run
  therefore showed `hybrid_rerank` identical to `hybrid` (NaN scores leave the order unchanged);
  after the fix the reranker returns correct scores (8.7 vs −11.1 on a sanity pair) and the rerank
  ablation was re-run. Cloning costs ~90 MB of RAM for the reranker and ~130 MB for bge-small.

## D-022 · Drop copy-pasted JSDoc from semantic gold candidates
*2026-09-23*

- **Decision:** `eval/generate_gold.py` rejects a JSDoc that mentions "<other service> device /
  service / integration" and never names the service directory it lives in.
- **Alternatives:** hand-edit the one bad label; keep it and document it.
- **Reason:** the benchmark must stay free of manual labelling, so the fix is a rule, not an edit.
  Over every `server/services/**` JSDoc the rule fires exactly once (the known bad label).

## D-023 · Independent text-based checker for structural labels
*2026-09-23*

- **Decision:** `eval/structural_checker.py` recomputes call-order, caller and callee answers
  without tree-sitter, SQLite or the call graph, and reports overlap agreement with the index.
- **Alternatives:** a Babel/Acorn parser (a second AST — less independent in method, adds Node);
  hand-checking a sample (manual labelling, excluded by the brief).
- **Reason:** structural labels were derived from the same call graph the system queries. A
  deliberately different method (regex + brace matching) gives an independent estimate of label
  quality. Checker bugs found while building it (destructured parameters, regex literals,
  expression-bodied arrows) were fixed generically; the remaining disagreements come from
  limitations of name-only resolution and are reported, not tuned away.

## D-024a · Paraphrased, lexically disjoint benchmark queries
*2026-09-23*

- **Decision:** 20 LLM-written questions validated programmatically for zero identifier-stem and
  zero doc-bigram overlap with their target, plus an LLM fidelity judge; targets and labels stay
  programmatic. Used as a held-out set: fusion weights and the refinement threshold are tuned on
  the original set only.
- **Alternatives:** hand-written paraphrases (manual labelling, excluded); synonym substitution
  without an LLM (produced ungrammatical questions); an embedding-similarity filter for fidelity
  (would bias the set toward the embedders being compared).
- **Reason:** the original semantic queries copy JSDoc wording, which favours BM25.
  qwen2.5-coder:3b accepted 0/60 under the rules (it could not avoid forbidden words); a focused
  "replace only these words" rewrite loop fixed acceptance but drifted in meaning, so generation
  uses the 7B model offline with the judge. Runtime inference stays on the 3B CPU model.

## D-024 · Query-adaptive min-max score fusion replaces equal-weight RRF
*2026-09-23*

- **Decision:** fuse normalised BM25 and vector scores; vector weight 0.25 for queries with code
  identifiers, BM25 weight 0.5 for queries without; everything else 1.0.
- **Alternatives:** weighted RRF (w_vec 0.25–1.0); plain weighted score fusion; identifier-only
  adaptation (all compared with `eval/sweep_fusion.py`).
- **Reason:** diagnosis showed vectors fail on identifier questions and BM25 fails on paraphrased
  ones; RRF cannot express "this list is useless for this query" and never beat BM25. Weights were
  picked from a small fixed grid on the original set plus half of the paraphrased set; the other
  half was held out (BM25 0.000 → fused 0.100).

## D-025 · Enriched chunk text for the embedder
*2026-09-23*

- **Decision:** embed a natural-language header (split symbol name, folder words, JSDoc purpose,
  split callee names) plus code instead of the raw `// file:` header plus code.
- **Alternatives:** plain header (kept as `EMBED_TEXT=plain`).
- **Reason:** bge-small vector-only MRR on paraphrased queries 0.148 → 0.271; fused 0.123 → 0.220
  and original 0.634 → 0.654. Re-embedding costs ~2.5 min on the CPU.

## D-026 · Keep bge-small after comparing three free code embedders
*2026-09-23*

- **Decision:** the default stays `BAAI/bge-small-en-v1.5` (enriched); CodeRankEmbed,
  st-codesearch-distilroberta-base and CodeT5+ 110M are selectable via `EMBED_MODEL` and supported
  by `MODEL_PROFILES` (query prefixes, a CodeT5+ encoder).
- **Alternatives:** CodeRankEmbed (best on the original set alone: 0.682, but 0.037 on the
  held-out half); st-codesearch; CodeT5+ 110M embedding.
- **Reason:** best selection score and best held-out generalisation, 4× faster than
  CodeRankEmbed. CodeRankEmbed's earlier crash is fixed: its remote NomicBERT code calls
  `get_extended_attention_mask`, which transformers 5 removed, so `_patch_remote_model` restores
  the transformers-4 implementation; CodeT5+'s config lacked `is_decoder` / `use_cache` defaults.
  Shims instead of downgrading transformers keep the rest of the stack unchanged.

## D-027 · LLM restricted to query rewriting and a streamed explanation (superseded by D-028)
*2026-09-23*

- **Decision:** remove the JSON tool-calling agent loop (D-017). Ranking, routing and snippet
  selection are deterministic. The LLM (a) rewrites the query for one extra retrieval round only
  when a calibrated confidence feature is below threshold, and (b) writes an explanation from the
  verified snippets, streamed after the results are displayed.
- **Alternatives:** keep the loop with tighter budgets; LLM reranking; LLM routing fallback.
- **Reason:** on the Gladys benchmark the loop cost p95 ≈ 30 s and its citations sometimes
  displaced better retrieval results. Calibration found no rewrite policy that helps with the 3B
  CPU model under a minimum-support rule, so rewriting ships disabled (threshold 0) and the full
  agent's ranking equals the no-LLM agent's; the LLM's user-visible contribution is the
  non-blocking explanation.

## D-028 · Planner-driven agent loop with grounded refinement
*2026-09-23*

- **Decision:** restore an explicit agentic loop — plan (rules, decomposition, LLM fallback) →
  search → read (candidates + call-graph neighbours) → assess (deterministic signals) → refine
  (round 1: graph/catalog expansion; round 2: constrained rewrite over real index terms, ≤ 2
  rounds) → verify → stop — with every step a structured trace event. Deterministic work runs
  before the first results; all LLM work (planning fallback, round 2, explanation) runs after.
- **Alternatives:** the earlier free-form JSON tool loop (D-017: p95 59 s, lower MRR); free-form
  LLM query rewriting (D-027: hallucinated identifiers); dropping refinement altogether.
- **Reason:** the brief required visible plan/search/read/refine behaviour without giving up the
  0.682 MRR or the sub-second first results. Grounding the LLM (it may only pick listed terms) and
  keeping ranking deterministic makes every step checkable.

## D-029 · Keep expansion on, rewrite off; fusion weights unchanged
*2026-09-23*

- **Decision:** `REFINE = RefineConfig(gate="low", expand=True, rewrite=False)`,
  `EXPAND = ExpandConfig(lam=0.1, seeds=3)`; `nl_w_bm25` stays 0.5.
- **Alternatives:** stronger propagation (λ 0.25/0.5: −0.01 to −0.04 selection MRR); rewrite on
  (−0.02 to −0.09); embedding-heavier fusion for identifier-free questions (`nl_w_bm25` 0.25 closes
  the reworded gap — held-out 0.200 = vector-only — but costs 0.022 on the original set, lower
  selection score 0.572 vs 0.586).
- **Reason:** refinement is allowed only where it does not lower MRR; λ = 0.1 is neutral on all
  three query sets, so the agentic round stays; the rewrite does not help with the 3B model and is
  off. Fusion: the pre-declared selection rule keeps 0.5 (the "NL bm25 0.25/0.1/0.0" rows of the
  fusion sweep).

## D-030 · Reintroduce an HTTP API (FastAPI + Server-Sent Events)
*2026-09-23*

- **Decision:** a FastAPI layer (`src/api/app.py`) wraps the engine unchanged; the ask endpoint
  streams the structured trace, results and explanation as Server-Sent Events over a `POST`
  (read with `fetch` and a stream reader on the client). It also serves the built frontend.
- **Alternatives:** keep Streamlit only (D-019 dropped the API because Streamlit imported the
  engine directly); WebSockets; `GET` + `EventSource` (cannot send a JSON body, query length
  limits); Flask.
- **Reason:** a React frontend needs HTTP. SSE is one-way, which is all the trace needs, works
  over plain HTTP and is trivial to test with `TestClient`; FastAPI gives typed request models and
  OpenAPI docs for free. A worker thread plus a queue lets the synchronous engine emit events as
  they happen without changing it. Supersedes the "API dropped" part of D-019.

## D-031 · Web frontend: Vite + React + TypeScript + Tailwind, served by the API
*2026-09-23*

- **Decision:** a React 19 SPA in `frontend/` (Vite 8, TypeScript, Tailwind v4 via
  `@tailwindcss/vite`), with prism-react-renderer for highlighting, @xyflow/react for the
  call-path graph, recharts for metrics charts, lucide-react icons and react-router. It is built to
  `frontend/dist` and served by FastAPI on the same port as the API; Streamlit stays as a
  fallback.
- **Alternatives:** Shiki (async, WASM/oniguruma, larger bundle) vs Prism; Cytoscape vs React
  Flow; a separate static server or the Vite dev server in production; Streamlit only.
- **Reason:** Prism highlighting is synchronous (no flash while streaming results) and small;
  React Flow gives an interactive, themeable graph with React nodes and needs no layout engine for
  the shallow call paths shown. Serving the build from FastAPI makes `./run.sh app` one process
  and one URL, with no CORS in the demo. All packages are free/MIT and run offline once installed.

## D-032 · One-process run model and a Playwright e2e test
*2026-09-23*

- **Decision:** `./run.sh app` builds the frontend if stale and runs a single uvicorn process that
  serves both the API and `frontend/dist` on :8000; `./run.sh dev` runs uvicorn `--reload` plus
  the Vite dev server (proxy `/api`); `./run.sh streamlit` keeps the Streamlit UI. Browser tests
  use the Python `playwright` package in `.venv` driving headless Chromium stored in
  `.cache/ms-playwright`, marked `e2e` and excluded from `./run.sh test`; each run starts its own
  servers on free ports (one with the LLM pointed at a closed port).
- **Alternatives:** two processes (Vite preview + API) for `app`; `@playwright/test` in Node;
  pytest-playwright.
- **Reason:** one URL and one process is simplest to demo and to verify from a clean clone.
  Python Playwright reuses the existing pytest setup and venv (no second test runner), and keeping
  the browser inside the project honours "nothing installed globally".

## D-033 · Pivot to snippet-library retrieval scored on CoIR APPS
*2026-09-27*

- **Decision:** the primary product becomes a CPU-only retrieval engine that ranks a library of
  code snippets for a natural-language query (`src/codesearch/`), evaluated officially through
  MTEB `AppsRetrieval`, plus versioned indexes (P1) and cross-version lineages (bonus). The
  Gladys assistant stays as secondary demo material; its LLM explanation is kept but off by
  default (`CODEINTEL_EXPLAIN`).
- **Alternatives:** adapt the Gladys agent (its JavaScript-specific AST, catalog and call graph do
  not apply to APPS Python snippets); start a separate repository (loses the working
  UI/API/test scaffolding).
- **Reason:** the organisers' updated guidelines are authoritative: retrieval only, screening on
  APPS NDCG@10/MRR. Reusing the API, frontend and run scripts keeps one demo.

## D-034 · Text-only inputs; no dataset-structure tricks
*2026-09-27*

- **Decision:** only `text` reaches any model or ranking step. Ids, `meta_information` (URL,
  starter code) and `partition` are never used at retrieval time. We also do not exploit that
  every snippet answers exactly one query (one-to-one assignment) or that train snippets answer
  known train problems.
- **Alternatives:** using those signals, which would raise the APPS score substantially.
- **Reason:** they are label leakage or benchmark artefacts, would not work on a real snippet
  library, and the hands-on round runs the code on new queries.

## D-035 · Tune on stdin-style train problems only
*2026-09-27*

- **Decision:** dev-tune / dev-holdout = 600 + 600 stdin-style APPS train queries (empty
  `starter_code`, read only to build the split) against the 5 000 train snippets; train-pool = the
  other 3 800 train pairs. The test split is never used for selection.
- **Alternatives:** random train queries (dominated by function-style problems whose names appear
  in the solution: bge-small 0.466 vs 0.067 on stdin-style, published test 0.056); a slice of the
  test split (forbidden).
- **Reason:** the dev set must predict test behaviour, and the test split is 99 % stdin-style.

## D-036 · Force float32 when loading embedding models on the CPU
*2026-09-27*

- **Decision:** `src/codesearch/encoders.py` loads every model with `dtype=torch.float32` (and
  `device="cpu"`, CUDA hidden).
- **Alternatives:** the transformers-5 default (checkpoint dtype, fp16 for gte-modernbert).
- **Reason:** measured on 16 × 512-token snippets: 25 s in fp16 vs 2.0 s in fp32 (12×) on the
  M1 Pro CPU; half-precision matmul has no fast CPU path here. The model benchmark's first
  attempt appeared to hang for this reason.

## D-037 · Hubness estimated from snippets only
*2026-09-27*

- **Decision:** the hubness penalty is each snippet's mean cosine to its k nearest *snippets*.
- **Alternatives:** the usual query-based estimate (mean similarity to the k nearest reference
  queries, e.g. APPS train queries).
- **Reason:** every APPS query set has its answers inside the corpus, so a query-based penalty
  demotes exactly the snippets that answer known problems — the leakage D-034 rules out. The
  snippet-side estimate needs no queries at all and works for any library.

## D-038 · Content-addressed caches for versioned indexes
*2026-09-27*

- **Decision:** versions are stored per (library, version); chunking is cached by git blob SHA and
  embeddings by sha1(model, max_seq, text), so rebuilding any version re-processes only what
  changed. The Gladys history is a separate bare clone in `data/versions/` (`sample_repo` stays
  read-only and shallow).
- **Alternatives:** diffing versions explicitly (git diff → changed files → patch the index); one
  index per version rebuilt from scratch; fetching history into `sample_repo`.
- **Reason:** content addressing needs no diff logic, is correct for any pair of versions (also
  non-adjacent ones and snippet sets without git), and makes the cache shared across versions,
  libraries and the MTEB run. `sample_repo` must not be modified.

## D-039 · No query adapter
*2026-09-27*

- **Decision:** do not ship a learned linear query adapter.
- **Alternatives:** a residual linear map on frozen query embeddings trained with InfoNCE on the
  3 800 train-pool pairs.
- **Reason:** with bge-small it moved dev-tune NDCG@10 from 0.0670 to 0.0695 at 20 epochs and
  below baseline with more training (0.0635, 0.0551): train-pool is 88 % function-style problems,
  unlike the stdin-style test set (`results/apps_negative_results.md`).

## D-040 · jina-code-embeddings is contaminated on train-derived dev sets; tune around it
*2026-09-27*

- **Finding:** jina-code-embeddings-0.5b scores 0.959 / 0.978 NDCG@10 on dev-tune / dev-holdout
  but 0.842 on the APPS test split (published); its paper lists AppsRetrieval's training split
  (and CodeForces Python submissions) among its training data. Every other model's dev score
  tracks its published test score (Qwen3-Embedding-0.6B: 0.757 dev vs 0.752 test; gte-modernbert
  0.583 vs 0.564; bge-small 0.067 vs 0.056).
- **Decision:** jina-code-0.5b is the base model (the best APPS model that runs on this CPU; using
  a model trained on the task's *train* split is legitimate). Post-processing knobs (query views,
  snippet view, style, literal, hubness, BM25) are tuned on **Qwen3-Embedding-0.6B as an
  uncontaminated proxy** and kept only if they do not hurt jina on dev-tune either. Ensemble
  weights are judged on jina's dev-tune, which is conservative: the contamination favours jina
  alone, so a gain from adding another model there understates the gain on unseen problems.
- **Alternatives:** tune everything on jina's dev scores (near ceiling, would reject any
  complementary signal); tune on the test split (forbidden); drop jina (−9 points published).
- **Reason:** keeps the no-test-tuning rule while still choosing sensibly. The README states the
  contamination openly.

## D-041 · Final APPS pipeline
*2026-09-27*

- **Decision:** `FINAL_CONFIG` = jina-code-embeddings-0.5b + Qwen3-Embedding-0.6B (weight 0.4)
  over the cleaned full query and raw snippets, top-100 reranked with program-style agreement
  0.5, literal overlap 0.1, snippet-side hubness 0.125 and BM25 0.025.
- **Evidence (dev-tune NDCG@10):** Qwen3 proxy 0.7573 raw → 0.7586 cleaned → 0.7748 with the
  knobs at half weight (0.7757 at the sweep optimum; dev-holdout 0.7712 → 0.7918); on jina's
  contaminated dev the knobs are neutral (±0.002) and the ensemble adds +0.0045–0.005 at weights
  0.3–0.5 (0.4 best). A second query view without examples: +0.004 dev-tune, +0.0006
  dev-holdout (−0.004 in the ablation run) — not worth doubling query-encoding time. Dropping
  examples hurt (−0.008 tune, −0.015 holdout). Normalising code is mixed (−0.0015 with two views
  in the sweep, +0.004 with one view in the ablation): within noise, not adopted.
- **Alternatives:** knobs at the proxy optimum (×1): best for a 0.76 model, slightly negative on
  jina; no knobs: loses the proxy's +0.016; jina alone: loses the ensemble gain; jina-code-1.5b
  (published +2.4 test points, ~3× CPU time; not evaluated).
- **Reason:** the best configuration that the tuning splits support without touching test, within
  a CPU budget of ~1.5 h for the full official run.

## D-042 · Evaluate all-versions retrieval with de-duplicated NDCG
*2026-09-28*

- **Decision:** the bonus benchmark's headline metric is NDCG@10 where each lineage counts once
  (later versions of an already-listed lineage score 0; ideal = the wanted version at rank 1),
  reported with exact-version@1, lineage MRR@10 and distinct lineages in the top 10.
- **Alternatives:** graded NDCG with every version of the target lineage relevant (the first
  version of the benchmark).
- **Reason:** graded NDCG rewards listing three copies of the same snippet, so it scored the
  lineage collapse *lower* (0.43 vs 0.67) while collapse doubled the distinct snippets shown and
  raised exact-version@1 from 0.41 to 0.61. The graded column stays in the results with that
  caveat.
