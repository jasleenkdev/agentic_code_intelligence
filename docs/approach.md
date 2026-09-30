# Approach: multi-pass code snippet retrieval on CPU

This document describes the problem, the retrieval pipeline, how it was tuned, what was tried and
rejected, the versioning (P1) and evolutionary-retrieval (bonus) designs, and the known
limitations. Individual design choices, with their alternatives, are recorded in
[`decisions.md`](decisions.md) and cited here as `D-0xx`.

---

## 1. Problem

Given a natural-language query and a large library of code snippets (thousands, some long), rank
the snippets by relevance. Only retrieval is in scope: no code generation or explanation after
retrieval. The organisers' guidelines allow query categorisation, query pre-processing, snippet
categorisation, snippet pre/post-processing and multiple retrieval passes, and require everything
to run on the CPU with free models.

Screening uses **NDCG@10 and MRR@10 on the test split of CoIR APPS**, computed by MTEB's
`AppsRetrieval` task. Project constraints on top of that: free, local, open-weight models only; no
paid APIs; CPU only, enforced in all model-loading code; nothing written outside the project
folder.

## 2. Data and evaluation protocol

### 2.1 MTEB interface

Findings from the installed MTEB source (mteb 2.21.8):

- `AbsEncoder` has one required method, `encode(inputs, *, task_metadata, hf_split, hf_subset,
  prompt_type, **kwargs)`. `prompt_type` distinguishes queries from documents, so query and
  snippet pre-processing can differ. Similarity defaults to cosine.
- **Custom multi-stage search is supported.** A model that also implements MTEB's
  `SearchProtocol` — `index(corpus, …)` and `search(queries, …, top_k) -> {qid: {doc_id: score}}`
  — is used directly instead of being wrapped in MTEB's default single-embedding search. MTEB
  calls `index` once with the whole corpus, then `search` with all queries, and scores whatever
  ranking is returned. The official run can therefore use any number of retrieval passes, fusion
  and reranking steps.
- `mteb.evaluate(model, tasks, encode_kwargs=…)` returns a `ModelResult`;
  `task_results[0].to_dict()` is the JSON the guidelines ask for. The default result cache skips
  tasks that already have results, so the submission script points MTEB's cache at a fresh
  directory and forces recomputation.
- AppsRetrieval uses dataset `CoIR-Retrieval/apps` @ `f22508f9`, evaluation split `test`, main
  score `ndcg_at_10`; MTEB also reports `mrr_at_10`.

### 2.2 Data profile

| | Train split | Test split (scored) |
|---|---|---|
| Queries | 5 000 | 3 765 |
| Corpus searched | all 8 765 snippets (5 000 train + 3 765 test solutions) | same 8 765 |
| Relevant per query | exactly 1 | exactly 1 |
| Query length (chars, p50) | 1 052 | 1 601 |
| Snippet length (chars, p50) | 280 | 399 (max 289 k) |
| Problem style | 67 % function-style (Codewars, LeetCode with starter code) | **99 % stdin-style** (Codeforces, AtCoder) |

Queries are full problem statements (story, `-----Input-----` / `-----Output-----` sections,
examples, notes, LaTeX `$…$`); snippets are Python solutions (stdin scripts, functions,
`class Solution`). The dataset's `meta_information` holds the problem URL on both sides and ids
pair up (`q5001` ↔ `d5001`); neither is ever used (D-034).

Published APPS test NDCG@10 for reference: bge-small 5.6, e5-base 11.5, Voyage-code-002 26.5,
SFR-Embedding-Code-400M 48.6, CodeSage-large-v2 50.5, gte-modernbert-base 56.4,
Qwen3-Embedding-0.6B 75.2, SFR-Embedding-Code-2B 75.0, jina-code-embeddings-0.5b 84.2 / 1.5b 86.6,
voyage-code-3 93.8 (CodeXEmbed paper Table 1; Granite-R2 paper Table 11; jina-code-embeddings
paper Table 2).

### 2.3 Tuning protocol: the test split is never used for choices

All model selection and tuning use the **train** split (D-035):

- **dev-tune** — 600 stdin-style train problems (seed 13), used for every choice;
- **dev-holdout** — another 600, used only to confirm a choice;
- both searched against the 5 000 train snippets;
- **train-pool** — the remaining 3 800 train pairs, the only data any fitted component may learn
  from.

The stdin-style restriction matters because train is mostly function-style while test is almost
entirely stdin-style: bge-small scores NDCG@10 0.466 on random train queries but 0.067 on
stdin-style ones — close to its published test score of 0.056. On stdin-style dev sets, dev scores
track published test scores closely (gte-modernbert 0.583 vs 0.564, Qwen3 0.757 vs 0.752).
`src/codesearch/data.py` builds the splits.

The test split was used once for the submitted pipeline (the official run) and afterwards for
reporting-only baselines (`results/apps_test_baselines.md`).

**No leakage (D-034, D-037).** Only the `text` of queries and snippets reaches the retriever.
Dataset ids, `meta_information` and the corpus `partition` field are never read at retrieval
time. Tricks that exploit benchmark structure — one-to-one assignment of queries to snippets,
demoting snippets that answer known train problems, query-based hubness — were rejected because
they would not work on a real snippet library.

## 3. Pipeline

```mermaid
flowchart LR
    Q[Query text] --> P0[Pass 0 · pre-process<br/>split statement / input / output /<br/>examples / notes, strip LaTeX,<br/>categorise: stdin · function · class]
    P0 --> P1[Pass 1 · dense recall<br/>Σ w_model · cos<br/>= one cosine over concatenated<br/>weighted embeddings]
    L[(Snippet library<br/>embeddings · styles ·<br/>literal index · hubness)] --> P1
    P1 -->|top 100| P2[Pass 2 · rerank<br/>z(dense) + w_style·[style agrees]<br/>+ w_literal·log(1+shared-literal IDF)<br/>+ w_bm25·z(BM25) − w_hub·hubness]
    P2 --> R[Ranked snippet ids + scores]
```

The implementation is `src/codesearch/pipeline.py`; all weights live in `PipelineConfig`.

### 3.1 Pass 0 — query and snippet pre-processing

APPS statements carry online-judge boilerplate: `-----Input-----` headers, sample I/O blocks and
LaTeX constraints such as `$1 \le n \le 10^5$`. `src/codesearch/preprocess.py`:

- splits the statement into sections (statement, input format, output format, examples, notes)
  using header rules; bare `Input` / `Output` lines inside the examples are treated as sample
  labels, not section headers;
- removes LaTeX markup and horizontal rules;
- **categorises** the program the query asks for: a *stdin* script, a *function*, or a *class*
  method, plus topics, multi-test and interactive flags.

The sample I/O is kept in the query: dropping it hurt (§5). Snippets are embedded as raw code;
each snippet is also categorised by program form (stdin script / function / `class Solution`) and
its string and number literals are indexed.

### 3.2 Pass 1 — dense recall with a two-model ensemble

Two free code embedding models run on the CPU in float32, each with its own instruction prefix:

| Model | Weight | Role |
|---|---|---|
| `jinaai/jina-code-embeddings-0.5b` | 1.0 | primary embedder |
| `Qwen/Qwen3-Embedding-0.6B` | 0.4 | ensemble member; uncontaminated tuning proxy (§4) |

A weighted sum of per-model cosines equals a single cosine over the weighted concatenation of the
L2-normalised embeddings, so the ensemble is one exact search over all snippets, and the MTEB
class's `encode()` returns exactly that concatenated vector. Models are forced to float32 because
half precision was 12× slower on this CPU (D-036).

### 3.3 Pass 2 — rerank and score fusion

The top 100 candidates from pass 1 are reranked with a linear fusion of per-query z-scored
signals:

- **dense score** from pass 1 (z-scored per query);
- **program-style agreement** — a bonus when the snippet's program form matches the query's
  category;
- **shared-literal overlap** — string and number literals that appear both in the code and in the
  statement (`"IMPOSSIBLE"`, `998244353`), IDF-weighted over the library (`lexical.py`);
- **snippet-side hubness penalty** — each snippet's mean cosine to its k nearest *snippets* is
  subtracted, damping generic snippets that are close to everything. It is estimated without any
  queries (D-037);
- **BM25** over code tokens at a small weight (harmful at larger weights, §5).

### 3.4 Final configuration (D-041)

jina-code-embeddings-0.5b + Qwen3-Embedding-0.6B (weight 0.4) over the cleaned full query and raw
snippets; top-100 reranked with program-style agreement 0.5, literal overlap 0.1, snippet-side
hubness 0.125 and BM25 0.025. The rerank weights are half of the proxy-model sweep optimum: at the
full optimum they were best for the 0.76-NDCG proxy but slightly negative on jina.

### 3.5 Official run

`PrePostPipelineEncoder(AbsEncoder)` in `src/codesearch/submission.py` implements `encode()` (the
weighted concatenated embedding) and MTEB's `SearchProtocol` (`index()` builds the snippet library
once; `search()` runs passes 0–2 for all queries). `./run.sh submit` (`scripts/submit.py`) runs
the guidelines' exact `mteb.evaluate(model, [task], encode_kwargs={"batch_size": 64})` with MTEB's
cache pointed at a fresh directory, then writes `release/appsretrieval_results.json` and
`release/run_info.json` (timings, pipeline configuration, whether embeddings were encoded or
reused from the content-hash cache).

**Result on the APPS test split:** NDCG@10 **0.8657**, MRR@10 **0.8380**, against 0.8387 / 0.8081
for jina-code-embeddings-0.5b on raw text measured on the same machine.

## 4. Model selection

Six free CPU-capable embedders were benchmarked on dev-tune / dev-holdout
(`results/apps_model_benchmark.md`):

| Model | Params | dev-tune NDCG@10 | dev-holdout NDCG@10 | Published test |
|---|---|---|---|---|
| jina-code-embeddings-0.5b | 494M | 0.9586¹ | 0.9776¹ | 0.842 |
| Qwen3-Embedding-0.6B | 596M | 0.7573 | 0.7712 | 0.752 |
| gte-modernbert-base | 149M | 0.5825 | 0.5542 | 0.564 |
| CodeRankEmbed | 137M | 0.2295 | 0.2556 | — |
| granite-embedding-small-english-r2 | 47M | 0.1469 | 0.1706 | — |
| bge-small-en-v1.5 | 33M | 0.0670 | 0.0738 | 0.056 |

¹ Contaminated — see §9.1.

jina-code-embeddings-0.5b is the base model: the strongest APPS model that runs acceptably on this
CPU. Because its dev scores are inflated, the pre/post-processing knobs were tuned on
**Qwen3-Embedding-0.6B as an uncontaminated proxy** and kept only if they did not hurt jina on
dev-tune either; the ensemble weight was judged on jina's dev-tune, which is conservative (D-040).

## 5. What was tried and what was rejected

### 5.1 Ablation ladder (`results/apps_ablation.md`)

Proxy model (Qwen3-Embedding-0.6B), NDCG@10; each row adds one change to the row above:

| Step | dev-tune | dev-holdout |
|---|---|---|
| raw text, cosine only | 0.7573 | 0.7712 |
| + cleaned query (LaTeX / judge boilerplate removed) | 0.7586 | 0.7804 |
| + program-style agreement (w 0.5) | 0.7643 | 0.7847 |
| + shared-literal overlap (w 0.1) | 0.7675 | 0.7877 |
| + snippet-side hubness penalty (w 0.125) | 0.7729 | 0.7899 |
| + small BM25 term (w 0.025) = the submitted knobs | **0.7748** | **0.7913** |
| knobs at the sweep optimum (×2) | 0.7757 | 0.7918 |

On jina the same steps sit at a contaminated ceiling (raw 0.9586 / 0.9776; final pipeline
0.9629 / 0.9737) and cannot rank variants; adding Qwen3 at weight 0.4 raised jina's dev-tune from
0.9586 to 0.9634. A coordinate-ascent sweep over all knobs is in `results/apps_sweep.md`.

### 5.2 Variants measured and not adopted

| Variant | Effect | Verdict |
|---|---|---|
| Dropping the examples from the query | −0.008 dev-tune, −0.015 dev-holdout | Hurts: sample I/O carries format and literal cues |
| Second query view without examples | +0.0001 tune, −0.004 holdout in the ablation (+0.004 / +0.0006 in the sweep) | Not worth doubling query-encoding time |
| Statement-only second view (w 0.5) | 0.7519 vs 0.7586 | Hurts |
| Comment/docstring-free (normalised) snippets | +0.004 with one view, −0.0015 with two views | Within noise, not adopted |
| Hubness penalty at w ≥ 0.5 | 0.7590 at 0.5, 0.7164 at 1.0 (vs 0.7699 at 0.25) | Strong hubness removes true answers |
| BM25 over code tokens as a meaningful rerank feature | gte-modernbert, w 0.2: 0.512 vs 0.592 | Harmful: stdin solutions use short variable names, so word overlap with the statement is noise; kept only at a very small weight |
| Learned query adapter (residual linear map, InfoNCE on train-pool) | bge-small: 0.0695 / 0.0635 / 0.0551 at 20 / 50 / 100 epochs vs 0.0670 | Dropped (D-039): train-pool is 88 % function-style, so the map learns identifier matching that stdin-style test problems lack |
| jina-code-embeddings-1.5b | published +2.4 test points, ~3× CPU time | Not evaluated within the CPU budget |

### 5.3 Rejected on principle (not run)

Recorded in `results/apps_negative_results.md`:

- **Query-based hubness** (reference = train queries): reference queries have their answers in
  the corpus, so the penalty would demote known answers — leakage (D-037). Replaced by the
  snippet-side estimate.
- **One-to-one query↔snippet assignment** and **demoting snippets that answer known train
  problems**: benchmark artefacts that would not work on a real snippet library (D-034).

## 6. Versioned libraries (P1)

```
index/versions/<library>/<version>/manifest.json · snippets.jsonl · emb__<model>.npy
index/emb_cache/<model>__<maxseq>/shard_*.npz      sha1(model, max_seq, text) → vector
index/chunk_cache/<repo>.pkl                         "<path>@<blob sha>" → function chunks
```

Implemented in `src/codesearch/versions.py` (D-038):

- A library version can come from a git ref (codebases; the Gladys history is a separate bare
  clone in `data/versions/gladys.git`), the APPS corpus, a JSONL snippet set, or a synthetic
  evolution of an indexed version (`synthetic.py`).
- Codebase snippets are function/method chunks from the tree-sitter extractor (windows of 120
  lines, overlap 10), with a stable lineage key `path::qualname#window`.
- **Content-addressed caches.** Chunking is cached by git blob SHA and embeddings by
  sha1(model, max_seq, text). A new version re-parses only files whose blob changed and re-encodes
  only snippets whose text changed. No diff logic is needed, the scheme is correct for any pair
  of versions (including non-adjacent ones and snippet sets without git), and the cache is shared
  across versions, libraries and the MTEB run.
- One command indexes a version: `./run.sh index-version <library> --ref … | --jsonl … |
  --evolve-from …`.

Measured with gte-modernbert on fresh caches (`results/p1_rebuild.md`):

| Library | Version | Build | Snippets | Re-encoded | Time |
|---|---|---|---|---|---|
| Gladys (JS) | v5.0.4 | full | 6 424 | 6 422 | 470 s |
| Gladys | v5.1.0 (146 files changed) | incremental | 6 642 | 394 | 51 s |
| Gladys | v5.1.1 (1 file changed) | incremental | 6 642 | 1 | 0.8 s |
| APPS-derived | v1 | full | 2 000 | 2 000 | 154 s |
| APPS-derived | v2 (10 % mutated) | incremental | 2 000 | 176 | 16 s |
| APPS-derived | v3 (1 % mutated) | incremental | 2 000 | 23 | 3.8 s |

Rebuild time is proportional to the changed snippets, not the library size.

## 7. Evolutionary retrieval (bonus)

Retrieval across all versions at once, implemented in `src/codesearch/lineage.py`:

- **Lineages.** Snippets across versions are grouped with union-find using, in order: a stable
  key when the source provides one; identical text; then content similarity — embedding cosine
  ≥ `cos_min` to one of the 5 nearest neighbours in another version **and** token-shape Jaccard
  ≥ `jaccard_min`, where identifiers and literals are abstracted so renames and reformatting do
  not break a lineage.
- **Version hints** are parsed from the query ("latest", "original", "v2", "5.1.0") and removed
  before encoding.
- **Ranking.** Every (snippet, version) is scored; results collapse to one hit per lineage (its
  maximum score). The hit shown is the hinted version, otherwise the best match with near-ties
  going to the newest version. The UI lists every version in the lineage with its score and
  whether its text differs.

**Benchmark** (`eval/evolution_bench.py` → `results/evolution_bench.md`). 5 000 APPS train
solutions evolved v1 → v2 → v3 with behaviour-preserving mutations (renames, comments, fast I/O,
`main()` wrapping, reformatting) give 15 000 near-duplicate entries indexed together; queries are
the 600 + 600 stdin-style dev problems with rotating version hints. Lineages inferred from content
reach pair precision 0.88 and recall 0.94. The headline metric is de-duplicated NDCG@10, where
each lineage counts once (D-042); graded NDCG rewards listing three copies of the same snippet
and is reported only for completeness.

| dev-holdout | NDCG@10 (dedup) | Right version @1 | Lineage MRR@10 | Distinct snippets in top 10 |
|---|---|---|---|---|
| flat ranking | 0.5595 | 0.4133 | 0.6901 | 4.5 |
| flat, version hint stripped | 0.5604 | 0.4150 | 0.6914 | 4.5 |
| **lineage handling (inferred)** | **0.7383** | **0.6083** | **0.7248** | **9.9** |
| lineage handling (true lineages) | 0.7460 | 0.6100 | 0.7270 | 10.0 |

## 8. Demo engine, API and UI

`engine.SearchEngine` loads versioned libraries (cached in memory per library and version), runs
the same pipeline as the official run and returns the query category, the pre-processed text,
per-pass traces and latency. `/api/cs/{libraries,examples,search,eval}`
(`src/api/codesearch_routes.py`) serve it; the React **Search** page has a library switcher (APPS
corpus / Gladys), a version selector with an "All versions (lineages)" mode, example queries, a
latency banner and the pass timeline. The CLI equivalent is `./run.sh search`.

A codebase library (Gladys, JavaScript, embedded with gte-modernbert) uses the same pipeline
restricted to the models it was indexed with; the program-style signal is off there, because the
stdin/function/class categories describe competitive-programming Python only. At API start a
background thread loads the newest APPS library and both query encoders so the first live query
does not pay ~15 s of model loading (`CODESEARCH_WARM=0` disables this). Query embeddings are
content-hash cached, and the pass timeline states whether a query was encoded on the CPU or
served from the cache, so the latencies shown are honest.

## 9. Known limitations

### 9.1 jina-code-embeddings dev-split contamination

jina-code-embeddings was trained on the AppsRetrieval **train** split (per its paper), and every
tuning split here is derived from train. Its dev scores (0.959 / 0.978) are therefore inflated
relative to its test score (0.842) and cannot discriminate between pipeline variants. Using a
model trained on a task's train split is legitimate, but it means:

- pre/post-processing was tuned on a different model (Qwen3 proxy) and transferred, which may be
  sub-optimal for jina;
- the ensemble weight was judged on contaminated dev scores, which favour jina alone, so the
  chosen weight is conservative rather than optimal.

### 9.2 Cold vs warm timing

A full **cold** official run on the Apple M1 Pro CPU takes **~1 h 51 min** (6 642 s: 55 min
encoding the 8 765 snippets with both models, 56 min encoding the 3 765 queries and searching),
measured in a clean copy with empty caches; all 149 numeric metrics matched the committed JSON. A
rerun with warm embedding caches takes 41 s, because both snippet and query embeddings are
content-hash cached. First-time setup (mostly model downloads) took 44 min. Any timing quoted
should be read with this distinction in mind; `release/run_info.json` records whether embeddings
were encoded or reused.

### 9.3 Size of the gain over the base model

On the test split, pre/post-processing plus the Qwen3 ensemble add **+0.027 NDCG@10** (0.8387 →
0.8657) and +0.030 MRR@10 over the same base model on raw text. Most of the final score comes from
the choice of base model; the multi-pass pipeline contributes a real but modest improvement. The
ablation shows larger relative gains on the uncontaminated proxy (+0.018 NDCG@10 on dev-tune),
which suggests the rerank signals help most where the embedder is weaker.

### 9.4 Other limitations

- **CPU budget.** The larger jina-code-embeddings-1.5b (published 0.866) was not evaluated: about
  3× the CPU time of the 0.5B model.
- **Domain specificity.** Program-style categories and literal overlap are designed around
  competitive-programming problems; on general codebases the style signal is disabled.
- **Synthetic evolution benchmark.** The bonus benchmark uses behaviour-preserving mutations of
  APPS solutions, which are easier to link into lineages than real refactors.
- **Licence.** jina-code-embeddings is CC-BY-NC-4.0 (non-commercial), which covers this
  submission but not commercial use.

## 10. Hardware

Apple MacBook Pro, M1 Pro (8 CPU cores: 6 performance + 2 efficiency), 16 GB RAM, macOS 26.3,
Python 3.10.13, torch 2.14 (CPU), transformers 5.17, sentence-transformers 6.1, mteb 2.21.8.
Inference is forced onto the CPU (`device="cpu"`, CUDA hidden, Ollama `num_gpu=0`).

---

## Appendix A. Secondary demo: Gladys code-intelligence agent

The first-round prototype answers plain-English, structural and usage questions over the Gladys
Assistant JavaScript codebase with verified `file:line` citations. It remains built and tested
but is not part of the scored submission. Its decisions are D-001 … D-032.

### A.1 Target codebase

**Gladys Assistant** (https://github.com/GladysAssistant/Gladys, Apache-2.0) @ `8574722` (v5.1.1),
chosen because it is 100 % JavaScript, large and actively maintained (D-012). The index covers
`server/` and `front/src/` minus tests, migrations and seeders: 2 186 files, 212 547 LOC. Server
code is CommonJS (Node, Express 4, Sequelize), front-end code ESM (Preact 10). Domain structure the
catalog relies on: `INTENTS` / `EVENTS` / `ACTIONS` constants in `server/utils/constants.js`, an
`event.emit/on` bus, services as `server/services/<name>/index.js` factories, MCP tools in
`server/services/mcp/lib/buildSchemas.js`, HTTP routes in `server/api/routes.js`, and front-end
routes that act as deeplinks. The built index holds 9 972 symbols, 10 765 resolved call-graph
edges and 44 065 string literals.

### A.2 Agent tools

Eight read-only tools, each returning structured records rather than prose: `semantic_search`,
`keyword_search`, `find_symbol`, `get_callers`, `get_callees`, `find_call_sequence`,
`find_string_usages`, `read_file_range`. A registry validates and coerces arguments and returns
`{"error": …}` instead of raising, so a failing tool becomes a trace event rather than a crash.
`read_file_range` is confined to `sample_repo/` and capped at 300 lines. `find_call_sequence`
orders call sites within a function and optionally follows up to two resolved call-graph hops,
labelling each result `certain` or `possible` (branches, callbacks and heuristic hops make it
`possible`).

### A.3 Query routing

A rule-based classifier (no LLM, D-027) routes questions:

| Class | Signals | Route |
|---|---|---|
| Structural | "calls", "before/after", "callers of", "called by" | graph tools, deterministic |
| Usage | quoted strings, "where is X used", route/URL/intent/event/constant words | catalog + `find_string_usages` / `find_symbol`, deterministic |
| Semantic / unclear | anything else | planner → hybrid search → read → assess → refine (≤ 2) → verify |

Misclassification degrades to hybrid retrieval, never to an error. The planner-driven loop
(D-028) returns deterministic first results before any LLM work; the optional LLM explanation is
off by default.

### A.4 Ranking and fusion

Function-level chunks with a context header (D-004); BM25 over identifier-split tokens (D-007);
bge-small embeddings over enriched chunk text (D-025) with exact numpy search (D-018). The two
lists are fused by **query-adaptive min-max score fusion** (D-024): vector weight 0.25 when the
query contains a code identifier, BM25 weight 0.5 when it contains none, otherwise 1.0. Equal-
weight RRF remains available as a baseline. The agent adds a structural list (definitions,
callers and callees of identifiers in the query, top catalog hit) and an optional CPU
cross-encoder rerank.

### A.5 Evaluation: gold set and metrics

- **Gold set** (`eval/generate_gold.py` → `eval/gold.json`): 36 queries (12 semantic, 12
  structural, 12 usage), seed 42. **Labels are auto-derived, not hand-made**: semantic queries
  come from a function's JSDoc (gold = that function's range); structural queries from the call
  graph; usage queries from string literals, constants and catalog entries found by exact search.
  Labels are only as good as the extractor and grep, and semantic queries are paraphrased from
  docstrings, so semantic scores are optimistic.
- **Paraphrased set** (`eval/gold_paraphrased.json`, D-024a): 20 LLM-written questions validated
  to share no identifier stems or doc bigrams with their target; split into reworded-tuning and
  reworded-held-out halves.
- **Independent structural check** (`eval/structural_checker.py`, D-023): a regex-based checker
  that shares no code with the indexer re-derives the structural labels.
- **Metrics** (`eval/metrics.py`, D-020): a result is relevant if it is in the same file as a gold
  location and the line ranges overlap; precision@k divides by min(k, returned); recall@k is over
  gold locations; MRR uses the first relevant result.
