# Multi-pass code retrieval on CPU — CoIR APPS

Submission for the Samsung PRISM Generative AI Hackathon (3rd edition), theme **"Agentic Code
Intelligence 01"**, under the organisers' updated guidelines: *given a natural-language query and a
large library of code snippets, rank the snippets by relevance.* Screening metric: **NDCG@10 and
MRR on the test split of CoIR APPS** via MTEB's `AppsRetrieval` task.

- **Retrieval only** — no generation. Free, local, open-weight models; **everything runs on the
  CPU** (every model is loaded on `device="cpu"` in float32; no GPU/MPS).
- **Multi-pass pipeline inside the official MTEB run**: query pre-processing and categorisation →
  dense recall with a two-model ensemble → rerank with program-style, literal-overlap, hubness and
  lexical signals. The MTEB class implements MTEB's `index()`/`search()` protocol, so the official
  evaluation runs the full pipeline, not just one embedding per text.
- **Versions (P1)** and **all-versions retrieval with lineages (bonus)**, with a live demo UI.

## Results

### APPS test split (official run, `./run.sh submit`)

| System | NDCG@10 | MRR@10 | Notes |
|---|---|---|---|
| **Our pipeline (`PrePostPipelineEncoder`)** | **0.8657** | **0.8380** | jina-code-0.5b + Qwen3-0.6B ensemble, pre/post-processing; CPU |
| jina-code-embeddings-0.5b, raw text (our run) | 0.8387 | 0.8081 | single-model baseline, same machine |
| bge-small-en-v1.5, raw text (our run) | 0.0564 | 0.0482 | single-model baseline; matches the published 0.056 |
| jina-code-embeddings-1.5b | 0.866 | — | published (jina-code-embeddings paper) |
| jina-code-embeddings-0.5b | 0.842 | — | published |
| Qwen3-Embedding-0.6B | 0.752 | — | published |
| gte-modernbert-base (149M) | 0.564 | — | published (Granite-R2 paper) |
| SFR-Embedding-Code-400M | 0.486 | — | published (CodeXEmbed paper) |
| Voyage-code-002 (API) | 0.265 | — | published |
| bge-small-en-v1.5 | 0.056 | — | published |

Full **cold** test run on an Apple M1 Pro **CPU**: **1 h 51 min** (6 642 s: 55 min encoding the
8 765 snippets with both models, 56 min encoding the 3 765 queries and searching), measured in a
clean copy with empty caches; all 149 numeric metrics in its json are identical to the committed
one. Re-running with warm embedding caches takes 41 s. First-time setup (`./run.sh setup`,
mostly model downloads) took 44 min. The json for the GitHub Release is
`release/appsretrieval_results.json` (`release/run_info.json` has timings, the pipeline
configuration and whether embeddings were encoded or reused).

The pre/post-processing and the Qwen3 ensemble add **+0.027 NDCG@10 and +0.030 MRR@10** over
the same base model on raw text. Baselines were run after the pipeline was frozen
(`results/apps_test_baselines.md`).

### How the choices were made — the test split was never used for tuning

- **Tuning data = APPS *train* split only.** APPS test is 99 % stdin-style problems
  (Codeforces/AtCoder) while train is 67 % function-style (Codewars/LeetCode), so the tuning
  queries are the **stdin-style train problems**: `dev-tune` (600, every choice) and
  `dev-holdout` (600, confirmation), searched against the 5 000 train snippets. This dev set
  predicts test well: bge-small 0.067 dev vs 0.056 test, gte-modernbert 0.583 vs 0.564, Qwen3
  0.757 vs 0.752.
- **Contamination, stated openly.** jina-code-embeddings was trained on the AppsRetrieval *train*
  split (its paper), so its dev scores are inflated (0.96 vs 0.84 test). We therefore tuned the
  pre/post-processing on **Qwen3-Embedding-0.6B as an uncontaminated proxy**, kept only what did
  not hurt jina, and judged the ensemble weight on jina's dev-tune (conservative).
- **No leakage.** Only the `text` of queries and snippets is used. Dataset ids (`q5001`↔`d5001`),
  `meta_information` (problem URL on both sides) and the corpus `partition` field are never read
  by the retriever; tricks that exploit benchmark structure (one-to-one assignment, demoting
  snippets that answer known train problems, query-based hubness) were rejected on principle.
- The test split was used exactly once for our pipeline (the official run) and afterwards for
  the reporting-only baselines above.

### Ablations (tuning splits; `results/apps_ablation.md`)

Proxy model (Qwen3-Embedding-0.6B, uncontaminated), NDCG@10 dev-tune / dev-holdout:

| Step | dev-tune | dev-holdout |
|---|---|---|
| raw text, cosine only | 0.7573 | 0.7712 |
| + cleaned query (LaTeX / judge boilerplate removed) | 0.7586 | 0.7804 |
| + program-style agreement | 0.7643 | 0.7847 |
| + shared-literal overlap | 0.7675 | 0.7877 |
| + snippet-side hubness penalty | 0.7729 | 0.7899 |
| + small BM25 term (= knobs used in the final pipeline) | **0.7748** | **0.7913** |
| *dropping the examples from the query* | *0.7504* | *0.7651* |
| *second query view without examples* | *0.7587* | *0.7764* |

On jina-code-0.5b the dev scores sit at a contaminated ceiling (raw 0.959 / 0.978; final
pipeline 0.963 / 0.974) and cannot rank variants, which is why the proxy was used. On the **test
split** the same pre/post-processing + ensemble moves jina from 0.8387 to **0.8657**.

Negative results (what did not help) are in `results/apps_negative_results.md`; the model
benchmark is in `results/apps_model_benchmark.md`.

### P1 — versions (`results/p1_rebuild.md`)

One command per version (`./run.sh index-version …`); fresh caches, CPU only, gte-modernbert:

| Library | Version | Build | Snippets | Re-encoded | Time |
|---|---|---|---|---|---|
| Gladys (JS) | v5.0.4 | full | 6 424 | 6 422 | 470 s |
| Gladys | v5.1.0 (146 files changed) | incremental | 6 642 | 394 | **51 s** |
| Gladys | v5.1.1 (1 file changed) | incremental | 6 642 | 1 | **0.8 s** |
| APPS-derived | v1 | full | 2 000 | 2 000 | 154 s |
| APPS-derived | v2 (10 % mutated) | incremental | 2 000 | 176 | **16 s** |
| APPS-derived | v3 (1 % mutated) | incremental | 2 000 | 23 | **3.8 s** |

The APPS demo library (8 765 snippets, two 0.5B models): v2 with 10 % mutated snippets re-encodes
791 snippets in 255 s; v3 (5 %) 384 in 155 s.

### Bonus — all versions at once (`results/evolution_bench.md`)

5 000 APPS train solutions evolved v1 → v2 → v3 with behaviour-preserving mutations (renames,
comments, fast I/O, `main()` wrapping, reformatting) = 15 000 near-duplicate entries indexed
together; queries = 600 + 600 stdin-style train problems with rotating version hints ("latest",
"original", "in v2", none). Lineages inferred from content only (pair precision 0.88, recall 0.94).

| dev-holdout | NDCG@10 (dedup) | Right version @1 | Lineage MRR@10 | Distinct snippets in top 10 |
|---|---|---|---|---|
| flat ranking | 0.5595 | 0.4133 | 0.6901 | 4.5 |
| flat, version hint stripped | 0.5604 | 0.4150 | 0.6914 | 4.5 |
| **lineage handling (inferred)** | **0.7383** | **0.6083** | **0.7248** | **9.9** |
| lineage handling (true lineages) | 0.7460 | 0.6100 | 0.7270 | 10.0 |

(dev-tune: 0.5419 → 0.7208. NDCG *dedup* counts each lineage once; the graded variant that rewards
listing every version is in the results file with its caveat.)

## Quick start

The submission is reproduced entirely from the command line; the web UI is optional.

**Requirements:** macOS or Linux, Python ≥ 3.10, git, and **~6 GB of free disk** for models.
Optional: Node ≥ 20 + npm for the web UI (setup skips it with a warning if npm is missing);
[Ollama](https://ollama.com) for the Gladys-agent explanation.

```bash
./run.sh setup        # .venv + pinned deps, local models (and the web UI build if npm is present)
./run.sh submit       # official APPS run on the CPU → release/appsretrieval_results.json
```

**Expected time on a CPU:** setup ~44 min (mostly model downloads); a **cold** `./run.sh submit`
**~1 h 51 min** on an Apple M1 Pro (encoding 8 765 snippets and 3 765 queries with two models);
a rerun with warm embedding caches ~41 s.

`./run.sh submit` executes exactly the guidelines' pattern (`scripts/submit.py`):

```python
model = PrePostPipelineEncoder()                     # src/codesearch/submission.py
task = mteb.get_task("AppsRetrieval")
result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": 64})
task_result = list(result.task_results)[0]
json.dump(task_result.to_dict(), open("release/appsretrieval_results.json", "w"), indent=2)
```

MTEB's result cache is pointed at a fresh directory for every run so the task really recomputes.
The first run downloads the dataset and the two models (~3 GB) into `.hf_cache/` inside the
project; nothing is written outside the project folder.

### Optional: interactive demo (terminal or web UI)

```bash
./run.sh index-apps                          # APPS corpus as library "apps": v1 + synthetic v2, v3
./run.sh index-version gladys --ref v5.0.4   # a codebase at any git ref (incremental)
./run.sh index-version gladys --ref v5.1.0
./run.sh index-version gladys --ref v5.1.1
./run.sh app                                 # http://localhost:8000
./run.sh search "Given n integers, print the maximum pairwise product" --library apps
```

The **Search** page: pick the library (APPS corpus / Gladys codebase) and a version (or **All
versions** to see lineages), paste a problem or pick an example; it shows the query category and
pre-processed text, every retrieval pass with its time, the total latency, and the ranked
snippets with scores, style and shared literals. **Metrics** shows the APPS, P1 and bonus
results. The earlier Gladys code-intelligence agent is still available at **Gladys agent**. The
demo walk-through is in `docs/demo_script.md`.

### Indexing a new version (P1)

```bash
./run.sh index-version gladys --ref <tag|commit>                   # codebase from git
./run.sh index-version mylib --jsonl snippets.jsonl --version 2026-10-01   # any snippet set
./run.sh index-version apps --evolve-from v3 --version v4 --rate 0.02      # synthetic next version
```

Snippets are re-parsed only for files whose git blob changed and re-encoded only when their text
changed (content-hash embedding cache), so a new version costs time proportional to the change.

### Other commands

| Command | What it does |
|---|---|
| `./run.sh test` | unit + API + integration tests (no browser) |
| `./run.sh e2e` | Playwright browser test (APPS search, all versions, Gladys, agent, metrics) |
| `./run.sh apps-eval` | tuning-split ablations, bonus benchmark, P1 timings |
| `.venv/bin/python -m eval.apps_bench --models …` | embedding-model benchmark |
| `.venv/bin/python -m eval.apps_sweep --models qwen3-emb-0.6b` | knob sweep on the proxy |
| `./run.sh lint` | ruff + TypeScript + oxlint |

## How it works

```
query ─► pass 0: pre-process ─► pass 1: dense recall ─────────► pass 2: rerank top-100 ─► ranked
         split statement /      jina-code-0.5b  (w 1.0)          + program-style agreement      snippets
         input / output /       Qwen3-Emb-0.6B (w 0.4)           + shared literals (IDF)
         examples / notes,      Σ w·cos  =  cosine over          + BM25 (small weight)
         strip LaTeX, cat-      weighted concatenated            − snippet hubness
         egorise (stdin /       embeddings, exact search         (per-query z-scored)
         function / class)      over all snippets
```

- **Pre-processing (pass 0).** Problem statements carry judge boilerplate (`-----Input-----`,
  `$1 \le n \le 10^5$`, sample I/O). We split sections, remove LaTeX markup and rules, and
  categorise the program the query asks for. Sample I/O is kept — dropping it hurt.
- **Dense recall (pass 1).** Two free code embedders with their own instruction prefixes; a
  weighted sum of cosines is one cosine over the weighted concatenation, which is also what the
  class's `encode()` returns. Models are loaded in float32: half precision is 12× slower on this
  CPU (D-036).
- **Rerank (pass 2).** On the top 100 only: a bonus when a snippet's program form (stdin script /
  function / `class Solution`) matches the query's; the IDF-weighted overlap of string/number
  literals shared by code and statement (`"IMPOSSIBLE"`, `998244353`); a query-free hubness
  penalty (mean similarity to a snippet's nearest snippets); a small BM25 term.
- **Versions / lineages.** Every version is its own index keyed by version label or git commit;
  the all-versions view groups near-identical snippets across versions into lineages (stable key,
  identical text, or embedding cosine + identifier-agnostic token shape), collapses each lineage
  to its best-matching version, and honours version hints in the query ("original version",
  "latest", "v2").

Full write-up (problem, pipeline, tuning protocol, rejected ideas, P1 and bonus designs, known
limitations): [`docs/approach.md`](docs/approach.md). Every design decision with its alternatives
and reasons: [`docs/decisions.md`](docs/decisions.md) (APPS engine: D-033 … D-042).

## Hardware used

Apple MacBook Pro, **M1 Pro (8 CPU cores: 6 performance + 2 efficiency), 16 GB RAM**, macOS 26.3,
Python 3.10.13, torch 2.14 (CPU), transformers 5.17, sentence-transformers 6.1, mteb 2.21.8.
All numbers in this repository were measured on the CPU of this machine.

## Secondary demo: Gladys code-intelligence agent

The first-round prototype — an agent that answers plain-English, structural ("which functions
call X before Y?") and usage questions over the Gladys JavaScript codebase with verified
`file:line` citations (tree-sitter AST, call graph, domain catalog, BM25 + embeddings,
planner-driven agent loop) — still works: `./run.sh index`, then the **Gladys agent** page or
`./run.sh ask "Who calls purgeStates?"`. Its local-LLM explanation is **optional and off by
default** (generation is out of scope now): tick "LLM explanation", pass `--explain`, or set
`CODEINTEL_EXPLAIN=1`. Its design is summarised in [`docs/approach.md`](docs/approach.md)
(Appendix A).

## Credits and licences

| Component | Licence | Use |
|---|---|---|
| [jinaai/jina-code-embeddings-0.5b](https://huggingface.co/jinaai/jina-code-embeddings-0.5b) | CC-BY-NC-4.0 (non-commercial) | primary embedder |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Apache-2.0 | ensemble embedder, tuning proxy |
| [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | Apache-2.0 | Gladys codebase library, P1 timings |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | MIT | baseline; Gladys agent |
| nomic-ai/CodeRankEmbed, ibm-granite/granite-embedding-small-english-r2 | MIT / Apache-2.0 | benchmarked |
| [CoIR APPS](https://huggingface.co/datasets/CoIR-Retrieval/apps) (Hendrycks et al., 2021) | MIT | benchmark data |
| [MTEB](https://github.com/embeddings-benchmark/mteb) | Apache-2.0 | official evaluation |
| [Gladys Assistant](https://github.com/GladysAssistant/Gladys) | Apache-2.0 | codebase for versions / agent demo (read-only) |
| qwen2.5-coder via Ollama, faster-whisper | Apache-2.0 / MIT | optional Gladys-agent explanation, voice input |

jina-code-embeddings is licensed for non-commercial use, which covers this hackathon submission.
