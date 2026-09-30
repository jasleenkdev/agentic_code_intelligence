# Slide content outline (for the organisers' PPT template)

Numbers come from `release/` and `results/` (all measured on an Apple M1 Pro CPU). Suggested
10–12 slides; speaker notes in *italics*.

---

## 1. Title

**Multi-pass code retrieval on CPU — CoIR APPS**
Team name · Samsung PRISM GenAI Hackathon 2026–27 · theme "Agentic Code Intelligence 01"
Headline: **APPS test NDCG@10 0.8657 · MRR@10 0.8380 — CPU only, free local models**

## 2. Problem

- Natural-language query → rank thousands of code snippets; "likely cannot be solved in a single
  pass". Retrieval only, no generation.
- Screening: MTEB `AppsRetrieval` (CoIR APPS), test split: 3 765 programming problems vs 8 765
  Python solutions, exactly one correct solution each.
- Constraints we kept: CPU only (float32, no GPU/MPS), free open models, no paid APIs, no tuning on
  the test split.

*Queries are full contest statements (story, input/output format, samples, LaTeX); snippets are
terse solutions with one-letter variables — very little lexical overlap.*

## 3. Approach at a glance (pipeline diagram)

```
query ─► pass 0 pre-process ─► pass 1 dense recall ─► pass 2 rerank top-100 ─► ranked snippets
         sections, LaTeX,      jina-code-0.5b (1.0)   + program-style agreement
         program category      + Qwen3-Emb-0.6B (0.4) + shared literals (IDF)
                               exact search            − hubness, + small BM25
```

- The official MTEB class implements MTEB's `index()` / `search()` protocol → **the whole
  multi-pass pipeline runs inside the official evaluation**, not just `encode()`.

## 4. Models (benchmarked on the tuning split, CPU)

Table from `results/apps_model_benchmark.md`: bge-small 0.067, granite-small-r2 0.147,
CodeRankEmbed 0.230, gte-modernbert 0.583, Qwen3-Embedding-0.6B 0.757, jina-code-0.5b 0.959¹
(dev-tune NDCG@10) + encode time per 5k snippets.

¹ trained on the APPS train split → contaminated on train-derived dev sets (0.84 on test).

*Why these two: best APPS models that fit a laptop CPU; complementary families (Qwen2.5-coder
based vs Qwen3 based, different training data).*

## 5. Pre-processing and categorisation

- **Query**: split statement / input / output / examples / notes; strip `$…$`, `\le`, `-----`;
  categorise *stdin program / function / class method*, multi-test, interactive, topic tags.
- **Snippet**: categorise stdin script / function / `class Solution`; literal extraction
  (strings, big numbers like `998244353`, `10**9+7` → normalised).
- Finding: **keep the sample I/O** in the query (dropping it cost −0.008 / −0.015 holdout);
  comment/docstring stripping of code was within noise (−0.0015 to +0.004), not adopted.

## 6. Multi-pass retrieval and post-processing

- Pass 1: weighted cosine ensemble = one cosine over weighted concatenated embeddings.
- Pass 2 (top 100): program-style agreement, shared-literal IDF overlap, query-free hubness
  penalty, small BM25; per-query z-scoring makes weights comparable.
- Example (UI screenshot): the demo query *"… determine whether you can delete at most one
  character so that s becomes a palindrome … print YES or NO"* → top 3 are all YES/NO
  palindrome-style solutions, each with the shared literals `YES` / `NO` highlighted; the pass
  timeline shows pre-process → dense recall (~1 s on CPU) → rerank (~2 ms).
- Honest note: on jina's (contaminated) dev set the rerank changes few queries (8 up, 3 down on
  dev-holdout); its value shows on the proxy (+0.016) and on test (jina raw 0.8387 → 0.8657
  together with the ensemble).

## 7. Tuning discipline (what judges will ask)

- Test split is 99 % stdin-style, train only 33 % → tuning queries = **stdin-style train
  problems** (dev-tune 600 / dev-holdout 600) vs 5 000 train snippets. Dev ≈ test for
  uncontaminated models (bge 0.067 vs 0.056, gte 0.583 vs 0.564, Qwen3 0.757 vs 0.752).
- jina's contamination handled explicitly: knobs tuned on the Qwen3 proxy, used at half weight;
  ensemble weight judged on jina's dev (conservative).
- Never used: ids, URLs in metadata, partition flags, one-to-one assignment, query-based hubness.

## 8. Ablations

Table from `results/apps_ablation.md` (Qwen3 proxy ladder: raw 0.757 → clean 0.759 → +style 0.764
→ +literal 0.768 → +hubness 0.773 → +BM25 0.775; holdout 0.771 → 0.791) and the test numbers vs
baselines (`results/apps_test_baselines.md`): jina raw 0.8387 → our pipeline 0.8657 (+0.027),
MRR 0.8081 → 0.8380; bge-small 0.0564 (= published 0.056, validates the metric code).

## 9. P1 — retrieval across versions

- Indexes keyed by version (git commit / label); content-addressed caches: git-blob-keyed chunks,
  sha1-keyed embeddings → a new version re-processes only what changed.
- Table from `results/p1_rebuild.md`: Gladys v5.0.4 full 470 s → v5.1.0 incremental 51 s
  (146 indexed files changed, 394 snippets re-encoded) → v5.1.1 0.8 s (1 file, 1 snippet);
  APPS-derived 2 000 snippets: full 154 s vs 16 s (10 % changed) and 3.8 s (1 %).
- One command: `./run.sh index-version gladys --ref v5.1.1`.

## 10. Bonus — evolutionary retrieval

- All versions indexed together; lineages from stable keys, identical text, or embedding cosine +
  identifier-agnostic token shape; collapse to one hit per lineage; version hints ("original",
  "latest", "v2") and recency tie-break.
- Synthetic benchmark: 5 000 APPS train solutions evolved v1→v2→v3 with behaviour-preserving
  mutations (15 000 entries). Table from `results/evolution_bench.md`: flat vs lineage handling
  (dev-holdout): NDCG@10 dedup 0.56 → 0.74, right version @1 0.41 → 0.61, distinct snippets in the top
  10: 4.5 → 9.9; inferred lineages within 0.01 of true lineages.

## 11. Speed (demo)

- Live query latency over the 8 765-snippet APPS library: **0.3–0.8 s for a new query** (two 0.5B
  encoders on CPU; ~20 ms when the query embedding is cached); Gladys all versions (19 708
  entries, gte-modernbert): **74 ms**. Model loading (~13 s) happens once at app start.
- Full official test run, cold on the CPU: **1 h 51 min** (55 min snippet encoding, 56 min
  queries + search); 41 s with warm embedding caches; the clean-copy rerun reproduced all 149
  metrics exactly.
- Screenshot: Search page with latency banner, query category, pass timeline, ranked snippets.

## 12. Reproduce / summary

- `./run.sh setup && ./run.sh submit` → `release/appsretrieval_results.json`.
- `./run.sh app` → demo; `docs/demo_script.md`.
- What worked: strong code embedder + complementary ensemble; program-style agreement; literal
  overlap; stdin-matched tuning set. What didn't: query adapter, BM25 at meaningful weight,
  dropping examples, a second query view; code normalisation was neutral.
