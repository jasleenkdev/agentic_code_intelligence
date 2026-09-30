# Demo script (video + hands-on round)

Target length: 5–7 minutes. Everything runs on the laptop's CPU; no network needed once set up.

## Before recording

```bash
./run.sh app          # API + web UI on http://localhost:8000 (builds the UI if stale)
```

- Libraries must be indexed once: `./run.sh index-apps` (APPS corpus v1 + synthetic v2/v3) and
  `./run.sh index-version gladys --ref v5.0.4`, `--ref v5.1.0`, `--ref v5.1.1`.
- The app warms the APPS library and both query encoders at start (~15 s); wait for it before
  recording. New queries then take ~0.3–0.8 s; repeated queries ~20 ms (the pass timeline says
  "query embeddings from cache" — mention it rather than passing it off as model speed).
- Browser at 100 % zoom, dark theme, window ~1440 px wide. Close other CPU-heavy apps.

## 1. The problem and the score (45 s)

1. Open **Metrics**. Point at the two big numbers: official **test NDCG@10 and MRR@10** from
   `mteb.evaluate` on AppsRetrieval, and the full test-run time on CPU.
2. Scroll to **Test split vs baselines**: our pipeline vs published models (bge-small, e5,
   Voyage-code-002, gte-modernbert, …) and our own single-model baseline.
3. One sentence on discipline: *every choice was tuned on stdin-style APPS train problems; the test
   split was used once.*

## 2. Live retrieval on APPS (2 min)

Open **Search** (home). Library = **APPS corpus**, version = latest.

1. Click the example **"Max subarray sum"**. Show:
   - the **latency banner** (ms to rank 8 765 snippets on CPU);
   - **Query category**: *stdin program*, topics, sections found (statement / input / output);
   - **Retrieval passes**: pre-process → dense recall → rerank, each with its time;
   - the ranked snippets: #1 is a Kadane-style solution; scores; style badge; shared literals.
2. Click **"Shortest path or -1"**: point at the BFS solutions and the shared literal `-1`.
3. **Paste a fresh problem** in APPS style (hands-on judges will do this), e.g.:

   > You are given a string s consisting of lowercase letters. Find the length of the longest
   > substring that contains no repeated characters.
   > -----Input----- The only line contains s (1 ≤ |s| ≤ 10^5).
   > -----Output----- Print one integer.

   Press ⌘/Ctrl+Enter; open **pre-processed text** to show the LaTeX/boilerplate cleaning.
4. Terminal alternative (same engine):
   `./run.sh search "Given n integers, print the maximum pairwise product" --library apps`

## 3. Versions (P1) (1 min)

1. In the version selector choose **v1**, rerun a query, then **v3**: results come from that
   version's index.
2. Terminal: index a new version live and show that only changed snippets are re-encoded:

   ```bash
   ./run.sh index-version apps --evolve-from v3 --version v4 --rate 0.02
   ```

   The JSON line reports `encoded` (only the mutated snippets) vs `reused`, and the seconds.
3. Metrics → **P1 · full vs incremental rebuild**: Gladys v5.0.4 full build vs v5.1.0 / v5.1.1
   incremental, APPS-derived v1 full vs v2 / v3 incremental.

## 4. All versions at once (bonus) (1 min)

1. Version selector → **All versions (lineages)**. Click the example **"Version hint"**
   (`… (original version)`).
2. Show: the hint is detected and stripped; each hit is **one lineage** with chips for every
   version (`v1 =`, `v2 ≠`, …) and their scores; the chosen version follows the hint.
3. Remove the hint and search again: near-ties go to the newest version.
4. Metrics → **Bonus** table: flat ranking vs lineage handling (NDCG@10, exact-version@1).

## 5. Codebase mode: Gladys (45 s)

1. Switch library to **Gladys codebase**; choose v5.1.1; example **"Purge old states"**.
2. All versions + example **"Version hint"** (`register a proxy service … (latest version)`): the
   function changed in v5.1.1 (`registerProxyService` made idempotent) shows both versions.
3. Mention the secondary Gladys agent at **Gladys agent** (structural call-graph questions), which
   is unchanged from the first round; its LLM explanation is now optional and off by default.

## 6. Close (15 s)

Reproduce everything: `./run.sh setup && ./run.sh submit` → `release/appsretrieval_results.json`.

## Fallbacks

- UI down → the terminal `./run.sh search …` shows the same category, passes and latency.
- A library missing → `./run.sh index-apps` (the APPS corpus embeddings are cached after
  `./run.sh submit`, so this is fast).
