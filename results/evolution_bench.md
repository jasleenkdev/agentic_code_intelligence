# Evolutionary retrieval benchmark (synthetic APPS versions)

Model `qwen3-emb-0.6b`; 15000 entries (8824 distinct texts) over versions v1–v3; lineages inferred from content: 5317 (true 5000), pair precision 0.8798, pair recall 0.9409 (cos ≥ 0.9, shape-Jaccard ≥ 0.3); lineage inference 5.7 s.

Relevance: wanted version of the target lineage = 2, other versions = 1. Queries = stdin-style APPS train problems with a rotating version hint.

- **NDCG@10 dedup** — each lineage counts once (later versions of an already-listed
  lineage score 0); ideal = the wanted version at rank 1. This is the task's metric:
  a user wants the right version of the right snippet, not three copies of it.
- **NDCG@10 graded** — every version of the target lineage relevant; it *rewards* listing
  near-duplicates, so collapse is penalised by construction (shown for completeness).
- **Exact version @1** — the top hit is the wanted version (identical texts count).
- **Distinct lineages @10** — how many different snippets the top 10 shows.

| Set | System | NDCG@10 dedup | Exact version @1 | Lineage MRR@10 | Distinct lineages @10 | NDCG@10 graded |
|---|---|---|---|---|---|---|
| dev-tune | flat | 0.5419 | 0.4017 | 0.6629 | 4.4 | 0.6532 |
| dev-tune | flat+strip | 0.5458 | 0.4067 | 0.6654 | 4.4 | 0.6577 |
| dev-tune | lineage | 0.7208 | 0.5883 | 0.7040 | 9.9 | 0.4189 |
| dev-tune | lineage-oracle | 0.7335 | 0.5917 | 0.7100 | 10.0 | 0.4236 |
| dev-holdout | flat | 0.5595 | 0.4133 | 0.6901 | 4.5 | 0.6695 |
| dev-holdout | flat+strip | 0.5604 | 0.4150 | 0.6914 | 4.5 | 0.6722 |
| dev-holdout | lineage | 0.7383 | 0.6083 | 0.7248 | 9.9 | 0.4327 |
| dev-holdout | lineage-oracle | 0.7460 | 0.6100 | 0.7270 | 10.0 | 0.4346 |
