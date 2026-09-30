# APPS: things we tried that did not make the final pipeline

All measured on dev-tune (600 stdin-style APPS train problems vs the 5 000 train snippets).

| Idea | Setting | dev-tune NDCG@10 | Baseline | Verdict |
|---|---|---|---|---|
| Query adapter (residual linear map on frozen query embeddings, InfoNCE over train snippets, trained on the 3 800 train-pool pairs) | bge-small, 20 / 50 / 100 epochs | 0.0695 / 0.0635 / 0.0551 | 0.0670 | Dropped: train-pool is 88 % function-style problems, so the map learns identifier matching that stdin-style test problems do not have; gains vanish or turn negative with more training. |
| Query-based hubness penalty (reference = train queries) | — | not run | — | Rejected on principle (D-037): reference queries have their answers in the corpus, so it would demote known answers (leakage). Replaced by a query-free, snippet-side hubness estimate. |
| One-to-one query↔snippet assignment, demoting snippets that answer known train problems | — | not run | — | Rejected on principle (D-034): benchmark artefacts that would not work on a real snippet library. |
| BM25 over code tokens as a rerank feature | gte-modernbert, w = 0.2 | 0.512 | 0.592 | Harmful at meaningful weights; stdin solutions use short variable names, so word overlap with the statement is noise. |
