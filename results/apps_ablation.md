# APPS pipeline ablations (train-derived tuning splits, D-035)

Queries: dev-tune / dev-holdout = 600 + 600 stdin-style APPS **train** problems; corpus =
the 5 000 train snippets. Hubness is snippet-side (query-free). The test split is not
used. Each row changes one thing relative to the row it names.

| Variant | What changes | dev-tune NDCG@10 | dev-tune MRR@10 | dev-tune R@10 | dev-holdout NDCG@10 | dev-holdout MRR@10 |
|---|---|---|---|---|---|---|
| bge-small raw | small general embedder, raw text | 0.0670 | 0.0572 | 0.098 | 0.0738 | 0.0617 |
| gte-modernbert raw | 149M general embedder, raw text | 0.5825 | 0.5367 | 0.727 | 0.5542 | 0.5116 |
| q0 qwen3 raw | proxy model (uncontaminated), raw text, cosine only | 0.7573 | 0.7136 | 0.893 | 0.7712 | 0.7282 |
| q1 +clean | q0 + LaTeX/boilerplate-cleaned query | 0.7586 | 0.7151 | 0.893 | 0.7804 | 0.7412 |
| q1b +2 views | q1 + second view without examples (not kept: +0.0001 tune, −0.004 holdout) | 0.7587 | 0.7185 | 0.882 | 0.7764 | 0.7373 |
| q1c no examples | q1 but examples dropped from the query (hurts) | 0.7504 | 0.7080 | 0.882 | 0.7651 | 0.7265 |
| q1d normalised code | q1 with comment/docstring-free snippets (mixed: −0.0015 with two views in the sweep; within noise, not adopted) | 0.7630 | 0.7204 | 0.895 | 0.7828 | 0.7441 |
| q2 +style | q1 + program-style agreement (w 0.5) | 0.7643 | 0.7210 | 0.898 | 0.7847 | 0.7456 |
| q3 +literal | q2 + shared-literal overlap (w 0.1) | 0.7675 | 0.7251 | 0.898 | 0.7877 | 0.7492 |
| q4 +hubness | q3 + snippet-side hubness penalty (w 0.125) | 0.7729 | 0.7322 | 0.898 | 0.7899 | 0.7505 |
| q5 +bm25 = half knobs | q4 + BM25 (w 0.025) | 0.7748 | 0.7343 | 0.900 | 0.7913 | 0.7515 |
| q6 full knobs | proxy sweep optimum (knobs ×2) | 0.7757 | 0.7353 | 0.900 | 0.7918 | 0.7516 |
| j0 jina raw | base model, raw text, cosine only (dev contaminated, D-040) | 0.9586 | 0.9469 | 0.993 | 0.9776 | 0.9699 |
| j1 jina +clean +half knobs | j0 + pre/post-processing | 0.9569 | 0.9446 | 0.993 | 0.9750 | 0.9664 |
| e0 jina+qwen raw | j0 + Qwen3 at 0.4 (ensemble via weighted concatenation) | 0.9634 | 0.9533 | 0.993 | 0.9757 | 0.9674 |
| final | the submitted pipeline | 0.9629 | 0.9526 | 0.993 | 0.9737 | 0.9647 |
