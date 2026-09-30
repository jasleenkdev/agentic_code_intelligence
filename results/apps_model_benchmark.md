# APPS embedding-model benchmark (train-derived tuning splits)

Corpus = the 5 000 APPS **train** snippets; queries = dev-tune / dev-holdout (600
stdin-style train problems each, seed 13, D-035). The test split is not used here. CPU
only, Apple M1 Pro. Encode seconds are wall-clock for texts not already in the embedding
cache (0 = everything was cached from an earlier run).

¹ jina-code-embeddings was trained on the AppsRetrieval **train** split (per its paper's
training-data list), so its dev scores are contaminated (0.96 vs 0.84 published on test);
for every other model the dev score tracks the published test score closely (D-040).

| Model | Params | Max seq | dev-tune NDCG@10 | dev-tune MRR@10 | R@100 | dev-holdout NDCG@10 | dev-holdout MRR@10 | Published test NDCG@10 | Doc encode s (5k) | Query encode s (1.2k) |
|---|---|---|---|---|---|---|---|---|---|---|
| jina-code-0.5b¹ (`jinaai/jina-code-embeddings-0.5b`) | 494M | 1024 | 0.9586 | 0.9469 | 0.998 | 0.9776 | 0.9699 | 0.842 | 575.4 | 431 |
| qwen3-emb-0.6b (`Qwen/Qwen3-Embedding-0.6B`) | 596M | 1024 | 0.7573 | 0.7136 | 0.985 | 0.7712 | 0.7282 | 0.752 | 920.3 | 788 |
| gte-modernbert (`Alibaba-NLP/gte-modernbert-base`) | 149M | 1024 | 0.5825 | 0.5367 | 0.912 | 0.5542 | 0.5116 | 0.564 | 364.5 | 212 |
| coderankembed (`nomic-ai/CodeRankEmbed`) | 137M | 512 | 0.2295 | 0.1998 | 0.628 | 0.2556 | 0.2234 | — | 252.3 | 147 |
| granite-small-r2 (`ibm-granite/granite-embedding-small-english-r2`) | 47M | 1024 | 0.1469 | 0.1203 | 0.508 | 0.1706 | 0.1465 | — | 105.1 | 72 |
| bge-small (`BAAI/bge-small-en-v1.5`) | 33M | 512 | 0.0670 | 0.0572 | 0.232 | 0.0738 | 0.0617 | 0.056 | 0.0 | 25 |

Not benchmarked: `jina-embeddings-v2-base-code` (its remote code fails under
transformers 5: removed `find_pruneable_heads_and_indices`, missing `is_decoder` config
defaults), `SFR-Embedding-Code-400M_R` (stopped to save CPU time; published test 0.486,
far below the leaders).
