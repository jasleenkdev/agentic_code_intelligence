# APPS test split: submitted pipeline vs baselines (reporting only)

Measured rows use the same metric code as MTEB (NDCG@10, MRR@10) on the 3 765 test
queries against the 8 765-snippet corpus; they were run after the pipeline was frozen.

| System | NDCG@10 | MRR@10 | Source |
|---|---|---|---|
| bge-small-en-v1.5 (33M) | 0.0560 | — | published: Granite-R2 paper, Table 11 |
| bge-small, raw text, cosine only | 0.0564 | 0.0482 | ours: measured on this machine (single model baseline) |
| OpenAI text-embedding-ada-002 (API) | 0.0870 | — | published: CodeXEmbed paper, Table 1 |
| e5-base-v2 | 0.1150 | — | published: CodeXEmbed paper, Table 1 |
| Voyage-code-002 (API) | 0.2650 | — | published: CodeXEmbed paper, Table 1 |
| SFR-Embedding-Code-400M | 0.4860 | — | published: CodeXEmbed paper, Table 1 |
| CodeSage-large-v2 (1.3B) | 0.5050 | — | published: CodeXEmbed paper, Table 1 |
| gte-modernbert-base (149M) | 0.5640 | — | published: Granite-R2 paper, Table 11 |
| Qwen3-Embedding-0.6B | 0.7520 | — | published: jina-code-embeddings paper, Table 2 |
| jina-code-0.5b, raw text, cosine only | 0.8387 | 0.8081 | ours: measured on this machine (single model baseline) |
| jina-code-embeddings-0.5b | 0.8420 | — | published: jina-code-embeddings paper, Table 2 |
| Submitted pipeline (PrePostPipelineEncoder) | 0.8657 | 0.8380 | ours: official mteb.evaluate run (release/appsretrieval_results.json) |
| jina-code-embeddings-1.5b | 0.8660 | — | published: jina-code-embeddings paper, Table 2 |
