# Pipeline sweep (coordinate ascent on dev-tune; dev-holdout only for the chosen setting)

Chosen: `{'models': (('qwen3-emb-0.6b', 1.0),), 'query_views': (('full', 1.0), ('no_examples', 1.0)), 'snippet_view': 'raw', 'first_pass_k': 100, 'w_literal': 0.2, 'w_bm25': 0.05, 'w_style': 1.0, 'w_hub': 0.25, 'hub_k': 10, 'top_k': 100}`

dev-tune NDCG@10 **0.7797**, MRR@10 0.7416; dev-holdout NDCG@10 **0.7924**, MRR@10 0.7555

| Knob | Value | dev-tune NDCG@10 | dev-tune MRR@10 |
|---|---|---|---|
| start | `(('qwen3-emb-0.6b', 1.0),)` | 0.7586 | 0.0000 |
| query_views | `(('full', 1.0),)` | 0.7586 | 0.7151 |
| query_views | `(('raw', 1.0),)` | 0.7573 | 0.7136 |
| query_views | `(('no_examples', 1.0),)` | 0.7504 | 0.7080 |
| query_views | `(('full', 1.0), ('no_examples', 1.0))` | 0.7587 | 0.7185 |
| query_views | `(('full', 1.0), ('statement', 0.5))` | 0.7519 | 0.7063 |
| snippet_view | `'raw'` | 0.7587 | 0.7185 |
| snippet_view | `'normalised'` | 0.7572 | 0.7151 |
| w_style | `0.0` | 0.7587 | 0.7185 |
| w_style | `0.25` | 0.7634 | 0.7233 |
| w_style | `0.5` | 0.7649 | 0.7246 |
| w_style | `1.0` | 0.7662 | 0.7259 |
| w_hub | `0.0` | 0.7662 | 0.7259 |
| w_hub | `0.25` | 0.7699 | 0.7296 |
| w_hub | `0.5` | 0.7590 | 0.7152 |
| w_hub | `1.0` | 0.7164 | 0.6678 |
| w_literal | `0.0` | 0.7699 | 0.7296 |
| w_literal | `0.05` | 0.7742 | 0.7353 |
| w_literal | `0.1` | 0.7757 | 0.7364 |
| w_literal | `0.2` | 0.7782 | 0.7402 |
| w_bm25 | `0.0` | 0.7782 | 0.7402 |
| w_bm25 | `0.05` | 0.7797 | 0.7416 |
| w_bm25 | `0.1` | 0.7727 | 0.7329 |
