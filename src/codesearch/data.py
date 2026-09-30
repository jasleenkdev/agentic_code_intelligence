"""APPS (CoIR) data loading and the train-derived tuning splits.

Only the `text` field of queries and snippets is ever given to the retriever. The dataset's
`meta_information` carries the problem URL on BOTH sides and ids pair up (`q5001` ↔ `d5001`);
using either would be label leakage, so neither reaches any model (D-034).

Tuning discipline (D-035): the APPS test split is used only for the final official run.
All model selection uses the train split, searched against the 5 000 train snippets only.

The train split is distributed very differently from test: 67 % of train problems are
function-style (Codewars/LeetCode starter code, the function name often appears in both query and
solution), while 99 % of test problems are stdin-style (Codeforces/AtCoder, no starter code), which
is far harder. Tuning on train as-is would reward identifier matching that test does not have, so
the tuning queries are the **stdin-style** train problems (empty `starter_code`), shuffled with a
fixed seed into
  dev-tune    (600 queries) — every choice is made on this;
  dev-holdout (600 queries) — reported, used only to confirm a choice generalises;
  train-pool  (the other 3 800 train queries) — the only pairs any fitting step may learn from.
`starter_code` is read only here, to build the splits; it never reaches a model.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from functools import lru_cache

from src import config  # noqa: F401  (sets HF_HOME inside the project)

logger = logging.getLogger(__name__)

SEED = 13
N_TUNE = 600
N_HOLDOUT = 600


@dataclass
class RetrievalSet:
    """A query set against a corpus. `qrels[qid] = {doc_id: relevance}`."""

    name: str
    queries: dict[str, str]
    corpus: dict[str, str]
    qrels: dict[str, dict[str, int]]
    meta: dict = field(default_factory=dict)


def _mteb_split(split: str):
    os.environ.setdefault("MTEB_CACHE", str(config.PROJECT_ROOT / ".cache" / "mteb"))
    import mteb

    task = mteb.get_task("AppsRetrieval", eval_splits=[split])
    task.load_data()
    task.convert_v1_dataset_format_to_v2(num_proc=None)
    return task.dataset["default"][split]


@lru_cache(maxsize=2)
def load_apps(split: str) -> RetrievalSet:
    """The full APPS split exactly as MTEB evaluates it (corpus = all 8 765 snippets)."""
    d = _mteb_split(split)
    corpus = {r["id"]: r["text"] for r in d["corpus"]}
    partition = {r["id"]: r["partition"] for r in d["corpus"]}
    queries = {r["id"]: r["text"] for r in d["queries"]}
    stdin_style = {
        r["id"]: not (r["meta_information"] or {}).get("starter_code") for r in d["queries"]
    }
    qrels = {q: dict(v) for q, v in d["relevant_docs"].items() if q in queries}
    meta = {"partition": partition, "stdin_style": stdin_style}
    return RetrievalSet(f"apps-{split}", queries, corpus, qrels, meta)


@lru_cache(maxsize=1)
def dev_splits() -> dict[str, RetrievalSet]:
    """dev-tune / dev-holdout / train-pool, all against the 5 000 train snippets only."""
    full = load_apps("train")
    corpus = {k: v for k, v in full.corpus.items() if full.meta["partition"][k] == "train"}
    qids = sorted(full.qrels, key=lambda q: int(q[1:]))
    random.Random(SEED).shuffle(qids)
    stdin = [q for q in qids if full.meta["stdin_style"][q]]
    dev = set(stdin[: N_TUNE + N_HOLDOUT])
    parts = {
        "dev-tune": stdin[:N_TUNE],
        "dev-holdout": stdin[N_TUNE : N_TUNE + N_HOLDOUT],
        "train-pool": [q for q in qids if q not in dev],
    }
    out = {}
    for name, ids in parts.items():
        out[name] = RetrievalSet(
            name,
            {q: full.queries[q] for q in ids},
            corpus,
            {q: full.qrels[q] for q in ids},
        )
    logger.info(
        "dev splits: %s over %d snippets", {k: len(v) for k, v in parts.items()}, len(corpus)
    )
    return out
