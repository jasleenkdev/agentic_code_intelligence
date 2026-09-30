import random

import numpy as np

from src.codesearch.lineage import (
    collapse,
    infer_lineages,
    jaccard,
    members_by_lineage,
    parse_version_hint,
    shape_tokens,
)
from src.codesearch.synthetic import evolve, mutate, rename_identifiers


def test_version_hints():
    vs = ["v1", "v2", "v3"]
    h = parse_version_hint("sort the array (latest version)", vs)
    assert h.kind == "latest" and h.cleaned_query == "sort the array"
    assert parse_version_hint("sort the array in v2", vs).label == "v2"
    assert parse_version_hint("the original version of the sorter", vs).kind == "earliest"
    assert parse_version_hint("sort the array", vs).kind is None


def test_shape_tokens_ignore_renames():
    a = "n = int(input())\nprint(n * 2)\n"
    b = rename_identifiers(a, random.Random(0))
    assert jaccard(shape_tokens(a), shape_tokens(b)) == 1.0


def test_infer_lineages_groups_mutations_not_strangers():
    base = {
        "s1": "n = int(input())\nfor i in range(n):\n    print(i * i)\n",
        "s2": "s = input()\nprint(s[::-1])\n",
    }
    hist = evolve(base, ["v1", "v2"], [0.0, 1.0], seed=1)
    texts = [hist[v][k] for v in ("v1", "v2") for k in ("s1", "s2")]
    versions = ["v1", "v1", "v2", "v2"]
    # Fake embeddings: same lineage → same direction.
    emb = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    lin = infer_lineages(texts, versions, emb, cos_min=0.9, jaccard_min=0.2)
    assert lin[0] == lin[2] and lin[1] == lin[3] and lin[0] != lin[1]


def test_collapse_prefers_hinted_and_recent_versions():
    scores = np.array([0.90, 0.89, 0.50, 0.40])
    lineages = [0, 0, 1, 1]
    vrank = np.array([0, 1, 0, 1])
    labels = ["v1", "v2", "v1", "v2"]
    mem = members_by_lineage(lineages)
    none = parse_version_hint("q", ["v1", "v2"])
    hits = collapse(scores, lineages, vrank, none, labels, members_of=mem, tie_eps=0.02)
    assert [h.lineage for h in hits] == [0, 1]  # one entry per lineage
    assert hits[0].index == 1  # near-tie → most recent version
    first = parse_version_hint("q (original version)", ["v1", "v2"])
    assert collapse(scores, lineages, vrank, first, labels, members_of=mem)[0].index == 0


def test_mutate_changes_code_deterministically():
    code = "n = int(input())\nprint(n + 1)\n"
    a, ops_a = mutate(code, random.Random(3))
    b, ops_b = mutate(code, random.Random(3))
    assert a == b and ops_a == ops_b and a != code
