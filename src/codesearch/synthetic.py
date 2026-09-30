"""Deterministic, behaviour-preserving mutations of Python solutions → synthetic versions.

Used to build (a) the APPS-derived multi-version snippet sets for the P1 rebuild timings and
(b) the evolutionary-retrieval benchmark (bonus): every snippet gets a lineage whose versions
are near-identical — exactly the case that is hard to rank.
"""

from __future__ import annotations

import builtins
import io
import keyword
import random
import re
import tokenize

_RESERVED = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls", "sys", "math", "main"}
_RENAMES = {
    "n": "num",
    "m": "cnt",
    "a": "arr",
    "b": "brr",
    "s": "text",
    "t": "tests",
    "i": "idx",
    "j": "jdx",
    "k": "kval",
    "x": "xv",
    "y": "yv",
    "ans": "result",
    "res": "answer",
    "l": "lst",
    "r": "right",
    "c": "ch",
    "d": "dct",
    "q": "queue_",
}


def rename_identifiers(code: str, rng: random.Random) -> str:
    """Rename local names consistently (tokenize-based; attributes after '.' untouched)."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code
    names = []
    for i, tk in enumerate(toks):
        prev = toks[i - 1].string if i else ""
        if tk.type == tokenize.NAME and tk.string not in _RESERVED and prev != ".":
            names.append(tk.string)
    uniq = list(dict.fromkeys(names))
    rng.shuffle(uniq)
    mapping = {}
    for name in uniq[: max(1, len(uniq) // 2)]:
        new = _RENAMES.get(name, f"{name}_{rng.choice(['v', 'val', 'tmp', 'x'])}")
        if new not in uniq and new not in mapping.values():
            mapping[name] = new
    out = []
    for i, tk in enumerate(toks):
        prev = toks[i - 1].string if i else ""
        if tk.type == tokenize.NAME and tk.string in mapping and prev != ".":
            out.append((tk.type, mapping[tk.string]))
        else:
            out.append((tk.type, tk.string))
    try:
        return tokenize.untokenize(out)
    except Exception:
        return code


def add_comments(code: str, rng: random.Random) -> str:
    notes = ["read input", "main logic", "compute answer", "edge case", "output result"]
    lines = code.split("\n")
    out = [f"# {rng.choice(['Solution', 'Approach', 'Refactored solution'])}"]
    for ln in lines:
        if ln.strip() and not ln.strip().startswith("#") and rng.random() < 0.15:
            indent = ln[: len(ln) - len(ln.lstrip())]
            out.append(f"{indent}# {rng.choice(notes)}")
        out.append(ln)
    return "\n".join(out)


def fast_io(code: str, rng: random.Random) -> str:
    if "input(" not in code or "sys.stdin" in code:
        return code
    return "import sys\ninput = sys.stdin.readline\n" + code


def wrap_main(code: str, rng: random.Random) -> str:
    if re.search(r"^def main\b", code, re.MULTILINE) or "return" in code.split("\n")[0]:
        return code
    body = "\n".join(("    " + ln) if ln.strip() else ln for ln in code.split("\n"))
    return f"def main():\n{body}\n\n\nmain()\n"


def reformat(code: str, rng: random.Random) -> str:
    code = re.sub(r"(?<=[\w)\]])([+\-*/%<>]=?|==|!=)(?=[\w(\[])", r" \1 ", code)
    return re.sub(r"\n{3,}", "\n\n", code)


MUTATIONS = {
    "rename": rename_identifiers,
    "comments": add_comments,
    "fast_io": fast_io,
    "wrap_main": wrap_main,
    "reformat": reformat,
}


def mutate(code: str, rng: random.Random, n_ops: int = 2) -> tuple[str, list[str]]:
    """Apply `n_ops` distinct random mutations; returns (new code, names of applied ops)."""
    ops = rng.sample(list(MUTATIONS), k=min(n_ops, len(MUTATIONS)))
    applied = []
    for op in ops:
        new = MUTATIONS[op](code, rng)
        if new != code:
            applied.append(op)
            code = new
    return code, applied


def evolve(
    snippets: dict[str, str],
    versions: list[str],
    change_rate: list[float],
    seed: int = 7,
) -> dict[str, dict[str, str]]:
    """Version history: `{version: {snippet_id: code}}`.

    Version 0 is the original set; each later version mutates a `change_rate[v]` fraction of the
    previous version's snippets (others carry over byte-identical).
    """
    rng = random.Random(seed)
    history = {versions[0]: dict(snippets)}
    for vi in range(1, len(versions)):
        prev = history[versions[vi - 1]]
        cur = {}
        for sid, code in prev.items():
            if rng.random() < change_rate[vi]:
                cur[sid], _ = mutate(code, rng, n_ops=rng.choice([1, 2]))
            else:
                cur[sid] = code
        history[versions[vi]] = cur
    return history
