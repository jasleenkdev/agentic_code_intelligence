"""Generate paraphrased plain-English benchmark queries with NO lexical overlap with their target.

Targets are chosen programmatically (seeded) from documented functions not used by the original
gold set; the label is the target's location, exactly as for the original semantic queries. The
local LLM only *writes the question* from the target's code; a validator then rejects any
question that
- contains a word whose stem equals the stem of any identifier or identifier sub-token in the
  target function (e.g. `getDeviceFeature` forbids get/device/feature/getdevicefeature), or any
  word of the file path, or
- shares any word bigram with the target's doc comment.
Rejected questions are retried with the violating words fed back (up to 3 attempts).

Usage: python -m eval.generate_paraphrases [--n 18]   → eval/gold_paraphrased.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from src import config
from src.agent.llm import LLMClient, LLMUnavailable, parse_json_object
from src.indexer import store
from src.retrieval.bm25 import JS_NOISE

SEED = 7
OUT = Path(__file__).parent / "gold_paraphrased.json"
GOLD = Path(__file__).parent / "gold.json"
STOP = frozenset(
    """a an the is are was be been of in on to for from with by at as and or not no it its this
    that these those which who whom what where when how why do does did we our us you your i my
    me can could would should will there here into onto than then so if any all some each every
    one two find show code part logic function method place where's which's""".split()
)
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

JUDGE = """A developer typed this search question to find some code:
Question: {question}
The code's documentation says it does this: {doc}
Is this question clearly asking for code that does what the documentation says (same action on
the same kind of thing), specific enough that this code would be a good answer?
Reply as JSON {{"match": true}} or {{"match": false}}."""

PROMPT = """You write search questions for a code-search benchmark.
Below is a JavaScript function from a smart-home assistant. Write ONE question (8 to 22 words) that
a developer who has never seen this code would ask to find it, describing WHAT it does in plain,
everyday English.
Rules:
- Do NOT use any of these forbidden words, nor their plural or verb forms: {forbidden}
- Do not mention file names, function names, variable names or code syntax.
- Use synonyms and describe the behaviour instead.
Reply as JSON {{"question": "..."}}.

Doc comment: {doc}
Code:
```js
{code}
```"""


REWRITE = """Rewrite this question so it keeps exactly the same meaning but does NOT contain any
of these words or phrases (nor their plural/verb forms): {words}
Replace each one with a different everyday word or a short description.
Question: {question}
Reply as JSON {{"question": "..."}}."""


def stem(w: str) -> str:
    """Light, consistent stemmer: plural/verb suffixes off, then a trailing 'e' off, so
    feature/features/featured and manage/manager/managing share a stem."""
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and (w.endswith(("ches", "shes")) or (w.endswith("es") and w[-3] in "sxz")):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    for suf in ("ing", "ers", "er", "ed"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    return w[:-1] if len(w) > 3 and w.endswith("e") else w


def stems(w: str) -> set[str]:
    """Stem plus its un-doubled variant (polling → poll / pol, running → runn / run)."""
    s = stem(w)
    out = {s}
    if len(s) > 3 and s[-1] == s[-2]:
        out.add(s[:-1])
    return out


def identifier_words(text: str) -> set[str]:
    out: set[str] = set()
    for ident in re.findall(r"[A-Za-z_$][\w$]*", text):
        out.add(ident.lower())
        for part in ident.split("_"):
            out.update(p.lower() for p in _CAMEL.findall(part))
    return {w for w in out if len(w) >= 3}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bigrams(text: str) -> set[tuple[str, str]]:
    """Content-word bigrams only: 'as a' / 'it as' are not doc-comment phrases."""
    ws = words(text)
    return {
        (a, b)
        for a, b in zip(ws, ws[1:], strict=False)
        if a not in STOP and b not in STOP and len(a) >= 3 and len(b) >= 3
    }


def validate(question: str, forbidden_stems: set[str], doc_bigrams: set) -> list[str]:
    """Return violations (empty list = valid)."""
    problems = []
    ws = words(question)
    if not 6 <= len(ws) <= 30:
        problems.append(f"length {len(ws)} words")
    for w in ws:
        if w not in STOP and len(w) >= 3 and stems(w) & forbidden_stems:
            problems.append(w)
    for bg in bigrams(question) & doc_bigrams:
        problems.append(" ".join(bg))
    return sorted(set(problems))


def _clean_doc(doc: str) -> str:
    m = re.search(r"@description\s+(.+?)(?:\s@\w+|$)", doc or "")
    return re.sub(r"\s+", " ", (m.group(1) if m else doc or "")).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--max-candidates", type=int, default=60)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument(
        "--model",
        default=config.LLM_MODEL,
        help="Ollama model that writes/judges the questions (offline, one-off)",
    )
    args = ap.parse_args()
    conn = store.connect(config.DB_PATH, readonly=True)
    used = {
        (g["file_path"], g["start_line"])
        for q in json.loads(GOLD.read_text())["queries"]
        for g in q["gold"]
    }
    rows = conn.execute(
        "SELECT s.name, s.qualname, s.start_line, s.end_line, s.doc, f.path FROM symbols s"
        " JOIN files f ON f.id = s.file_id WHERE s.doc LIKE '%@description%' AND s.kind IN"
        " ('function','method','arrow') AND (s.end_line - s.start_line) BETWEEN 8 AND 60"
        " ORDER BY f.path, s.start_line"
    ).fetchall()
    by_dir: dict[str, list] = {}
    for r in rows:
        if (r["path"], r["start_line"]) in used or len(_clean_doc(r["doc"])) < 20:
            continue
        by_dir.setdefault("/".join(r["path"].split("/")[:3]), []).append(r)
    rng = random.Random(SEED)
    dirs = sorted(by_dir)
    rng.shuffle(dirs)
    candidates = [rng.choice(by_dir[d]) for d in dirs][: args.max_candidates]

    llm = LLMClient(model=args.model, fallback_model="")
    if not llm.available():
        raise SystemExit("Ollama is not reachable; start it to generate paraphrases.")
    accepted, rejected = [], []
    for r in candidates:
        if len(accepted) >= args.n:
            break
        lines = (config.REPO_ROOT / r["path"]).read_text(errors="replace").split("\n")
        code = "\n".join(lines[r["start_line"] - 1 : r["end_line"]])
        doc = _clean_doc(r["doc"])
        # JS keywords (return, function, async…) are not identifiers; BM25 drops them too.
        forbidden = (identifier_words(code) | identifier_words(r["path"])) - JS_NOISE - STOP
        forbidden_stems = {v for w in forbidden for v in stems(w)}
        doc_bigrams = bigrams(doc)
        shown = sorted(forbidden, key=lambda w: (-len(w), w))[:120]
        messages = [
            {
                "role": "user",
                "content": PROMPT.format(
                    forbidden=", ".join(shown), doc=doc, code="\n".join(code.split("\n")[:45])
                ),
            }
        ]
        question, problems, text = None, ["no reply"], ""
        for attempt in range(1, 7):  # noqa: B007 — `attempt` is reported after the loop
            try:
                text = llm.chat(messages, max_tokens=80).text
                question = str(parse_json_object(text).get("question", "")).strip()
            except (LLMUnavailable, ValueError):
                question = ""
            problems = validate(question, forbidden_stems, doc_bigrams) if question else ["empty"]
            if not problems or not question:
                if not question:
                    continue
                break
            # Focused, stateless rewrite: only the offending words, nothing else to juggle.
            messages = [
                {
                    "role": "user",
                    "content": REWRITE.format(question=question, words=", ".join(problems)),
                }
            ]
        judged = None
        if not problems:
            try:
                verdict = parse_json_object(
                    llm.chat(
                        [{"role": "user", "content": JUDGE.format(question=question, doc=doc)}],
                        max_tokens=20,
                    ).text
                )
                judged = bool(verdict.get("match"))
            except (LLMUnavailable, ValueError):
                judged = False
            if not judged:
                problems = ["fidelity judge: question does not match the documented behaviour"]
        entry = {
            "target": f"{r['path']}:{r['start_line']}",
            "doc": doc,
            "question": question,
            "attempts": attempt,
            "violations": problems,
        }
        if problems:
            rejected.append(entry)
            print(flush=True)
            print(f"REJECT {entry['target']}: {question!r} {problems}")
            continue
        accepted.append(
            {
                "id": f"par-{len(accepted) + 1:02d}",
                "type": "semantic",
                "set": "paraphrased",
                "query": question,
                "gold": [
                    {
                        "file_path": r["path"],
                        "start_line": r["start_line"],
                        "end_line": r["end_line"],
                    }
                ],
                "source": f"paraphrase of JSDoc of {r['qualname']} "
                "(LLM-written, lexically validated)",
                "attempts": attempt,
                "forbidden_words": len(forbidden),
            }
        )
        print(f"ACCEPT {accepted[-1]['id']} {entry['target']}: {question}")
    OUT.write_text(
        json.dumps(
            {
                "seed": SEED,
                "labels": "targets chosen programmatically; questions LLM-written and validated "
                "to share no identifier stem or doc-comment bigram with the target",
                "model": llm.model,
                "queries": accepted,
                "rejected": rejected,
            },
            indent=2,
        )
    )
    print(f"accepted {len(accepted)}, rejected {len(rejected)} → {args.out}")


if __name__ == "__main__":
    main()
