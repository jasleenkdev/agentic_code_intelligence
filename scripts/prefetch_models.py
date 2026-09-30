"""Download every local model once into ./.hf_cache: the APPS pipeline's embedders, the codebase
embedder, and the Gladys agent's embeddings / reranker / speech-to-text.

Usage: python -m scripts.prefetch_models [--skip-whisper] [--skip-rerank] [--skip-codesearch]
"""

from __future__ import annotations

import argparse
import time

from src import config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-whisper", action="store_true")
    ap.add_argument("--skip-rerank", action="store_true")
    ap.add_argument("--skip-codesearch", action="store_true")
    args = ap.parse_args()
    if not args.skip_codesearch:
        from src.codesearch.encoders import PROFILES, _load_model
        from src.codesearch.submission import FINAL_CONFIG

        keys = [k for k, _ in FINAL_CONFIG.models] + [config.CODESEARCH_CODEBASE_MODEL]
        for key in dict.fromkeys(keys):
            t = time.perf_counter()
            p = PROFILES[key]
            m = _load_model(p.model_id, p.trust_remote_code)
            dim = m.encode(["print(sum(map(int, input().split())))"]).shape[1]
            print(f"codesearch  {p.model_id}: ok, dim {dim} ({time.perf_counter() - t:.1f}s)")
    t = time.perf_counter()
    from src.retrieval.embeddings import embed_texts

    v = embed_texts(["function scan() { return devices; }"])
    print(
        f"embeddings  {config.EMBED_MODEL}: ok, dim {v.shape[1]} ({time.perf_counter() - t:.1f}s)"
    )
    if not args.skip_rerank:
        t = time.perf_counter()
        from src.retrieval.search import _get_reranker

        score = _get_reranker().predict([("scan devices", "function scan() {}")])
        print(
            f"reranker    {config.RERANK_MODEL}: ok, score {float(score[0]):.2f} "
            f"({time.perf_counter() - t:.1f}s)"
        )
    if not args.skip_whisper:
        t = time.perf_counter()
        from src.speech.stt import get_whisper

        get_whisper()
        print(f"whisper     {config.WHISPER_MODEL}: ok ({time.perf_counter() - t:.1f}s)")


if __name__ == "__main__":
    main()
