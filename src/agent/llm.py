"""Provider-agnostic chat client (D-009). No paid APIs; no keys needed for the default.

- Ollama (default, detected by port 11434 or `/api` in the URL): native `/api/chat`, because only
  the native API accepts `options.num_gpu` (0 = CPU-only inference) and `format: "json"`.
- Anything else: OpenAI-compatible `/chat/completions` with `response_format: json_object`
  (Groq / Gemini free tiers work by setting LLM_BASE_URL, LLM_API_KEY, LLM_MODEL in .env).

Responses are cached on disk (`.cache/llm/`) keyed by model + messages + params, so re-running an
evaluation or demo query is instant and deterministic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field

import requests

from src import config

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """The LLM endpoint is down or every model failed — callers must degrade gracefully."""


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: float = 0.0
    cached: bool = False
    raw: dict = field(default_factory=dict)


def _is_ollama(base_url: str) -> bool:
    return ":11434" in base_url or "ollama" in base_url.lower()


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
        use_cache: bool = True,
    ) -> None:
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or config.LLM_API_KEY
        self.model = model or config.LLM_MODEL
        self.fallback_model = (
            fallback_model if fallback_model is not None else (config.LLM_FALLBACK_MODEL)
        )
        self.use_cache = use_cache
        self.ollama = _is_ollama(self.base_url)
        self._health: tuple[float, bool] | None = None

    # ------------------------------------------------------------------ health
    def _root(self) -> str:
        return self.base_url.removesuffix("/v1") if self.ollama else self.base_url

    def available(self, ttl: float = 20.0) -> bool:
        now = time.time()
        if self._health and now - self._health[0] < ttl:
            return self._health[1]
        ok = False
        try:
            if self.ollama:
                r = requests.get(self._root() + "/api/tags", timeout=2)
                names = {m["name"] for m in r.json().get("models", [])}
                ok = r.ok and (self.model in names or self.fallback_model in names)
            else:
                r = requests.get(
                    self.base_url + "/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=4,
                )
                ok = r.ok
        except (requests.RequestException, ValueError):
            ok = False
        self._health = (now, ok)
        return ok

    def installed_models(self) -> list[str]:
        if not self.ollama:
            return [self.model]
        try:
            r = requests.get(self._root() + "/api/tags", timeout=2)
            return sorted(m["name"] for m in r.json().get("models", []))
        except (requests.RequestException, ValueError):
            return []

    # ------------------------------------------------------------------ chat
    def _cache_key(self, model: str, messages: list[dict], json_mode: bool, max_tokens: int) -> str:
        blob = json.dumps(
            [model, messages, json_mode, max_tokens, config.LLM_NUM_CTX], sort_keys=True
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _call(self, model: str, messages: list[dict], json_mode: bool, max_tokens: int) -> dict:
        timeout = config.LLM_TIMEOUT_SEC
        if self.ollama:
            options = {
                "temperature": 0,
                "num_predict": max_tokens,
                "num_ctx": config.LLM_NUM_CTX,
                "seed": 7,
            }
            if config.LLM_NUM_GPU >= 0:
                options["num_gpu"] = config.LLM_NUM_GPU
            body = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": options,
                "keep_alive": "30m",
            }
            if json_mode:
                body["format"] = "json"
            r = requests.post(self._root() + "/api/chat", json=body, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return {
                "text": data["message"]["content"],
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }
        body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(
            self.base_url + "/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        return {
            "text": data["choices"][0]["message"]["content"],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

    def chat(
        self,
        messages: list[dict],
        json_mode: bool = True,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        max_tokens = max_tokens or config.LLM_MAX_TOKENS
        models = [model or self.model]
        if self.fallback_model and self.fallback_model not in models:
            models.append(self.fallback_model)
        last_exc: Exception | None = None
        for m in models:
            key = self._cache_key(m, messages, json_mode, max_tokens)
            path = config.LLM_CACHE_DIR / f"{key}.json"
            if self.use_cache and path.exists():
                data = json.loads(path.read_text())
                return LLMResponse(
                    text=data["text"],
                    model=m,
                    cached=True,
                    prompt_tokens=data.get("prompt_tokens", 0),
                    completion_tokens=data.get("completion_tokens", 0),
                )
            t0 = time.perf_counter()
            try:
                data = self._call(m, messages, json_mode, max_tokens)
            except (requests.RequestException, KeyError, ValueError) as exc:
                logger.warning("LLM call failed with %s: %s", m, exc)
                last_exc = exc
                continue
            elapsed = (time.perf_counter() - t0) * 1000
            if self.use_cache:
                config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            return LLMResponse(
                text=data["text"],
                model=m,
                elapsed_ms=round(elapsed, 1),
                prompt_tokens=data["prompt_tokens"],
                completion_tokens=data["completion_tokens"],
            )
        raise LLMUnavailable(f"all models failed: {last_exc}")

    def stream(self, messages: list[dict], max_tokens: int = 220):
        """Plain-text streaming (see `stream_chat`)."""
        return stream_chat(self, messages, max_tokens)


def _stream_ollama(client: LLMClient, model: str, messages: list[dict], max_tokens: int):
    options = {
        "temperature": 0,
        "num_predict": max_tokens,
        "num_ctx": config.LLM_NUM_CTX,
        "seed": 7,
    }
    if config.LLM_NUM_GPU >= 0:
        options["num_gpu"] = config.LLM_NUM_GPU
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
        "keep_alive": "30m",
    }
    with requests.post(
        client._root() + "/api/chat", json=body, stream=True, timeout=config.LLM_TIMEOUT_SEC
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                break


def _stream_openai(client: LLMClient, model: str, messages: list[dict], max_tokens: int):
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    with requests.post(
        client.base_url + "/chat/completions",
        json=body,
        stream=True,
        headers={"Authorization": f"Bearer {client.api_key}"},
        timeout=config.LLM_TIMEOUT_SEC,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
            if delta:
                yield delta


def stream_chat(client: LLMClient, messages: list[dict], max_tokens: int = 220):
    """Yield text pieces as the model produces them (plain text, not JSON). The full reply is
    written to the same disk cache as `chat`, so a repeated question replays instantly."""
    model = client.model
    key = client._cache_key(model, messages, False, max_tokens)
    path = config.LLM_CACHE_DIR / f"{key}.json"
    if client.use_cache and path.exists():
        yield json.loads(path.read_text())["text"]
        return
    gen = _stream_ollama if client.ollama else _stream_openai
    parts: list[str] = []
    try:
        for piece in gen(client, model, messages, max_tokens):
            parts.append(piece)
            yield piece
    except (requests.RequestException, KeyError, ValueError) as exc:
        raise LLMUnavailable(f"streaming failed: {exc}") from exc
    if client.use_cache and parts:
        config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"text": "".join(parts), "prompt_tokens": 0, "completion_tokens": 0})
        )


def parse_json_object(text: str) -> dict:
    """Parse the first JSON object in `text` (tolerates code fences and trailing prose)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    obj = json.loads(text[start : i + 1])
                    break
        else:
            raise
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object")
    return obj


_DEFAULT: LLMClient | None = None


def get_llm() -> LLMClient:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LLMClient()
    return _DEFAULT
