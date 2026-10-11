"""A local, free stand-in for the Anthropic client: same call, served by Ollama.

``src.news.label`` builds one request per article and reads the response's
``content`` / ``stop_reason`` / ``model`` / ``usage``. This adapter accepts
that same request and returns a response with those same attributes, so the
prompt, the JSON schema, parsing, resuming and the quote check are shared by
both backends, and labels from either compare like with like.

Setup (once):

    brew install ollama && ollama serve      # or the Ollama app
    ollama pull qwen3:8b

Ollama constrains decoding to the JSON schema (``format``), so as with the
API the model cannot answer outside the guide's categories. Unlike the API,
temperature can be pinned to 0 with a fixed seed, so reruns are repeatable.
"""

from __future__ import annotations

from types import SimpleNamespace

import requests

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"
# Guide (~2k tokens) + the longest article (~3.5k) + the answer, with room.
NUM_CTX = 12288
TIMEOUT = 600


class OllamaError(RuntimeError):
    """Ollama unreachable, model missing, or a malformed reply."""


class _Messages:
    def __init__(self, client: "OllamaClient"):
        self._client = client

    def create(self, *, model, system, messages, output_config, max_tokens=None, **_):
        """Translate an Anthropic Messages request into one Ollama /api/chat call."""
        c = self._client
        system_text = "".join(b["text"] for b in system) if isinstance(system, list) else system
        chat = [{"role": "system", "content": system_text},
                *({"role": m["role"], "content": m["content"]} for m in messages)]
        body = {
            "model": model,
            "messages": chat,
            "stream": False,
            "format": output_config["format"]["schema"],
            # Thinking off: the schema-constrained answer is short, and on a
            # laptop thinking multiplies the run time several-fold.
            "think": c.think,
            "options": {"temperature": 0, "seed": 0, "num_ctx": NUM_CTX,
                        **({"num_predict": max_tokens} if max_tokens else {})},
        }
        try:
            r = requests.post(f"{c.host}/api/chat", json=body, timeout=TIMEOUT)
        except requests.RequestException as exc:
            raise OllamaError(f"cannot reach Ollama at {c.host}: {exc}") from exc
        if r.status_code != 200:
            raise OllamaError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        stop = {"stop": "end_turn", "length": "max_tokens"}.get(data.get("done_reason"),
                                                                 data.get("done_reason"))
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=data.get("message", {}).get("content", ""))],
            stop_reason=stop,
            model=data.get("model", model),
            usage=SimpleNamespace(input_tokens=data.get("prompt_eval_count"),
                                  cache_read_input_tokens=None,
                                  output_tokens=data.get("eval_count")),
        )


class OllamaClient:
    def __init__(self, host: str = DEFAULT_HOST, think: bool = False):
        self.host, self.think = host.rstrip("/"), think
        self.messages = _Messages(self)

    def check(self, model: str) -> None:
        """Fail fast, with the fix, if the server is down or the model isn't pulled."""
        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=10).json()
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama is not running at {self.host} "
                              f"(start it with `ollama serve`): {exc}") from exc
        names = {m["name"] for m in tags.get("models", [])}
        if model not in names and f"{model}:latest" not in names:
            raise OllamaError(f"model {model!r} is not pulled; run `ollama pull {model}`")
