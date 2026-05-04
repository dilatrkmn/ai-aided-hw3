"""Local LLM client (Ollama HTTP API).

We talk to Ollama via plain HTTP rather than via the official Python client
to keep the dependency surface small and to make the wire format obvious in
the source. Both blocking and streaming modes are supported.
"""

from __future__ import annotations

import json
from typing import Iterator

import requests

from . import config


class OllamaError(RuntimeError):
    """Raised when we can't reach Ollama or the request fails."""


def _endpoint(path: str) -> str:
    return f"{config.OLLAMA_HOST.rstrip('/')}/{path.lstrip('/')}"


def is_available(model: str | None = None) -> tuple[bool, str]:
    """Best-effort liveness check.

    Returns ``(ok, message)``. If a model is supplied we additionally check
    that it appears in the ``/api/tags`` listing; otherwise we just confirm
    the daemon is reachable.
    """
    try:
        resp = requests.get(_endpoint("/api/tags"), timeout=3)
    except requests.RequestException as e:
        return False, f"Ollama not reachable at {config.OLLAMA_HOST}: {e}"

    if resp.status_code != 200:
        return False, f"Ollama returned HTTP {resp.status_code}"

    if model:
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        # Ollama tags are stored as "llama3.2:3b". Match exactly or by prefix.
        if not any(n == model or n.startswith(model + ":") or n.split(":")[0] == model.split(":")[0] for n in names):
            return False, (
                f"Model {model!r} is not pulled. Run: ollama pull {model}"
            )

    return True, "ok"


def generate(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.2,
    num_ctx: int = 4096,
) -> str:
    """Blocking generation. Returns the full response string."""

    model = model or config.LLM_MODEL_NAME
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    if system:
        body["system"] = system

    try:
        resp = requests.post(_endpoint("/api/generate"), json=body, timeout=300)
    except requests.RequestException as e:
        raise OllamaError(f"Request to Ollama failed: {e}") from e

    if resp.status_code != 200:
        raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

    return resp.json().get("response", "").strip()


def generate_stream(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.2,
    num_ctx: int = 4096,
) -> Iterator[str]:
    """Streaming generation. Yields token chunks as they arrive."""

    model = model or config.LLM_MODEL_NAME
    body = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    if system:
        body["system"] = system

    try:
        with requests.post(
            _endpoint("/api/generate"), json=body, stream=True, timeout=300
        ) as resp:
            if resp.status_code != 200:
                raise OllamaError(
                    f"Ollama returned HTTP {resp.status_code}: {resp.text}"
                )
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = payload.get("response", "")
                if token:
                    yield token
                if payload.get("done"):
                    break
    except requests.RequestException as e:
        raise OllamaError(f"Streaming request to Ollama failed: {e}") from e
