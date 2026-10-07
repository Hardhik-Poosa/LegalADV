"""LLM abstraction with an LM Studio implementation.

LM Studio exposes an OpenAI-compatible server (default http://localhost:1234/v1).
Swap in a cloud client later by implementing the same `LLMClient` protocol.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base class for LLM errors."""


class LLMUnavailableError(LLMError):
    """Server unreachable or timed out."""


class LLMResponseError(LLMError):
    """Server replied with an error or a malformed payload."""


@dataclass(frozen=True)
class LMStudioConfig:
    base_url: str = "http://localhost:1234/v1"
    chat_model: str = "local-model"  # use the id shown in LM Studio
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    timeout_s: float = 120.0
    temperature: float = 0.1
    max_tokens: int = 1024


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class LLMClient(Protocol):
    def chat(
        self, messages: list[ChatMessage], json_schema: dict[str, Any] | None = None
    ) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def health(self) -> bool: ...


class LMStudioClient:
    def __init__(
        self, config: LMStudioConfig, http: httpx.Client | None = None
    ) -> None:
        self._config = config
        self._http = http or httpx.Client(
            base_url=config.base_url, timeout=config.timeout_s
        )

    def chat(
        self, messages: list[ChatMessage], json_schema: dict[str, Any] | None = None
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._config.chat_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": json_schema},
            }
        data = self._post("/chat/completions", payload)
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("Malformed chat response") from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post(
            "/embeddings", {"model": self._config.embedding_model, "input": texts}
        )
        try:
            items = sorted(data["data"], key=lambda d: d["index"])
            return [list(map(float, d["embedding"])) for d in items]
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMResponseError("Malformed embedding response") from exc

    def health(self) -> bool:
        try:
            return self._http.get("/models").status_code == 200
        except httpx.HTTPError:
            logger.warning("LM Studio health check failed")
            return False

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._http.post(path, json=payload)
        except httpx.HTTPError as exc:
            logger.error("LM Studio unreachable: %s", exc)
            raise LLMUnavailableError(str(exc)) from exc
        if resp.status_code != 200:
            logger.error("LM Studio error %s: %s", resp.status_code, resp.text[:200])
            raise LLMResponseError(f"HTTP {resp.status_code}")
        try:
            result: dict[str, Any] = resp.json()
        except ValueError as exc:
            raise LLMResponseError("Invalid JSON") from exc
        return result
