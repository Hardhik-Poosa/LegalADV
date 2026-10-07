import json

import httpx
import pytest

from app.llm.llm_client import (
    ChatMessage,
    LLMResponseError,
    LLMUnavailableError,
    LMStudioClient,
    LMStudioConfig,
)


def make_client(handler) -> LMStudioClient:
    cfg = LMStudioConfig()
    http = httpx.Client(base_url=cfg.base_url, transport=httpx.MockTransport(handler))
    return LMStudioClient(cfg, http)


def test_chat_returns_content() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert "response_format" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    client = make_client(handler)
    assert client.chat([ChatMessage("user", "hello")]) == "hi"


def test_chat_sends_json_schema() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["response_format"]["type"] == "json_schema"
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    client = make_client(handler)
    assert client.chat([ChatMessage("user", "x")], json_schema={"type": "object"}) == "{}"


def test_chat_malformed_response() -> None:
    client = make_client(lambda r: httpx.Response(200, json={"oops": 1}))
    with pytest.raises(LLMResponseError):
        client.chat([ChatMessage("user", "x")])


def test_http_error_status() -> None:
    client = make_client(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(LLMResponseError):
        client.chat([ChatMessage("user", "x")])


def test_unreachable_raises_unavailable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = make_client(handler)
    with pytest.raises(LLMUnavailableError):
        client.chat([ChatMessage("user", "x")])


def test_embed_sorted_by_index() -> None:
    data = {
        "data": [
            {"index": 1, "embedding": [2.0]},
            {"index": 0, "embedding": [1.0]},
        ]
    }
    client = make_client(lambda r: httpx.Response(200, json=data))
    assert client.embed(["a", "b"]) == [[1.0], [2.0]]


def test_health() -> None:
    assert make_client(lambda r: httpx.Response(200, json={})).health() is True

    def down(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert make_client(down).health() is False
