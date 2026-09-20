"""Offline tests: every request is served by a mock transport, no vonk needed."""
import json

import httpx
import pytest

from vonk import AsyncVonk, Vonk, VonkError, clean, parse_json
from vonk._core import CONTEXT_LIMIT, Config


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("VONK_KEY", "test-key")
    monkeypatch.setenv("VONK_BASE_URL", "http://vonk:4000/v1")


def _chat(content):
    return {"choices": [{"message": {"content": content}}]}


def _mount(client, handler):
    """Swap the real pool for a mock transport, keeping headers/base_url."""
    client._client = type(client._client)(
        base_url=client.config.base_url,
        headers=client.config.headers,
        transport=httpx.MockTransport(handler) if not isinstance(
            client._client, httpx.AsyncClient
        ) else httpx.MockTransport(handler),
    )
    return client


# --- cleaning -------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("plain", "plain"),
    ("```json\n{\"a\": 1}\n```", '{"a": 1}'),
    ("<think>hmm</think>answer", "answer"),
    ("<think>a</think>\n```\ncode\n```", "code"),
])
def test_clean(raw, want):
    assert clean(raw) == want


def test_parse_json_handles_prose_wrapper():
    assert parse_json('Sure! {"concepts": ["a"]} hope that helps') == {"concepts": ["a"]}


def test_parse_json_raises_on_garbage():
    with pytest.raises(VonkError):
        parse_json("no json here")


# --- config ---------------------------------------------------------------

def test_missing_key_is_a_loud_error(monkeypatch):
    monkeypatch.delenv("VONK_KEY", raising=False)
    with pytest.raises(VonkError, match="No vonk API key"):
        Config.from_env()


def test_no_key_is_baked_in(monkeypatch):
    monkeypatch.setenv("VONK_KEY", "abc")
    assert Config.from_env().api_key == "abc"


def test_oversized_max_tokens_rejected():
    v = Vonk()
    with pytest.raises(VonkError, match="caps input"):
        v.chat("hi", max_tokens=CONTEXT_LIMIT)


# --- requests -------------------------------------------------------------

def test_chat_sends_key_and_cleans_reply():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_chat("<think>x</think>hello"))

    v = _mount(Vonk(), handler)
    assert v.chat("hi", system="be brief") == "hello"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "be brief"}


def test_json_mode_sets_response_format():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_chat('```json\n{"ok": true}\n```'))

    v = _mount(Vonk(), handler)
    assert v.json("give me json") == {"ok": True}
    assert seen["response_format"] == {"type": "json_object"}


def test_retry_then_succeed():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500)
        return httpx.Response(200, json=_chat("second try"))

    v = _mount(Vonk(retries=1), handler)
    assert v.chat("hi") == "second try"
    assert calls["n"] == 2


def test_gives_up_after_retries():
    v = _mount(Vonk(retries=1), lambda r: httpx.Response(503))
    with pytest.raises(VonkError, match="failed"):
        v.chat("hi")


def test_embed_single_string_is_wrapped():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})

    v = _mount(Vonk(), handler)
    assert v.embed("text") == [[0.1, 0.2]]
    assert seen["input"] == ["text"]


def test_bad_response_shape_is_vonk_error():
    v = _mount(Vonk(), lambda r: httpx.Response(200, json={"nope": 1}))
    with pytest.raises(VonkError, match="Unexpected chat response"):
        v.chat("hi")


def test_image_returns_url():
    v = _mount(Vonk(), lambda r: httpx.Response(200, json={"data": [{"url": "http://x/y.png"}]}))
    assert v.image("a cat") == "http://x/y.png"


# --- async twin -----------------------------------------------------------

async def test_async_chat_matches_sync():
    v = AsyncVonk()
    v._client = httpx.AsyncClient(
        base_url=v.config.base_url,
        headers=v.config.headers,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_chat("async hi"))),
    )
    assert await v.chat("hi") == "async hi"
    await v.aclose()


async def test_async_json_parses():
    v = AsyncVonk()
    v._client = httpx.AsyncClient(
        base_url=v.config.base_url,
        headers=v.config.headers,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_chat('{"n": 1}'))),
    )
    assert await v.json("x") == {"n": 1}
    await v.aclose()
