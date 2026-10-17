"""Groq's gpt-oss occasionally attempts a tool call with none offered; the adapter must recover from it
(before any text was produced) instead of failing the whole turn, and OpenAlex must tolerate '?'."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from app.exceptions.provider import ProviderError
from app.providers.groq import GroqAdapter
from app.tools.search_providers.openalex import OpenAlexProvider, _sanitize

TOOL_ERR = "Tool choice is none, but model called a tool"


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=None)])


class _Stream:
    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        async def gen():
            for i in self.items:
                if isinstance(i, Exception):
                    raise i
                yield _chunk(i)

        return gen()


def _adapter(script):
    """`script`: list of per-call behaviours; each is a list of tokens/Exceptions, or an Exception raised at create()."""
    calls = []

    class _Completions:
        async def create(self, **kwargs):
            calls.append(kwargs["messages"])
            step = script[min(len(calls) - 1, len(script) - 1)]
            if isinstance(step, Exception):
                raise step
            return _Stream(step)

    a = GroqAdapter(api_key="k", model="openai/gpt-oss-20b")
    a.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    a.logger = MagicMock()
    return a, calls


async def _collect(adapter, messages=None):
    return [t async for t in adapter.stream(messages or [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}])]


async def test_tool_call_error_before_any_text_is_retried_with_a_no_tools_instruction():
    a, calls = _adapter([[Exception(TOOL_ERR)], ["Hel", "lo"]])
    assert "".join(await _collect(a)) == "Hello"
    assert len(calls) == 2
    assert "Do not call" in calls[1][0]["content"] and calls[0][0]["content"] == "S"  # only the retry is nudged


async def test_error_raised_when_the_stream_is_opened_is_also_retried():
    a, calls = _adapter([Exception(TOOL_ERR), ["ok"]])
    assert await _collect(a) == ["ok"]


async def test_nudge_is_added_when_there_is_no_system_message():
    a, calls = _adapter([[Exception(TOOL_ERR)], ["ok"]])
    await _collect(a, [{"role": "user", "content": "q"}])
    assert calls[1][0]["role"] == "system" and "Do not call" in calls[1][0]["content"]


async def test_gives_up_after_the_retry_budget_with_a_provider_error():
    a, calls = _adapter([[Exception(TOOL_ERR)]])
    with pytest.raises(ProviderError):
        await _collect(a)
    assert len(calls) == 3  # first try + 2 retries


async def test_never_retries_once_text_has_already_been_streamed():
    a, calls = _adapter([["partial", Exception(TOOL_ERR)], ["SHOULD-NOT-APPEAR"]])
    got = []
    with pytest.raises(ProviderError):
        async for t in a.stream([{"role": "user", "content": "q"}]):
            got.append(t)
    assert got == ["partial"] and len(calls) == 1


async def test_other_errors_are_not_retried():
    a, calls = _adapter([Exception("401 invalid api key")])
    with pytest.raises(ProviderError):
        await _collect(a)
    assert len(calls) == 1


def test_openalex_query_drops_wildcard_characters():
    assert _sanitize("What are quantization techniques and what is JEV?") == "What are quantization techniques and what is JEV"
    assert _sanitize("a*b  ?") == "a b"


async def test_openalex_request_has_no_wildcards():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["q"] = request.url.params.get("search")
        return httpx.Response(200, json={"results": [{"display_name": "Quantization survey", "id": "https://openalex.org/W1", "publication_year": 2021, "cited_by_count": 5}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await OpenAlexProvider().search("what is JEV?", 3, client)
    assert "?" not in seen["q"] and results


# ------------------------------------------------------------------ rate limits (Groq free tier: 8000 TPM)


async def test_short_rate_limit_waits_once_and_then_answers(monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("app.providers.groq.asyncio.sleep", fake_sleep)
    a, calls = _adapter([Exception("Error code: 429 - Rate limit reached ... Please try again in 2.5s."), ["ok"]])
    assert await _collect(a) == ["ok"]
    assert len(calls) == 2 and sleeps and 2.5 <= sleeps[0] <= 3.0


async def test_long_rate_limit_fails_immediately_with_a_rate_limit_error_instead_of_stalling(monkeypatch):
    from app.exceptions.provider import ProviderRateLimitError

    async def no_sleep(s):  # pragma: no cover - must not be reached
        raise AssertionError("must not wait for a long rate limit")

    monkeypatch.setattr("app.providers.groq.asyncio.sleep", no_sleep)
    a, calls = _adapter([Exception("Error code: 429 - Rate limit reached ... Please try again in 16.45s.")])
    with pytest.raises(ProviderRateLimitError):
        await _collect(a)
    assert len(calls) == 1


async def test_only_one_rate_limit_wait_per_request(monkeypatch):
    from app.exceptions.provider import ProviderRateLimitError

    async def fake_sleep(s):
        return None

    monkeypatch.setattr("app.providers.groq.asyncio.sleep", fake_sleep)
    a, calls = _adapter([Exception("429 Rate limit. Please try again in 1s.")])
    with pytest.raises(ProviderRateLimitError):
        await _collect(a)
    assert len(calls) == 2


def test_retry_after_is_parsed_in_seconds_and_milliseconds():
    from app.providers.groq import rate_limit_wait_s

    assert rate_limit_wait_s(Exception("429 ... try again in 16.455s")) == pytest.approx(16.455)
    assert rate_limit_wait_s(Exception("429 ... try again in 250ms")) == pytest.approx(0.25)
    assert rate_limit_wait_s(Exception("429 rate limit, no hint")) == 60.0
    assert rate_limit_wait_s(Exception("500 server error")) is None


def test_default_client_does_not_silently_retry():
    a = GroqAdapter(api_key="k", model="openai/gpt-oss-20b")
    assert a.client.max_retries == 0
