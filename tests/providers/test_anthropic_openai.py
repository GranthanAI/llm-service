"""P5b: Anthropic and OpenAI adapters against faked SDK streams (no network, no keys)."""

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from app.config.settings import LLMServiceConfig
from app.exceptions.provider import ProviderError, ProviderRateLimitError, ProviderTimeoutError
from app.providers.anthropic import AnthropicAdapter, to_anthropic_messages
from app.providers.base import TRUNCATION_NOTICE
from app.providers.openai import OpenAIAdapter


class Sink:
    def __init__(self):
        self.calls = []

    def record_provider_usage(self, **kw):
        self.calls.append(kw)


class Stream:
    def __init__(self, items, error=None):
        self.items, self.error = items, error

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for item in self.items:
            yield item
        if self.error:
            raise self.error


# --- Anthropic ----------------------------------------------------------------------------------------------

def a_events(text_parts=("Hel", "lo"), stop="end_turn", usage_in=100, cache_read=40, cache_create=10, out_start=1, out_end=25):
    ev = [NS(type="message_start", message=NS(usage=NS(input_tokens=usage_in, cache_creation_input_tokens=cache_create,
                                                      cache_read_input_tokens=cache_read, output_tokens=out_start)))]
    ev += [NS(type="content_block_delta", delta=NS(type="text_delta", text=t)) for t in text_parts]
    ev.append(NS(type="content_block_delta", delta=NS(type="thinking_delta", thinking="hmm")))   # never shown
    ev.append(NS(type="message_delta", delta=NS(stop_reason=stop), usage=NS(output_tokens=out_end)))
    ev.append(NS(type="message_stop"))
    return ev


def a_client(streams=None, create_result=None):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream"):
            item = (streams or []).pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return create_result

    return NS(messages=NS(create=create)), calls


async def collect(agen):
    return [t async for t in agen]


def test_message_mapping_system_separate_alternating_and_first_is_user():
    system, msgs = to_anthropic_messages([
        {"role": "system", "content": "Be brief."}, {"role": "system", "content": "Use English."},
        {"role": "assistant", "content": "earlier answer"}, {"role": "user", "content": "q1"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "   "}, {"role": "user", "content": ""},
    ])
    assert system == "Be brief.\n\nUse English."
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == "(earlier conversation)" and msgs[2]["content"] == "q1\n\nq2"
    assert to_anthropic_messages([{"role": "system", "content": "s"}]) == ("s", [{"role": "user", "content": "Hello"}])


async def test_anthropic_request_has_system_separate_and_max_tokens():
    client, calls = a_client([Stream(a_events())])
    adapter = AnthropicAdapter(api_key="k", model="claude-sonnet-5-5", client=client)
    await collect(adapter.stream([{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}], {"max_tokens": 777}))
    (call,) = calls
    assert call["system"] == "sys" and call["max_tokens"] == 777 and call["stream"] is True
    assert call["messages"] == [{"role": "user", "content": "hi"}] and call["model"] == "claude-sonnet-5-5"
    assert "temperature" not in call
    await collect(adapter.stream([{"role": "user", "content": "x"}], {})) if False else None
    client2, calls2 = a_client([Stream(a_events())])
    await collect(AnthropicAdapter(api_key="k", client=client2, max_tokens=1234).stream([{"role": "user", "content": "x"}], {}))
    assert calls2[0]["max_tokens"] == 1234                       # required by the API: never omitted


async def test_anthropic_streamed_text_and_usage_equal_what_the_provider_reported():
    client, _ = a_client([Stream(a_events(("A", "B", "C")))])
    sink = Sink()
    text = await collect(AnthropicAdapter(api_key="k", client=client).stream([{"role": "user", "content": "hi"}], {"_usage": sink}))
    assert "".join(text) == "ABC"                                # thinking deltas are not shown
    (u,) = sink.calls
    assert u["provider"] == "anthropic" and u["model"] == "claude-sonnet-5-5"
    assert u["input_tokens"] == 100 + 10 + 40                     # total prompt: uncached + cache write + cache read
    assert u["output_tokens"] == 25 and u["cached_input_tokens"] == 40


async def test_anthropic_cut_off_answer_gets_the_visible_notice():
    client, _ = a_client([Stream(a_events(stop="max_tokens"))])
    text = await collect(AnthropicAdapter(api_key="k", client=client).stream([{"role": "user", "content": "hi"}], {}))
    assert text[-1] == TRUNCATION_NOTICE


async def test_anthropic_execute_reports_usage_and_assembles_text():
    resp = NS(content=[NS(type="text", text="Hello "), NS(type="thinking", thinking="x"), NS(type="text", text="world")],
              usage=NS(input_tokens=12, cache_creation_input_tokens=0, cache_read_input_tokens=2, output_tokens=7))
    client, _ = a_client(create_result=resp)
    sink = Sink()
    out = await AnthropicAdapter(api_key="k", client=client).execute([{"role": "user", "content": "hi"}], {"_usage": sink})
    assert out.content == "Hello world" and (out.prompt_tokens, out.completion_tokens) == (14, 7)
    assert sink.calls[0]["input_tokens"] == 14 and sink.calls[0]["cached_input_tokens"] == 2


class RateLimitError(Exception):
    status_code = 429


class APITimeoutError(Exception):
    pass


async def test_anthropic_error_mapping():
    for err, expected in [(RateLimitError("Error code: 429 - slow down"), ProviderRateLimitError),
                          (APITimeoutError("Request timed out."), ProviderTimeoutError),
                          (ValueError("boom"), ProviderError)]:
        client, _ = a_client([err])
        with pytest.raises(expected):
            await collect(AnthropicAdapter(api_key="k", client=client).stream([{"role": "user", "content": "hi"}], {}))


async def test_anthropic_short_rate_limit_is_retried_once_but_never_after_text(monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    client, calls = a_client([RateLimitError("Error code: 429 - rate limit, please try again in 1s"), Stream(a_events())])
    text = await collect(AnthropicAdapter(api_key="k", client=client).stream([{"role": "user", "content": "hi"}], {}))
    assert "".join(text) == "Hello" and len(calls) == 2 and slept
    mid = Stream(a_events(("Hi",))[:2], error=RateLimitError("Error code: 429 - try again in 1s"))
    client2, calls2 = a_client([mid, Stream(a_events())])
    with pytest.raises(ProviderRateLimitError):
        await collect(AnthropicAdapter(api_key="k", client=client2).stream([{"role": "user", "content": "hi"}], {}))
    assert len(calls2) == 1                                       # text was already shown: no splice, no retry


async def test_a_failed_attempt_that_reached_the_provider_still_reports_its_usage():
    broken = Stream(a_events()[:2], error=ValueError("stream died"))
    client, _ = a_client([broken])
    sink = Sink()
    with pytest.raises(ProviderError):
        await collect(AnthropicAdapter(api_key="k", client=client).stream([{"role": "user", "content": "hi"}], {"_usage": sink}))
    assert sink.calls and sink.calls[0]["input_tokens"] == 150


# --- OpenAI ---------------------------------------------------------------------------------------------------

def o_chunks(parts=("Hel", "lo"), finish="stop", prompt=200, completion=30, cached=64):
    ch = [NS(choices=[NS(delta=NS(content=p), finish_reason=None)], usage=None) for p in parts]
    ch.append(NS(choices=[NS(delta=NS(content=None), finish_reason=finish)], usage=None))
    ch.append(NS(choices=[], usage=NS(prompt_tokens=prompt, completion_tokens=completion,
                                      prompt_tokens_details=NS(cached_tokens=cached))))
    return ch


def o_client(streams=None, create_result=None):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream"):
            item = streams.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return create_result

    return NS(chat=NS(completions=NS(create=create))), calls


async def test_openai_request_asks_for_stream_usage_and_uses_max_completion_tokens():
    client, calls = o_client([Stream(o_chunks())])
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    await collect(OpenAIAdapter(api_key="k", model="gpt-6-luna", client=client).stream(msgs, {"max_tokens": 900}))
    (call,) = calls
    assert call["stream"] is True and call["stream_options"] == {"include_usage": True}
    assert call["max_completion_tokens"] == 900 and "max_tokens" not in call and call["messages"] == msgs
    assert call["model"] == "gpt-6-luna"


async def test_openai_streamed_text_and_usage_from_the_final_chunk():
    client, _ = o_client([Stream(o_chunks(("x", "y", "z")))])
    sink = Sink()
    text = await collect(OpenAIAdapter(api_key="k", model="gpt-6-luna", client=client).stream(
        [{"role": "user", "content": "hi"}], {"_usage": sink}))
    assert "".join(text) == "xyz"
    (u,) = sink.calls
    assert (u["provider"], u["model"]) == ("openai", "gpt-6-luna")
    assert (u["input_tokens"], u["output_tokens"], u["cached_input_tokens"]) == (200, 30, 64)


async def test_openai_missing_usage_is_reported_as_none_so_the_collector_estimates():
    chunks = o_chunks()[:-1]                                        # no final usage chunk
    client, _ = o_client([Stream(chunks)])
    sink = Sink()
    await collect(OpenAIAdapter(api_key="k", client=client).stream([{"role": "user", "content": "hi"}], {"_usage": sink}))
    assert sink.calls[0]["input_tokens"] is None and sink.calls[0]["output_tokens"] is None


async def test_openai_cut_off_answer_gets_the_visible_notice_and_execute_reports_usage():
    client, _ = o_client([Stream(o_chunks(finish="length"))])
    text = await collect(OpenAIAdapter(api_key="k", client=client).stream([{"role": "user", "content": "hi"}], {}))
    assert text[-1] == TRUNCATION_NOTICE
    resp = NS(choices=[NS(message=NS(content="done"))],
              usage=NS(prompt_tokens=11, completion_tokens=5, prompt_tokens_details=NS(cached_tokens=3)))
    client2, _ = o_client(create_result=resp)
    sink = Sink()
    out = await OpenAIAdapter(api_key="k", client=client2).execute([{"role": "user", "content": "hi"}], {"_usage": sink})
    assert out.content == "done" and (out.prompt_tokens, out.completion_tokens) == (11, 5)
    assert sink.calls[0]["cached_input_tokens"] == 3


async def test_openai_error_mapping_and_one_short_rate_limit_retry(monkeypatch):
    for err, expected in [(RateLimitError("Error code: 429 - Rate limit reached, try again in 60s"), ProviderRateLimitError),
                          (APITimeoutError("Request timed out."), ProviderTimeoutError),
                          (ValueError("boom"), ProviderError)]:
        client, _ = o_client([err])
        with pytest.raises(expected):
            await collect(OpenAIAdapter(api_key="k", client=client).stream([{"role": "user", "content": "hi"}], {}))

    async def fake_sleep(s):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    client, calls = o_client([RateLimitError("Error code: 429 - Rate limit reached. Please try again in 1.5s"), Stream(o_chunks())])
    text = await collect(OpenAIAdapter(api_key="k", client=client).stream([{"role": "user", "content": "hi"}], {}))
    assert "".join(text) == "Hello" and len(calls) == 2


# --- registry wiring: a missing key means the model is skipped -----------------------------------------------------

def test_missing_keys_skip_the_provider_models_in_group_routing():
    from app.models.billing import BillingGrant
    from app.providers.group_chain import build_group_chains

    cfg = LLMServiceConfig(groq_api_key="g", gemini_api_key="", anthropic_api_key="", openai_api_key="o")
    builder = build_group_chains(cfg, MagicMock(), MagicMock())
    chain = builder.build(BillingGrant(granted_group="smart"))
    assert [e.spec.id for e in chain.entries] == ["openai/gpt-oss-120b", "gpt-6-luna"]
    assert all(isinstance(e.adapter, (AnthropicAdapter, OpenAIAdapter)) or e.spec.provider == "groq" for e in chain.entries)
    cfg = LLMServiceConfig(anthropic_api_key="a", openai_api_key="o", groq_api_key="g", gemini_api_key="m")
    best = build_group_chains(cfg, MagicMock(), MagicMock()).build(BillingGrant(granted_group="best"))
    assert [e.spec.id for e in best.entries] == ["claude-sonnet-5-5", "gpt-6.1-sol"]
    assert isinstance(best.entries[0].adapter, AnthropicAdapter) and isinstance(best.entries[1].adapter, OpenAIAdapter)
