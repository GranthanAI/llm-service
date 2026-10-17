"""Ordered provider chain (Groq models first, NVIDIA last) and the planning / rewrite fallbacks behind it."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config.settings import LLMServiceConfig
from app.context.schemas import Message, Role
from app.context_window.models import PromptSection, TrimmedPrompt
from app.exceptions.provider import AllProvidersFailedError, ProviderRateLimitError
from app.models.pipeline_context import PipelineContext
from app.providers.chain import ChainEntry, ProviderChain
from app.request_analyzer.analyzer import RequestAnalyzer
from app.tools.query_rewrite import SearchQueryRewriter
from app.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from app.utils.error_handler import ErrorHandler


def _prompt():
    return TrimmedPrompt(
        sections=[PromptSection(name="q", content="x", priority=1, token_count=1, trimmable=False)],
        messages=[{"role": "user", "content": "x"}], total_tokens=1, mode="default", target_model="m",
    )


class _Adapter:
    """Streams `tokens`, optionally after a delay, or raises `error` before the first token."""

    def __init__(self, name, tokens=(), error=None, delay=0.0, fail_after=None):
        self.name, self.tokens, self.error, self.delay, self.fail_after = name, list(tokens), error, delay, fail_after
        self.calls = 0
        self.closed = False

    async def stream(self, messages, params=None, **kw):
        self.calls += 1
        try:
            if self.error:
                raise self.error
            await asyncio.sleep(self.delay)
            for i, t in enumerate(self.tokens):
                if self.fail_after is not None and i == self.fail_after:
                    raise RuntimeError("died mid-answer")
                yield t
        finally:
            self.closed = True


def _entry(adapter, first=0.5, breaker=True):
    return ChainEntry(
        name=adapter.name,
        adapter=adapter,
        breaker=CircuitBreaker(adapter.name, CircuitBreakerConfig(failure_threshold=2)) if breaker else None,
        first_token_timeout_s=first,
    )


async def _collect(chain, params=None):
    return [t async for t in chain.generate_stream(_prompt(), params)]


# ------------------------------------------------------------------ settings


def test_default_order_puts_groq_first_and_nvidia_last():
    c = LLMServiceConfig(_env_file=None)
    assert c.generation_order_list == ["groq:openai/gpt-oss-120b", "groq:qwen/qwen3.8-27b", "groq:openai/gpt-oss-20b", "nvidia"]
    assert c.groq_order_models == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]


def test_order_parsing_trims_and_dedupes():
    c = LLMServiceConfig(_env_file=None, generation_provider_order=" groq:a , nvidia,groq:a,, gemini ")
    assert c.generation_order_list == ["groq:a", "nvidia", "gemini"]


# ------------------------------------------------------------------ chain


async def test_first_provider_answers_and_the_rest_are_never_called():
    a, b = _Adapter("groq:120b", ["Hel", "lo"]), _Adapter("groq:qwen", ["no"])
    params: dict = {}
    assert "".join(await _collect(ProviderChain([_entry(a), _entry(b)]), params)) == "Hello"
    assert b.calls == 0 and params["_route"] == {"provider": "groq:120b", "fallback_used": False, "fallback_reason": None}


async def test_rate_limited_first_provider_falls_through_to_the_next():
    a = _Adapter("groq:120b", error=ProviderRateLimitError("Groq Rate Limit: 429", provider="groq"))
    b = _Adapter("groq:qwen", ["from qwen"])
    params: dict = {}
    assert await _collect(ProviderChain([_entry(a), _entry(b)]), params) == ["from qwen"]
    assert params["_route"]["provider"] == "groq:qwen" and params["_route"]["fallback_used"] is True
    assert "Rate Limit" in params["_route"]["fallback_reason"]


async def test_slow_provider_is_abandoned_at_its_deadline():
    slow, ok = _Adapter("groq:120b", ["late"], delay=5.0), _Adapter("groq:qwen", ["fast"])
    t0 = asyncio.get_event_loop().time()
    assert await _collect(ProviderChain([_entry(slow, first=0.2), _entry(ok)])) == ["fast"]
    assert asyncio.get_event_loop().time() - t0 < 2.0 and slow.closed


async def test_walks_the_whole_chain_down_to_the_last_resort():
    a = _Adapter("groq:120b", error=RuntimeError("down"))
    b = _Adapter("groq:qwen", error=RuntimeError("down too"))
    c = _Adapter("groq:20b", error=RuntimeError("quota"))
    n = _Adapter("nvidia", ["finally"])
    assert await _collect(ProviderChain([_entry(a), _entry(b), _entry(c), _entry(n)])) == ["finally"]
    assert [x.calls for x in (a, b, c, n)] == [1, 1, 1, 1]


async def test_never_switches_provider_after_text_has_started():
    a, b = _Adapter("groq:120b", ["part", "ial"], fail_after=1), _Adapter("groq:qwen", ["SHOULD-NOT-APPEAR"])
    got = []
    with pytest.raises(RuntimeError):
        async for t in ProviderChain([_entry(a), _entry(b)]).generate_stream(_prompt()):
            got.append(t)
    assert got == ["part"] and b.calls == 0


async def test_open_circuit_breaker_skips_a_provider_without_calling_it():
    a, b = _Adapter("groq:120b", ["x"]), _Adapter("groq:qwen", ["ok"])
    ea, eb = _entry(a), _entry(b)
    ea.breaker.trip()
    params: dict = {}
    assert await _collect(ProviderChain([ea, eb]), params) == ["ok"]
    assert a.calls == 0 and params["_route"]["fallback_reason"] == "circuit breaker open"


async def test_repeated_failures_open_the_breaker_so_later_turns_skip_the_provider():
    bad, ok = _Adapter("groq:120b", error=RuntimeError("x")), _Adapter("groq:qwen", ["ok"])
    chain = ProviderChain([_entry(bad), _entry(ok)])
    for _ in range(3):
        await _collect(chain)
    assert bad.calls == 2  # threshold 2: the third turn did not even try it


async def test_all_failing_raises_one_error_naming_every_provider_and_keeps_rate_limit_visible():
    a = _Adapter("groq:120b", error=ProviderRateLimitError("Groq Rate Limit: 429 try again", provider="groq"))
    b = _Adapter("nvidia", error=RuntimeError("timeout"))
    with pytest.raises(AllProvidersFailedError) as e:
        await _collect(ProviderChain([_entry(a), _entry(b)]))
    msg = str(e.value)
    assert "groq:120b" in msg and "nvidia" in msg
    assert ErrorHandler().classify(e.value).error_code == "RATE_LIMIT_EXCEEDED"  # user gets the clear "busy" message


def test_chain_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        ProviderChain([])


# ------------------------------------------------------------------ request planning fallbacks


class _Client:
    def __init__(self, model, reply=None, exc=None):
        self.model, self.timeout_ms, self.reply, self.exc, self.calls = model, 1000, reply, exc, 0

    async def complete(self, messages, **kw):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.reply


PLAN = json.dumps({"intent": "GENERAL_CHAT", "mode": "default", "skill": "general_chat", "reasoning": "DIRECT",
                   "tools": [], "max_iterations": 1, "suggested_temperature": 0.3, "analysis_confidence": 0.9})


def _analyzer(primary, fallbacks, nvidia=None):
    a = RequestAnalyzer(groq_client=primary, config=LLMServiceConfig(_env_file=None), nvidia_adapter=nvidia, fallback_clients=fallbacks)
    a.logger = MagicMock()
    return a


def _ctx():
    return PipelineContext(conversation_id="c", user_id="u", message_id="m", request_id="r", user_message="explain closures")


async def test_planning_falls_back_to_the_next_groq_model_when_the_primary_is_rate_limited():
    primary = _Client("openai/gpt-oss-20b", exc=RuntimeError("429 daily limit"))
    fb1, fb2 = _Client("openai/gpt-oss-120b", reply=PLAN), _Client("qwen/qwen3.8-27b", reply=PLAN)
    plan = await _analyzer(primary, [fb1, fb2]).analyze(_ctx())
    assert plan.groq_model_used == "openai/gpt-oss-120b" and fb2.calls == 0


async def test_planning_tries_every_fallback_before_the_nvidia_last_resort_and_then_the_safe_default():
    primary = _Client("p", exc=RuntimeError("x"))
    fbs = [_Client("a", exc=RuntimeError("x")), _Client("b", reply="not json")]
    nvidia = MagicMock()
    nvidia.execute = AsyncMock(side_effect=asyncio.TimeoutError())
    plan = await _analyzer(primary, fbs, nvidia).analyze(_ctx())
    assert all(c.calls == 1 for c in fbs) and nvidia.execute.await_count == 1
    assert plan.groq_model_used == "fallback_safe_default"  # never an exception, never a hang


async def test_a_slow_nvidia_planning_call_is_capped():
    primary = _Client("p", exc=RuntimeError("x"))

    async def slow(messages):
        await asyncio.sleep(30)

    nvidia = MagicMock()
    nvidia.execute = slow
    analyzer = _analyzer(primary, [], nvidia)
    analyzer.config = LLMServiceConfig(_env_file=None, analysis_fallback_timeout_ms=200)
    t0 = asyncio.get_event_loop().time()
    plan = await analyzer.analyze(_ctx())
    assert asyncio.get_event_loop().time() - t0 < 3 and plan.groq_model_used == "fallback_safe_default"


# ------------------------------------------------------------------ search-query rewriting fallbacks


def _turns():
    return [Message(role=Role.USER, content="How to use duckduckgo in my project?"), Message(role=Role.ASSISTANT, content="Use duckduckgo-search.")]


async def test_rewriter_uses_the_next_model_when_the_first_fails():
    primary = _Client("20b", exc=RuntimeError("429"))
    fb = _Client("120b", reply=json.dumps({"query": "duckduckgo-search python"}))
    assert await SearchQueryRewriter(primary, fallback_clients=[fb]).rewrite("check again and tell", _turns()) == "duckduckgo-search python"


async def test_rewriter_falls_to_the_heuristic_only_after_every_model_failed():
    clients = [_Client("a", exc=RuntimeError("x")), _Client("b", reply="garbage")]
    q = await SearchQueryRewriter(clients[0], fallback_clients=clients[1:]).rewrite("check again and tell", _turns())
    assert q == "How to use duckduckgo in my project? check again and tell" and all(c.calls == 1 for c in clients)


async def test_pipeline_reports_the_provider_that_actually_answered_even_without_a_fallback():
    from app.pipeline.request_pipeline import RequestPipeline

    ctx = PipelineContext(conversation_id="c", user_id="u", message_id="m", request_id="r", user_message="q", selected_provider="nvidia")
    params = {"_route": {"provider": "groq:openai/gpt-oss-120b", "fallback_used": False}}

    async def src():
        yield "a"

    assert [t async for t in RequestPipeline._track_route(src(), params, ctx)] == ["a"]
    assert ctx.selected_provider == "groq:openai/gpt-oss-120b" and ctx.generation_fallback_used is False
