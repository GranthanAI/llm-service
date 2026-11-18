"""R2: request analysis leaves room to answer: reasoning_effort low (gpt-oss only), a 1024 cap, one doubled retry, then the fallback."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config.settings import LLMServiceConfig
from app.providers.groq_budget import ANALYSIS_EMPTY_OUTPUT
from app.request_analyzer.analyzer import RequestAnalyzer
from app.request_analyzer.groq_client import GroqAnalysisClient, supports_reasoning_effort

pytestmark = pytest.mark.asyncio
MSG = [{"role": "user", "content": "hi"}]


def reply(text):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
                           choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def client(model, *answers):
    sdk = MagicMock()
    sdk.chat.completions.create = AsyncMock(side_effect=[reply(a) for a in answers])
    return GroqAnalysisClient("k", model=model, client=sdk), sdk.chat.completions.create


async def test_defaults_are_1024_and_2048():
    cfg = LLMServiceConfig(_env_file=None)
    assert cfg.request_analysis_max_tokens == 1024 and cfg.request_analysis_retry_max_tokens == 2048
    assert not hasattr(cfg, "groq_analysis_max_tokens")


async def test_reasoning_effort_low_for_gpt_oss_only_and_the_1024_cap():
    assert supports_reasoning_effort("openai/gpt-oss-20b") and not supports_reasoning_effort("llama-3.3-70b-versatile")
    c, create = client("openai/gpt-oss-20b", "{}")
    await c.complete(MSG)
    assert create.await_args.kwargs["reasoning_effort"] == "low" and create.await_args.kwargs["max_tokens"] == 1024
    c2, create2 = client("llama-3.3-70b-versatile", "{}")
    await c2.complete(MSG)
    assert "reasoning_effort" not in create2.await_args.kwargs


async def test_an_empty_answer_is_retried_once_with_the_cap_doubled():
    before = ANALYSIS_EMPTY_OUTPUT._value.get()
    c, create = client("openai/gpt-oss-20b", "  ", '{"ok": true}')
    assert await c.complete(MSG) == '{"ok": true}'
    assert [call.kwargs["max_tokens"] for call in create.await_args_list] == [1024, 2048]
    assert ANALYSIS_EMPTY_OUTPUT._value.get() == before + 1


async def test_still_empty_after_the_retry_raises_the_usual_error_once_not_a_loop():
    from app.exceptions.provider import ProviderError

    c, create = client("openai/gpt-oss-20b", "", "")
    with pytest.raises(ProviderError):
        await c.complete(MSG)
    assert create.await_count == 2


async def test_a_non_empty_answer_makes_a_single_call():
    c, create = client("openai/gpt-oss-20b", '{"a": 1}')
    await c.complete(MSG)
    assert create.await_count == 1


def analyzer_with(groq_client):
    a = RequestAnalyzer.__new__(RequestAnalyzer)
    a.groq_client, a.fallback_clients, a.fallback_adapters, a.nvidia_adapter = groq_client, [], [], None
    a.logger = MagicMock()
    a.prompt_builder = SimpleNamespace(build_prompt=lambda ctx, tools: MSG)
    a.circuit_breaker = MagicMock()
    a.circuit_breaker.call = lambda: _Ctx()
    a.retry_manager = SimpleNamespace(execute_with_retry=lambda fn, **kw: fn())
    a.config = LLMServiceConfig(_env_file=None)
    return a


class _Ctx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *a):
        return False


async def test_an_unparseable_plan_gets_one_retry_with_a_doubled_cap_then_the_heuristic_fallback():
    calls = []

    class FakeClient:
        model, max_tokens, retry_max_tokens = "openai/gpt-oss-20b", 1024, 2048

        async def reserve(self, messages):
            return 1

        async def complete(self, messages, estimate=None, max_tokens=None, **kw):
            calls.append(max_tokens)
            return '{"mode": "this is cut off'                      # not JSON, both times

    plan = await analyzer_with(FakeClient()).analyze(SimpleNamespace(conversation_id="c1"), None)
    assert calls == [None, 2048]                                      # the first call, then ONE retry with twice the room
    assert plan is not None                                           # the existing heuristic (safe default) plan came back
