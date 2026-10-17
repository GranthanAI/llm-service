"""Fast path routing, first-token deadlines/fallback, and pipeline timing instrumentation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config.settings import LLMServiceConfig
from app.context.merger import ContextMerger
from app.context_window.models import PromptSection, TrimmedPrompt
from app.exceptions.provider import AllProvidersFailedError
from app.models.execution_plan import ExecutionPlan, IntentCategory, ReasoningMode, Skill, UserMode
from app.models.request import ChatMessageCreatedEvent
from app.pipeline.request_pipeline import RequestPipeline
from app.providers.router import GenerationRouter
from app.request_analyzer.fast_path import is_simple_smalltalk
from app.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# ---------------------------------------------------------------- matcher


@pytest.mark.parametrize(
    "text",
    [
        "hi",
        "Hi!",
        "hello there",
        "hey!!",
        "Good morning",
        "how are you?",
        "thanks",
        "Thank you so much!",
        "hi, how are you",
        "ok",
        "bye 👋",
        "HELLO",
    ],
)
def test_greetings_take_the_fast_path(text):
    assert is_simple_smalltalk(text)


@pytest.mark.parametrize(
    "text",
    [
        "hi, what is kafka?",
        "hello can you summarize my notes",
        "thanks for the earlier answer, now explain it",
        "ok so what about the previous one",
        "hi again, continue where we left off",
        "what is 2+2",
        "hey write me a poem",
        "",
        "good morning, remind me what I told you yesterday",
        "how are you feeling about python",
    ],
)
def test_substantive_or_context_referencing_messages_do_not(text):
    assert not is_simple_smalltalk(text)


def test_files_or_mode_hint_disable_the_fast_path():
    assert not is_simple_smalltalk("hi", file_ids=["f1"])
    assert not is_simple_smalltalk("hi", mode_hint="smart")


# ---------------------------------------------------------------- router


class _Adapter:
    def __init__(self, name, tokens=(), delay_first=0.0, fail=False, delay_between=0.0):
        self.provider_name, self.tokens = name, list(tokens)
        self.delay_first, self.fail, self.delay_between = delay_first, fail, delay_between
        self.calls = 0
        self.closed = False

    async def stream(self, messages, params=None, **kw):
        self.calls += 1
        try:
            if self.fail:
                raise RuntimeError("boom")
            await asyncio.sleep(self.delay_first)
            for i, t in enumerate(self.tokens):
                if i:
                    await asyncio.sleep(self.delay_between)
                yield t
        finally:
            self.closed = True


def _prompt():
    return TrimmedPrompt(
        sections=[PromptSection(name="q", content="x", priority=1, token_count=1, trimmable=False)],
        messages=[{"role": "user", "content": "x"}],
        total_tokens=1,
        mode="default",
        target_model="m",
    )


def _router(primary, fallback, first=0.15, idle=0.15):
    def cb(n):
        return CircuitBreaker(n, CircuitBreakerConfig(failure_threshold=3))

    return GenerationRouter(
        nvidia_adapter=primary,
        gemini_adapter=fallback,
        circuit_breakers={"nvidia": cb("nvidia"), "groq": cb("groq")},
        fallback_name="groq",
        first_token_timeout_s=first,
        idle_timeout_s=idle,
        fallback_first_token_timeout_s=1.0,
    )


async def _collect(router, params=None):
    return [t async for t in router.generate_stream(_prompt(), params)]


async def test_slow_primary_hands_over_to_fallback_instead_of_waiting():
    primary = _Adapter("nvidia", ["slow"], delay_first=5.0)
    fallback = _Adapter("groq", ["Hello", "!"])
    params: dict = {}
    t0 = asyncio.get_event_loop().time()
    out = await _collect(_router(primary, fallback), params)
    assert "".join(out) == "Hello!"
    assert asyncio.get_event_loop().time() - t0 < 1.0  # did not wait the 5s
    assert params["_route"]["fallback_used"] and params["_route"]["provider"] == "groq"
    assert primary.closed  # the slow upstream request is abandoned, not leaked


async def test_fast_primary_is_used_without_fallback():
    primary, fallback = _Adapter("nvidia", ["a", "b"]), _Adapter("groq", ["x"])
    params: dict = {}
    assert "".join(await _collect(_router(primary, fallback), params)) == "ab"
    assert fallback.calls == 0 and not params["_route"]["fallback_used"]


async def test_never_switches_provider_after_tokens_were_emitted():
    primary = _Adapter("nvidia", ["part", "ial"], delay_between=5.0)  # stalls after first token
    fallback = _Adapter("groq", ["SHOULD-NOT-APPEAR"])
    got = []
    with pytest.raises(Exception):
        async for t in _router(primary, fallback).generate_stream(_prompt()):
            got.append(t)
    assert got == ["part"] and fallback.calls == 0  # no spliced answer


async def test_both_providers_slow_fails_fast_instead_of_hanging():
    router = _router(_Adapter("nvidia", ["x"], delay_first=5), _Adapter("groq", ["y"], delay_first=5))
    router.fallback_first_token_timeout_s = 0.2
    t0 = asyncio.get_event_loop().time()
    with pytest.raises(AllProvidersFailedError):
        await _collect(router)
    assert asyncio.get_event_loop().time() - t0 < 2.0


# ---------------------------------------------------------------- pipeline


def _event(content, **kw):
    return ChatMessageCreatedEvent(
        conversation_id="c1",
        user_id="u1",
        message_id="m1",
        response_message_id="r1",
        content=content,
        **kw,
    )


def _pipeline():
    collector = MagicMock()
    collector.collect = AsyncMock(return_value=ContextMerger().merge())
    plan = ExecutionPlan(
        intent=IntentCategory.GENERAL_CHAT,
        mode=UserMode.DEFAULT,
        skill=Skill.GENERAL_CHAT,
        reasoning=ReasoningMode.DIRECT,
        tools=[],
        max_iterations=1,
    )
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(return_value=plan)
    engine = MagicMock()
    engine.execute = AsyncMock(return_value=MagicMock())
    builder = MagicMock()
    builder.build = MagicMock(return_value=MagicMock())
    cwm = MagicMock()
    cwm.manage = MagicMock(return_value=_prompt())

    async def gen(prompt, params):
        params.setdefault("_route", {}).update(provider="nvidia", fallback_used=False)
        yield "ok"

    router = MagicMock()
    router.generate_stream = gen
    streaming = MagicMock()

    async def consume(it, ctx):
        async for _ in it:
            pass

    streaming.stream = consume
    pipe = RequestPipeline(
        context_collector=collector,
        request_analyzer=analyzer,
        workflow_engine=engine,
        prompt_builder=builder,
        context_window_manager=cwm,
        generation_router=router,
        streaming_engine=streaming,
        publisher=MagicMock(),
        config=LLMServiceConfig(),
        logger=MagicMock(),
    )
    return pipe, collector, analyzer, engine


async def test_greeting_skips_context_collection_and_analysis_but_still_generates():
    pipe, collector, analyzer, engine = _pipeline()
    await pipe.run(_event("hi"))
    collector.collect.assert_not_called()
    analyzer.analyze.assert_not_called()
    engine.execute.assert_called_once()  # workflow + generation still run


@pytest.mark.parametrize("content", ["hi, explain kafka partitions", "what did we discuss yesterday?"])
async def test_substantive_request_keeps_full_pipeline(content):
    pipe, collector, analyzer, _ = _pipeline()
    await pipe.run(_event(content))
    collector.collect.assert_awaited_once()
    analyzer.analyze.assert_awaited_once()


async def test_attached_files_always_use_full_pipeline():
    from app.context.schemas import ContextBundle, RetrievalContext

    pipe, collector, analyzer, _ = _pipeline()
    collector.collect = AsyncMock(
        return_value=ContextBundle(retrieval=RetrievalContext(status="OK", requested_count=1, authorized_count=1, ready_count=1))
    )
    await pipe.run(_event("hi", file_ids=["f1"]))
    collector.collect.assert_awaited_once()
    analyzer.analyze.assert_awaited_once()


async def test_fast_path_can_be_disabled():
    pipe, collector, _, _ = _pipeline()
    pipe.config = LLMServiceConfig(enable_fast_path=False)
    await pipe.run(_event("hi"))
    collector.collect.assert_awaited_once()


async def test_timing_log_covers_every_stage_with_ids_and_no_content():
    pipe, *_ = _pipeline()
    await pipe.run(_event("hello there"))
    calls = [c for c in pipe.logger.info.call_args_list if c.args and c.args[0] == "pipeline_timing"]
    assert len(calls) == 1
    kw = calls[0].kwargs
    for key in (
        "queue_delay_ms",
        "context_ms",
        "analysis_ms",
        "workflow_ms",
        "prompt_ms",
        "generation_ms",
        "total_ms",
        "fast_path",
        "outcome",
        "conversation_id",
        "message_id",
        "response_message_id",
        "request_id",
    ):
        assert key in kw, key
    assert kw["outcome"] == "ok"
    assert "hello" not in str(kw)  # message content is never logged


async def test_pipeline_failure_still_logs_timing_and_publishes_error():
    pipe, _, analyzer, _ = _pipeline()
    analyzer.analyze = AsyncMock(side_effect=RuntimeError("x"))
    pipe.publisher.publish_error_response = AsyncMock()
    await pipe.run(_event("explain kafka in depth please"))
    pipe.publisher.publish_error_response.assert_awaited_once()
    assert any(
        c.args and c.args[0] == "pipeline_timing" and c.kwargs["outcome"] == "error"
        for c in pipe.logger.info.call_args_list
    )
