"""Follow-up web searches use the conversation; history survives a degraded memory service; cut-off answers
are never presented as complete."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from tests.doubles import built_prompt, manager_double, trimmed_prompt

import pytest

from app.config.settings import LLMServiceConfig
from app.context.schemas import ContextBundle, MemoryContext, Message, Role
from app.models.execution_plan import ExecutionPlan, IntentCategory, ReasoningMode, Skill, ToolCall, UserMode
from app.models.pipeline_context import PipelineContext
from app.models.request import ChatMessageCreatedEvent
from app.pipeline.request_pipeline import RequestPipeline
from app.providers.base import TRUNCATION_NOTICE
from app.providers.groq import GENERATION_MAX_TOKENS, MIN_OUTPUT_TOKENS, GroqAdapter, output_budget
from app.providers.nvidia import NVIDIAAdapter
from app.tools.query_rewrite import SearchQueryRewriter, fallback_query, needs_context
from app.tools.registry import ToolRegistry


def _turns():
    return [
        Message(role=Role.USER, content="I want to use DuckDuckGo search in my Python project. Which library?"),
        Message(role=Role.ASSISTANT, content="Use the duckduckgo-search package."),
    ]


class _Groq:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.prompts = reply, exc, []

    async def complete(self, messages, **kw):
        self.prompts.append(messages)
        if self.exc:
            raise self.exc
        return self.reply


# ------------------------------------------------------------------ rewriter


@pytest.mark.parametrize("msg", ["Check again and tell", "and what about pricing?", "tell me more", "ok do it again"])
def test_referential_or_short_messages_need_context(msg):
    assert needs_context(msg)


def test_self_contained_questions_do_not():
    assert not needs_context("What are the main differences between Kafka partitions and consumer groups in practice")


async def test_follow_up_is_rewritten_into_a_standalone_query_using_the_conversation():
    groq = _Groq(reply=json.dumps({"query": "duckduckgo-search python library usage"}))
    q = await SearchQueryRewriter(groq).rewrite("Check again and tell", _turns())
    assert q == "duckduckgo-search python library usage"
    sent = groq.prompts[0][1]["content"]
    assert "DuckDuckGo" in sent and "Check again and tell" in sent  # the model saw the topic and the follow-up


async def test_self_contained_message_is_untouched_and_costs_no_model_call():
    groq = _Groq(reply="{}")
    msg = "What are the main differences between Kafka partitions and consumer groups in practice"
    assert await SearchQueryRewriter(groq).rewrite(msg, _turns()) == msg
    assert groq.prompts == []


async def test_no_history_means_no_rewrite():
    groq = _Groq(reply="{}")
    assert await SearchQueryRewriter(groq).rewrite("Check again and tell", []) == "Check again and tell"
    assert groq.prompts == []


@pytest.mark.parametrize("groq", [_Groq(exc=TimeoutError()), _Groq(reply="not json"), _Groq(reply='{"query": ""}'), None])
async def test_model_failure_degrades_to_a_topic_prefixed_query_never_an_error(groq):
    q = await SearchQueryRewriter(groq).rewrite("Check again and tell", _turns())
    assert "DuckDuckGo" in q and q.endswith("Check again and tell")


def test_fallback_ignores_an_identical_previous_message():
    assert fallback_query("hi again", [Message(role=Role.USER, content="hi again")]) == "hi again"


# ------------------------------------------------------------------ pipeline


def _plan(mode=UserMode.DEFAULT, tools=None):
    return ExecutionPlan(intent=IntentCategory.GENERAL_CHAT, mode=mode, skill=Skill.GENERAL_CHAT,
                         reasoning=ReasoningMode.DIRECT, tools=tools or [], max_iterations=1)


def _pipeline(plan, bundle, rewriter, registry=None):
    seen = {}
    collector = MagicMock()
    collector.collect = AsyncMock(return_value=bundle)
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(return_value=plan)
    workflow = MagicMock()

    async def execute(p, ctx):
        seen["search_query"] = ctx.search_query
        seen["plan_tools"] = [dict(t.params) for t in p.tools]
        memory = ctx.context_bundle.memory if ctx.context_bundle else None
        seen["history"] = [m.content for m in (memory.short_term_messages if memory else [])]
        return MagicMock(tool_outputs=[], conversation_history=[])

    workflow.execute = execute
    builder = MagicMock()
    builder.build = MagicMock(return_value=built_prompt())
    cwm = manager_double()

    async def gen(prompt, params):
        yield "ok"

    router = MagicMock()
    router.generate_stream = gen
    streaming = MagicMock()

    async def consume(it, ctx):
        async for _ in it:
            pass

    streaming.stream = consume
    pub = MagicMock()
    pub.publish_error_response = AsyncMock()
    pipe = RequestPipeline(
        context_collector=collector, request_analyzer=analyzer, workflow_engine=workflow, prompt_builder=builder,
        context_window_manager=cwm, generation_router=router, streaming_engine=streaming, publisher=pub,
        config=LLMServiceConfig(), logger=MagicMock(), tool_registry=registry, query_rewriter=rewriter,
    )
    return pipe, seen, pub


def _event(text, **kw):
    return ChatMessageCreatedEvent(conversation_id="c", user_id="u", message_id="m", response_message_id="r", content=text, **kw)


HIST = [
    {"role": "user", "content": "How to use duckduckgo in my project?"},
    {"role": "assistant", "content": "Use duckduckgo-search."},
]


def _rewriter(answer="duckduckgo-search python"):
    r = MagicMock()
    r.rewrite = AsyncMock(return_value=answer)
    return r


async def test_history_from_the_event_fills_in_when_memory_is_degraded():
    bundle = ContextBundle(degraded=True, missing_sources=["memory"])  # memory service down: no memory at all
    pipe, seen, pub = _pipeline(_plan(), bundle, _rewriter())
    await pipe.run(_event("explain more", history=HIST))
    assert seen["history"] == ["How to use duckduckgo in my project?", "Use duckduckgo-search."]
    pub.publish_error_response.assert_not_called()


async def test_memory_history_wins_when_it_exists():
    bundle = ContextBundle(memory=MemoryContext(short_term_messages=[Message(role=Role.USER, content="from memory")]))
    pipe, seen, _ = _pipeline(_plan(), bundle, _rewriter())
    await pipe.run(_event("explain more", history=HIST))
    assert seen["history"] == ["from memory"]


async def test_web_search_mode_searches_the_rewritten_query_not_the_raw_follow_up():
    rewriter = _rewriter("duckduckgo-search python library usage")
    pipe, seen, _ = _pipeline(_plan(), ContextBundle(), rewriter)
    await pipe.run(_event("Check again and tell", mode_hint="web_search", history=HIST))
    assert seen["search_query"] == "duckduckgo-search python library usage"
    # the rewriter was given the conversation that came with the event
    given = [m.content for m in rewriter.rewrite.await_args.args[1]]
    assert given == ["How to use duckduckgo in my project?", "Use duckduckgo-search."]


async def test_web_search_handler_uses_the_rewritten_query():
    from app.workflow_engine.mode_handlers.web_search_handler import WebSearchHandler

    dispatcher = MagicMock()
    dispatcher.dispatch = AsyncMock(return_value=[])
    ctx = PipelineContext(conversation_id="c", user_id="u", message_id="m", request_id="r",
                          user_message="Check again and tell", search_query="duckduckgo-search python")
    await WebSearchHandler(tool_dispatcher=dispatcher).handle(_plan(UserMode.WEB_SEARCH), ctx)
    assert dispatcher.dispatch.await_args.args[0][-1].params["query"] == "duckduckgo-search python"
    ctx.search_query = None
    await WebSearchHandler(tool_dispatcher=dispatcher).handle(_plan(UserMode.WEB_SEARCH), ctx)
    assert dispatcher.dispatch.await_args.args[0][-1].params["query"] == "Check again and tell"


async def test_planner_chosen_search_calls_also_get_the_standalone_query():
    from tests.unit.test_tool_selection_and_rag import FakeTool

    reg = ToolRegistry()
    reg.register(FakeTool("web_search"))
    planned = [ToolCall(tool_name="web_search", params={"query": "Check again and tell"})]
    pipe, seen, _ = _pipeline(_plan(tools=planned), ContextBundle(), _rewriter("standalone q"), registry=reg)
    await pipe.run(_event("Check again and tell", history=HIST))
    assert seen["plan_tools"] == [{"query": "standalone q"}]


async def test_no_search_means_no_rewrite():
    rewriter = _rewriter()
    pipe, seen, _ = _pipeline(_plan(), ContextBundle(), rewriter)
    await pipe.run(_event("explain closures in more detail please", history=HIST))
    rewriter.rewrite.assert_not_called()
    assert seen["search_query"] is None


def test_event_without_history_is_still_valid():
    assert ChatMessageCreatedEvent(conversation_id="c", user_id="u", message_id="m", content="x").history == []


# ------------------------------------------------------------------ truncation + budgets


def _chunk(text=None, finish=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=finish)])


class _Stream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self.chunks:
                yield c

        return gen()


def _groq(chunks, captured):
    class _C:
        async def create(self, **kw):
            captured.update(kw)
            return _Stream(chunks)

    a = GroqAdapter(api_key="k", model="openai/gpt-oss-20b")
    a.client = SimpleNamespace(chat=SimpleNamespace(completions=_C()))
    a.logger = MagicMock()
    return a


USER = [{"role": "user", "content": "q"}]


async def test_groq_answer_that_hit_the_limit_is_marked_as_cut_off():
    out = [t async for t in _groq([_chunk("Partial answer"), _chunk(None, "length")], {}).stream(USER)]
    assert out == ["Partial answer", TRUNCATION_NOTICE]


async def test_groq_complete_answer_gets_no_notice():
    out = [t async for t in _groq([_chunk("Done."), _chunk(None, "stop")], {}).stream(USER)]
    assert out == ["Done."]


async def test_groq_budget_is_a_floor_and_reasoning_is_kept_low():
    seen = {}
    _ = [t async for t in _groq([_chunk("x", "stop")], seen).stream(USER, {"max_tokens": 1024})]
    assert seen["max_tokens"] == GENERATION_MAX_TOKENS and seen["reasoning_effort"] == "low"  # 1024 from another provider is raised
    seen.clear()
    _ = [t async for t in _groq([_chunk("x", "stop")], seen).stream(USER, {"max_tokens": 6000})]
    assert seen["max_tokens"] == 6000  # a larger caller budget is respected


def test_output_budget_shrinks_to_fit_the_tokens_per_minute_limit():
    # small prompt: full budget; big prompt: only what still fits under 8000 (prompt + output)
    assert output_budget(1024, prompt_tokens=500, tpm_limit=8000) == GENERATION_MAX_TOKENS
    assert output_budget(1024, prompt_tokens=5000, tpm_limit=8000) == 8000 - 5000 - 200
    assert output_budget(1024, prompt_tokens=7900, tpm_limit=8000) == MIN_OUTPUT_TOKENS  # never below a usable minimum
    assert output_budget(1024, prompt_tokens=7900, tpm_limit=0) == GENERATION_MAX_TOKENS  # 0 = uncapped tier


async def test_groq_request_never_asks_for_more_than_the_limit_allows():
    seen = {}
    big_prompt = [{"role": "user", "content": "x" * 15_000}]  # ~5000 estimated tokens
    adapter = _groq([_chunk("x", "stop")], seen)
    adapter.tpm_limit = 8000
    _ = [t async for t in adapter.stream(big_prompt)]
    assert seen["max_tokens"] < GENERATION_MAX_TOKENS and seen["max_tokens"] + 5000 <= 8000


def test_rate_limited_providers_give_an_actionable_message():
    from app.exceptions.provider import AllProvidersFailedError
    from app.utils.error_handler import ErrorHandler

    c = ErrorHandler().classify(AllProvidersFailedError("All generation providers failed. Groq error: [PROVIDER_RATE_LIMIT] Groq Rate Limit: 429"))
    assert c.error_code == "RATE_LIMIT_EXCEEDED" and "try again" in c.user_message and "Upstream" not in c.user_message


def test_heuristic_query_uses_the_topic_sentence_of_the_earlier_question():
    q = fallback_query("Check again and tell", [Message(role=Role.USER, content="How to use duckduckgo for my project? Give a complete module with tests and retries and caching.")])
    assert q == "How to use duckduckgo for my project? Check again and tell"


async def test_nvidia_answer_that_hit_the_limit_is_marked_as_cut_off():
    class _C:
        async def create(self, **kw):
            return _Stream([_chunk("Half an ans"), _chunk(None, "length")])

    a = NVIDIAAdapter(api_key="k", client=SimpleNamespace(chat=SimpleNamespace(completions=_C())))
    a.logger = MagicMock()
    out = [t async for t in a.stream(USER)]
    assert out == ["Half an ans", TRUNCATION_NOTICE]


def test_default_nvidia_budget_is_large_enough_for_long_answers():
    assert LLMServiceConfig(_env_file=None).nvidia_max_tokens >= 8192
