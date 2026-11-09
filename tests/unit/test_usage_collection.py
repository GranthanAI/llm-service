"""P5c: real per-call token usage for every model call, with the purpose of each call."""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.providers.gemini import GeminiAdapter, gemini_usage_numbers
from app.providers.groq import GroqAdapter
from app.providers.nvidia import NVIDIAAdapter
from app.providers.usage_hook import report_usage
from app.usage.collector import UsageCollector, current_collector, usage_purpose, use_collector
from app.utils.metrics import USAGE_ESTIMATED_TOTAL
from tests.unit.test_request_pipeline import make_event, make_pipeline

MSGS = [{"role": "user", "content": "Explain Kafka partitions in two sentences please."}]


class Stream:
    def __init__(self, items, error=None):
        self.items, self.error = items, error

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for i in self.items:
            yield i
        if self.error:
            raise self.error


async def drain(agen):
    return [t async for t in agen]


def estimated_count(provider, model):
    return USAGE_ESTIMATED_TOTAL.labels(provider=provider, model=model)._value.get()


# --- every adapter records exactly what the provider reported -------------------------------------------------------

async def test_groq_stream_usage_from_the_final_chunk_and_x_groq():
    chunks = [NS(choices=[NS(delta=NS(content="Hi"), finish_reason=None)], usage=None, x_groq=None),
              NS(choices=[NS(delta=NS(content=None), finish_reason="stop")], usage=None,
                 x_groq=NS(usage=NS(prompt_tokens=321, completion_tokens=45, prompt_tokens_details=NS(cached_tokens=100))))]

    async def create(**kw):
        return Stream(chunks)

    collector = UsageCollector()
    adapter = GroqAdapter(api_key="k", model="openai/gpt-oss-120b", client=NS(chat=NS(completions=NS(create=create))))
    with use_collector(collector), usage_purpose("answer"):
        assert "".join(await drain(adapter.stream(MSGS, {}))) == "Hi"
    (c,) = collector.calls
    assert (c.provider, c.model, c.purpose) == ("groq", "openai/gpt-oss-120b", "answer")
    assert (c.input_tokens, c.cached_input_tokens, c.output_tokens, c.usage_estimated) == (321, 100, 45, False)


async def test_groq_execute_usage_equals_the_response_usage_object():
    resp = NS(choices=[NS(message=NS(content='{"a":1}'))], usage=NS(prompt_tokens=77, completion_tokens=9, total_tokens=86,
                                                                  prompt_tokens_details=None))

    async def create(**kw):
        return resp

    collector = UsageCollector()
    adapter = GroqAdapter(api_key="k", client=NS(chat=NS(completions=NS(create=create))))
    with use_collector(collector), usage_purpose("request_analysis"):
        await adapter.execute(MSGS, {})
    c = collector.calls[0]
    assert (c.purpose, c.input_tokens, c.output_tokens, c.cached_input_tokens) == ("request_analysis", 77, 9, 0)


async def test_gemini_stream_usage_metadata_from_the_last_chunk_counts_thinking_as_output():
    meta = NS(prompt_token_count=500, candidates_token_count=120, thoughts_token_count=80, cached_content_token_count=200)
    chunks = [NS(text="A", usage_metadata=None), NS(text="B", usage_metadata=meta)]

    async def gen(**kw):
        return Stream(chunks)

    client = NS(aio=NS(models=NS(generate_content_stream=gen)))
    collector = UsageCollector()
    with use_collector(collector):
        out = await drain(GeminiAdapter(api_key="k", model="gemini-2.5-flash", client=client).stream(MSGS, {}))
    assert "".join(out) == "AB"
    c = collector.calls[0]
    assert (c.provider, c.model) == ("google", "gemini-2.5-flash")            # billing's provider name
    assert (c.input_tokens, c.output_tokens, c.cached_input_tokens, c.usage_estimated) == (500, 200, 200, False)
    assert gemini_usage_numbers(None) == (None, None, 0)


async def test_nvidia_stream_asks_for_usage_and_records_the_final_chunk():
    seen = {}
    chunks = [NS(choices=[NS(delta=NS(content="x"), finish_reason=None)], usage=None),
              NS(choices=[], usage=NS(prompt_tokens=40, completion_tokens=6, prompt_tokens_details=NS(cached_tokens=8)))]

    async def create(**kw):
        seen.update(kw)
        return Stream(chunks)

    collector = UsageCollector()
    adapter = NVIDIAAdapter(api_key="k", client=NS(chat=NS(completions=NS(create=create))))
    with use_collector(collector):
        await drain(adapter.stream(MSGS, {}))
    assert seen["stream_options"] == {"include_usage": True}
    c = collector.calls[0]
    assert (c.provider, c.input_tokens, c.output_tokens, c.cached_input_tokens) == ("nvidia", 40, 6, 8)


async def test_analysis_client_records_the_groq_json_call():
    from app.request_analyzer.groq_client import GroqAnalysisClient

    resp = NS(choices=[NS(message=NS(content="{}"))], usage=NS(prompt_tokens=1200, completion_tokens=90))

    async def create(**kw):
        return resp

    collector = UsageCollector()
    client = GroqAnalysisClient(api_key="k", client=NS(chat=NS(completions=NS(create=create))))
    with use_collector(collector), usage_purpose("request_analysis"):
        await client.complete([{"role": "user", "content": "x"}])
    c = collector.calls[0]
    assert (c.provider, c.model, c.purpose, c.input_tokens, c.output_tokens) == ("groq", client.model, "request_analysis", 1200, 90)


# --- missing usage: estimated, flagged, counted -----------------------------------------------------------------------

def test_missing_usage_is_estimated_with_the_token_counter_flagged_and_counted():
    collector = UsageCollector()
    before = estimated_count("groq", "est-model")
    with use_collector(collector), usage_purpose("answer"):
        report_usage(None, provider="groq", model="est-model", input_tokens=None, output_tokens=None,
                     messages=MSGS, output_text="Partitions split a topic so consumers can read in parallel.")
    (c,) = collector.calls
    assert c.usage_estimated is True and c.input_tokens > 0 and c.output_tokens > 0
    assert c.output_tokens < len("Partitions split a topic so consumers can read in parallel.")   # tokens, not characters
    assert estimated_count("groq", "est-model") == before + 1


def test_provider_reported_usage_is_never_flagged_and_partial_usage_estimates_only_the_missing_part():
    collector = UsageCollector()
    with use_collector(collector):
        report_usage(None, provider="openai", model="m", input_tokens=10, output_tokens=5)
        report_usage(None, provider="openai", model="m2", input_tokens=10, output_tokens=None, output_text="some words here")
    assert collector.calls[0].usage_estimated is False
    assert collector.calls[1].usage_estimated is True and collector.calls[1].input_tokens == 10


def test_no_collector_means_no_report_and_no_error():
    assert current_collector() is None
    report_usage(None, provider="groq", model="m", input_tokens=1, output_tokens=1)


def test_catalog_ids_and_billing_provider_names_are_used():
    collector = UsageCollector(model_resolver=lambda p, m: {"google": {"g-pm": "gemini-3.1-pro"}}.get(p, {}).get(m, m))
    with use_collector(collector):
        report_usage(None, provider="gemini", model="g-pm", input_tokens=1, output_tokens=1)
        report_usage(None, provider="groq", model="qwen/qwen3.8-27b", input_tokens=1, output_tokens=1)
    assert [(c.provider, c.model) for c in collector.calls] == [("google", "gemini-3.1-pro"), ("groq", "qwen/qwen3.8-27b")]


def test_unknown_purpose_is_a_bug():
    with pytest.raises(ValueError):
        with usage_purpose("chat"):
            pass


# --- a whole request: purposes, retries --------------------------------------------------------------------------------

async def test_a_smart_request_reports_request_analysis_workflow_and_one_answer():
    pipeline, mocks = make_pipeline()
    captured = {}

    async def analyze(ctx):
        report_usage(None, provider="groq", model="openai/gpt-oss-20b", input_tokens=2000, output_tokens=250)
        return mocks["plan"]

    async def execute(plan, ctx):
        report_usage(None, provider="groq", model="openai/gpt-oss-20b", input_tokens=900, output_tokens=60)
        return mocks["workflow_result"]

    async def stream(token_iter, ctx):
        async for _ in token_iter:
            pass
        captured["calls"] = ctx.usage_collector.snapshot()
        return "Hello world"

    async def answer_tokens(*_a, **_k):
        report_usage(None, provider="groq", model="openai/gpt-oss-120b", input_tokens=6000, output_tokens=700)
        yield "Hello"

    mocks["request_analyzer"].analyze.side_effect = analyze
    mocks["workflow_engine"].execute.side_effect = execute
    mocks["streaming_engine"].stream.side_effect = stream
    mocks["generation_router"].generate_stream = MagicMock(side_effect=lambda *a, **k: answer_tokens())
    await pipeline.run(make_event())
    calls = captured["calls"]
    assert [c["purpose"] for c in calls] == ["request_analysis", "workflow", "answer"]
    assert sum(1 for c in calls if c["purpose"] == "answer") == 1
    assert calls[2] == {"provider": "groq", "model": "openai/gpt-oss-120b", "purpose": "answer", "input_tokens": 6000,
                        "cached_input_tokens": 0, "output_tokens": 700, "usage_estimated": False}
    assert current_collector() is None            # the request's collector does not leak into the next request


async def test_a_retried_call_reports_both_attempts():
    from app.context_window.models import PromptSection, TrimmedPrompt
    from app.providers.chain import ChainEntry, ProviderChain

    class Failing:
        """Reached the provider (which billed the prompt) but produced no text."""
        async def stream(self, messages, params=None):
            report_usage(params, provider="groq", model="openai/gpt-oss-120b", input_tokens=5000, output_tokens=0)
            raise RuntimeError("model called a tool")
            yield  # pragma: no cover

    class Good:
        async def stream(self, messages, params=None):
            report_usage(params, provider="google", model="gemini-2.5-flash", input_tokens=5000, output_tokens=300)
            yield "answer"

    chain = ProviderChain([ChainEntry("groq:a", Failing(), None, 1.0), ChainEntry("gemini", Good(), None, 1.0)])
    prompt = TrimmedPrompt(sections=[PromptSection(name="q", content="x", priority=1, token_count=1, trimmable=False)],
                           messages=MSGS, total_tokens=1, mode="default", target_model="m")
    collector = UsageCollector()
    with use_collector(collector), usage_purpose("answer"):
        assert await drain(chain.generate_stream(prompt, {})) == ["answer"]
    assert [(c.model, c.input_tokens, c.output_tokens) for c in collector.calls] == [
        ("openai/gpt-oss-120b", 5000, 0), ("gemini-2.5-flash", 5000, 300)]
    assert collector.totals("answer") == (10000, 300)
