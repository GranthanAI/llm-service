"""P5d: chat.response.generated 2.2 / chat.response.cancelled billing fields, and exactly one terminal event per request."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config.settings import LLMServiceConfig
from app.exceptions.provider import ProviderTimeoutError
from app.models.billing import BillingGrant
from app.models.pipeline_context import PipelineContext
from app.models.request import ChatMessageCreatedEvent
from app.models.response import ChatResponseGeneratedEvent
from app.producers.kafka_producer import KafkaPublisher
from app.providers.usage_hook import report_usage
from app.services.streaming_service import StreamingEngine
from app.usage.collector import UsageCollector, use_collector, usage_purpose
from app.utils.cancellation import CancellationToken
from app.utils.error_handler import ErrorHandler
from tests.unit.test_request_pipeline import make_event, make_pipeline

TERMINAL_TOPICS = ("chat.response.generated", "chat.response.cancelled")

GRANT = {"hold_id": "hold-1", "requested_model": "claude-sonnet-5-5", "granted_group": "best",
         "granted_model": "claude-sonnet-5-5", "max_context_tokens": 128000, "funded_by": "plan",
         "downgraded": True, "enforcement": "enforce"}


class FakeProducer:
    def __init__(self, fail_topics=()):
        self.sent: list[tuple[str, dict]] = []
        self.fail_topics = set(fail_topics)

    async def send_and_wait(self, topic, key, value, headers=None):
        if topic in self.fail_topics:
            raise RuntimeError(f"cannot publish to {topic}")
        self.sent.append((topic, json.loads(value)))

    def of(self, *topics):
        return [v for t, v in self.sent if t in topics]


def publisher(producer=None):
    config = LLMServiceConfig()
    p = KafkaPublisher(config=config, producer=producer or FakeProducer())
    p._started = True
    async def run_once(fn, **_kw):
        return await fn()

    p._retry_manager = MagicMock(execute_with_retry=run_once)
    return p


def ctx_for(billing=None, message_id="msg-1"):
    event = ChatMessageCreatedEvent.model_validate({
        "message_id": message_id, "response_message_id": "resp-1", "conversation_id": "c1", "user_id": "u1",
        "content": "hi", "billing": billing})
    ctx = PipelineContext.from_event(event, request_id="req-12345678901234")
    ctx.usage_collector = UsageCollector()
    return ctx


def add_calls(ctx, answer_tokens=(1000, 120)):
    with use_collector(ctx.usage_collector):
        with usage_purpose("request_analysis"):
            report_usage(None, provider="groq", model="openai/gpt-oss-20b", input_tokens=200, output_tokens=25)
        with usage_purpose("answer"):
            report_usage(None, provider="anthropic", model="claude-sonnet-5-5", input_tokens=answer_tokens[0],
                         output_tokens=answer_tokens[1], cached_input_tokens=300)


# --- fields on success / error / cancel ------------------------------------------------------------------------------

async def test_success_event_carries_the_billing_fields():
    prod = FakeProducer()
    ctx = ctx_for(GRANT)
    ctx.answered_model = "gpt-6.1-sol"                 # answered after a fallback inside the group
    ctx.selected_provider = "openai:gpt-6.1-sol"
    add_calls(ctx)
    await publisher(prod).publish_response(ctx, "answer text")
    (ev,) = prod.of("chat.response.generated")
    assert ev["schema_version"] == "2.2" and ev["status"] == "success"
    assert (ev["hold_id"], ev["model_group"], ev["model"], ev["downgraded"]) == ("hold-1", "best", "gpt-6.1-sol", True)
    assert [c["purpose"] for c in ev["calls"]] == ["request_analysis", "answer"]
    assert ev["calls"][1] == {"provider": "anthropic", "model": "claude-sonnet-5-5", "purpose": "answer",
                              "input_tokens": 1000, "cached_input_tokens": 300, "output_tokens": 120, "usage_estimated": False}


async def test_error_event_has_status_error_and_the_calls_made_so_far():
    prod = FakeProducer()
    ctx = ctx_for(GRANT)
    add_calls(ctx, answer_tokens=(1000, 0))
    err = ErrorHandler().build_error_response_event(ctx, ProviderTimeoutError("slow", provider="groq"))
    await publisher(prod).publish_error_response(ctx, err)
    (ev,) = prod.of("chat.response.generated")
    assert ev["status"] == "error" and ev["hold_id"] == "hold-1" and ev["model_group"] == "best"
    assert ev["model"] is None                                     # nothing answered
    assert len(ev["calls"]) == 2 and ev["calls"][1]["output_tokens"] == 0


async def test_cancel_event_has_the_calls_including_partial_output_tokens():
    prod = FakeProducer()
    pub = publisher(prod)
    ctx = ctx_for(GRANT)
    ctx.selected_provider = "anthropic:claude-sonnet-5-5"
    ctx.final_prompt_messages = [{"role": "user", "content": "Explain Kafka in detail please."}]
    with use_collector(ctx.usage_collector), usage_purpose("request_analysis"):
        report_usage(None, provider="groq", model="openai/gpt-oss-20b", input_tokens=200, output_tokens=25)
    token = CancellationToken()
    seen = []

    async def tokens():
        for t in ["Kafka ", "is ", "a ", "log "]:
            seen.append(t)
            if len(seen) == 3:
                token.cancel("client_disconnected")
            yield t

    out = await StreamingEngine(publisher=pub).stream(tokens(), ctx, cancellation_token=token)
    assert out
    (ev,) = prod.of("chat.response.cancelled")
    assert ev["schema_version"] == "2.2" and ev["hold_id"] == "hold-1" and ev["model_group"] == "best"
    answer = [c for c in ev["calls"] if c["purpose"] == "answer"]
    assert len(answer) == 1 and answer[0]["output_tokens"] > 0 and answer[0]["usage_estimated"] is True
    assert answer[0]["provider"] == "anthropic"
    assert prod.of("chat.response.generated") == []                # a cancelled request has no success event


async def test_legacy_request_has_null_hold_and_group_but_filled_model_and_calls():
    prod = FakeProducer()
    ctx = ctx_for(None)
    ctx.selected_provider = "groq:openai/gpt-oss-120b"
    with use_collector(ctx.usage_collector), usage_purpose("answer"):
        report_usage(None, provider="groq", model="openai/gpt-oss-120b", input_tokens=400, output_tokens=50)
    await publisher(prod).publish_response(ctx, "x")
    (ev,) = prod.of("chat.response.generated")
    assert (ev["hold_id"], ev["model_group"], ev["downgraded"]) == (None, None, False)
    assert ev["model"] == "openai/gpt-oss-120b" and len(ev["calls"]) == 1


async def test_memory_update_carries_the_request_message_id():
    prod = FakeProducer()
    await publisher(prod).publish_memory_update(ctx_for(None, message_id="req-9"), "answer")
    (ev,) = prod.of("memory.update.requested")
    assert ev["request_message_id"] == "req-9"


# --- legacy snapshot / compatibility --------------------------------------------------------------------------------

OLD_GENERATED_FIELDS = {
    "event_type", "schema_version", "response_id", "conversation_id", "user_id", "request_message_id",
    "response_message_id", "full_content", "provider", "generation_fallback_used", "mode", "skill", "engine_type",
    "usage", "cost_usd", "latency_ms", "ttft_ms", "finish_reason", "tools_used", "context_sources", "citations",
    "trace_context", "status", "error_code", "error_message", "timestamp"}
NEW_GENERATED_FIELDS = {"hold_id", "model_group", "model", "downgraded", "calls"}


async def test_legacy_event_differs_from_the_old_one_only_by_the_new_optional_fields():
    prod = FakeProducer()
    ctx = ctx_for(None)
    await publisher(prod).publish_response(ctx, "x")
    (ev,) = prod.of("chat.response.generated")
    assert set(ev) == OLD_GENERATED_FIELDS | NEW_GENERATED_FIELDS
    assert (ev["hold_id"], ev["model_group"], ev["downgraded"], ev["calls"]) == (None, None, False, [])
    assert ev["schema_version"] == "2.2" and ev["status"] == "success" and ev["finish_reason"] == "stop"
    assert ev["provider"] == "nvidia" and ev["mode"] == "default" and ev["usage"] == {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_old_consumers_ignore_the_new_fields():
    from pydantic import BaseModel

    class OldGenerated(BaseModel):                       # what conversation-service / memory-service parse today
        event_type: str
        response_id: str
        conversation_id: str
        request_message_id: str
        full_content: str | None = None
        status: str = "success"

    new = ChatResponseGeneratedEvent(response_id="r", conversation_id="c", user_id="u", request_message_id="m",
                                     hold_id="h", model_group="best", model="claude-sonnet-5-5", calls=[{"purpose": "answer"}])
    old = OldGenerated.model_validate_json(new.model_dump_json())
    assert old.request_message_id == "m" and old.status == "success"


# --- exactly one terminal event per request --------------------------------------------------------------------------

async def test_a_second_terminal_event_for_the_same_request_is_dropped():
    prod = FakeProducer()
    pub = publisher(prod)
    ctx = ctx_for(GRANT)
    await pub.publish_response(ctx, "answer")
    await pub.publish_error_response(ctx, ErrorHandler().build_error_response_event(ctx, RuntimeError("late")))
    await pub.publish_cancellation(ctx, "client_disconnected")
    assert len(prod.of(*TERMINAL_TOPICS)) == 1
    # a redelivery (new context, same request message id) in the same process is dropped as well
    await pub.publish_response(ctx_for(GRANT, message_id="msg-1"), "answer again")
    assert len(prod.of(*TERMINAL_TOPICS)) == 1
    await pub.publish_response(ctx_for(GRANT, message_id="msg-2"), "another request")
    assert len(prod.of(*TERMINAL_TOPICS)) == 2


def real_pipeline(prod):
    pipeline, mocks = make_pipeline()
    pub = publisher(prod)
    pipeline.publisher = pub
    pipeline.streaming_engine = StreamingEngine(publisher=pub)

    async def tokens(*_a, **_k):
        yield "Hello"
        yield " world"

    mocks["generation_router"].generate_stream = MagicMock(side_effect=lambda *a, **k: tokens())
    return pipeline, mocks


async def test_success_path_publishes_exactly_one_terminal_event():
    prod = FakeProducer()
    pipeline, _ = real_pipeline(prod)
    await pipeline.run(make_event())
    assert len(prod.of(*TERMINAL_TOPICS)) == 1 and prod.of("chat.response.generated")[0]["status"] == "success"


async def test_a_failure_after_the_success_event_does_not_add_an_error_event():
    prod = FakeProducer(fail_topics={"memory.update.requested"})        # the memory publish fails AFTER the answer
    pipeline, _ = real_pipeline(prod)
    await pipeline.run(make_event())
    events = prod.of(*TERMINAL_TOPICS)
    assert len(events) == 1 and events[0]["status"] == "success"


async def test_provider_timeout_publishes_exactly_one_error_event():
    prod = FakeProducer()
    pipeline, mocks = real_pipeline(prod)

    async def timing_out(*_a, **_k):
        raise ProviderTimeoutError("no first token within 12s", provider="groq")
        yield  # pragma: no cover

    mocks["generation_router"].generate_stream = MagicMock(side_effect=lambda *a, **k: timing_out())
    await pipeline.run(make_event())
    (ev,) = prod.of(*TERMINAL_TOPICS)
    assert ev["status"] == "error" and ev["error_code"]
    assert prod.of("memory.update.requested") == []


async def test_a_grant_with_no_usable_model_group_ends_in_one_error_event():
    prod = FakeProducer()
    pipeline, _ = real_pipeline(prod)
    event = make_event()
    event.billing = BillingGrant(**GRANT)                 # group routing is not configured in this pipeline
    await pipeline.run(event)
    (ev,) = prod.of(*TERMINAL_TOPICS)
    assert ev["status"] == "error" and ev["hold_id"] == "hold-1" and ev["model_group"] == "best"


async def test_consumer_restart_before_the_terminal_event_still_yields_exactly_one():
    prod = FakeProducer()
    pipeline, mocks = real_pipeline(prod)
    event = make_event()

    async def crash(*_a, **_k):
        raise asyncio.CancelledError()                     # the process dies mid-request: no terminal event is sent
        yield  # pragma: no cover

    good = mocks["generation_router"].generate_stream
    mocks["generation_router"].generate_stream = MagicMock(side_effect=lambda *a, **k: crash())
    with pytest.raises(asyncio.CancelledError):
        await pipeline.run(event)
    assert prod.of(*TERMINAL_TOPICS) == []
    mocks["generation_router"].generate_stream = good      # the restarted consumer gets the message again
    await pipeline.run(event)
    assert len(prod.of(*TERMINAL_TOPICS)) == 1


async def test_restart_after_the_terminal_event_redelivers_with_the_same_identity():
    """Across a restart delivery is at-least-once: the repeat carries the same request_message_id, so downstream
    (billing dedupes on it) counts it once."""
    first, second = FakeProducer(), FakeProducer()
    for prod in (first, second):                           # a new process = a new publisher with an empty guard
        pipeline, _ = real_pipeline(prod)
        await pipeline.run(make_event())
    a, b = first.of("chat.response.generated")[0], second.of("chat.response.generated")[0]
    assert a["request_message_id"] == b["request_message_id"] == "msg_1"
