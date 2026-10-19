"""Response-identity contract: the assistant message id allocated by Conversation
Service (`response_message_id`) must travel unchanged from the request event through
every event this service emits, and stay distinct from the request/correlation id
(`message_id`).
"""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.models.pipeline_context import PipelineContext
from app.models.request import ChatMessageCreatedEvent
from app.producers.kafka_producer import KafkaPublisher
from app.services.streaming_service import StreamingEngine
from app.utils.error_handler import ErrorHandler

REQ = "00000000-0000-7000-8000-000000000001"  # request / correlation id (the user's message)
RESP = "00000000-0000-7000-8000-000000000002"  # pending assistant message id


def _event(**over) -> ChatMessageCreatedEvent:
    base = {
        "message_id": REQ,
        "response_message_id": RESP,
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "content": "What does the handbook say?",
        "file_ids": ["file-1"],
    }
    base.update(over)
    return ChatMessageCreatedEvent(**base)


def _ctx(**over) -> PipelineContext:
    return PipelineContext.from_event(_event(**over), request_id="req-1")


class CapturingPublisher(KafkaPublisher):
    """Real KafkaPublisher event construction; transport replaced by a list."""

    def __init__(self):
        self.sent: list[tuple[str, object]] = []
        self.config = SimpleNamespace(kafka_chunk_topic="chat.response.chunk", kafka_output_topic="chat.response.generated")
        self.logger = MagicMock()

    async def _send(self, topic, key, payload, **kwargs):
        self.sent.append((topic, payload))

    def of(self, topic):
        return [p for t, p in self.sent if t == topic]


async def _tokens(parts):
    for p in parts:
        yield p


# ---- request model / context -------------------------------------------------


def test_request_event_preserves_both_identities_through_json():
    wire = _event().model_dump_json()
    parsed = ChatMessageCreatedEvent.model_validate_json(wire)
    assert parsed.message_id == REQ and parsed.response_message_id == RESP
    assert parsed.message_id != parsed.response_message_id
    assert json.loads(wire)["schema_version"] == "2.3"


def test_legacy_event_without_response_id_still_deserialises_as_none():
    legacy = {"message_id": REQ, "conversation_id": "c", "user_id": "u", "content": "hi"}
    assert ChatMessageCreatedEvent(**legacy).response_message_id is None


def test_pipeline_context_carries_both_identities():
    ctx = _ctx()
    assert (ctx.message_id, ctx.response_message_id) == (REQ, RESP)


def test_redelivered_event_yields_the_same_response_id():
    """The id lives in the event, so a redelivery can never mint a new one."""
    wire = _event().model_dump_json()
    first = PipelineContext.from_event(ChatMessageCreatedEvent.model_validate_json(wire), request_id="r1")
    second = PipelineContext.from_event(ChatMessageCreatedEvent.model_validate_json(wire), request_id="r2")
    assert first.response_message_id == second.response_message_id == RESP


# ---- emitted events -------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_chunk_and_the_final_event_address_the_assistant_message():
    pub = CapturingPublisher()
    ctx = _ctx()
    ctx.selected_provider = "nvidia"
    ctx.engine_type = "mode_handler"

    text = await StreamingEngine(publisher=pub).stream(_tokens(["Hello", " ", "world", "."] * 30), ctx)

    chunks = pub.of("chat.response.chunk")
    assert len(chunks) >= 2, "test needs a multi-chunk stream"
    assert {c.response_message_id for c in chunks} == {RESP}
    assert {c.message_id for c in chunks} == {REQ}, "request id keeps its own meaning"
    assert all(c.schema_version == "2.1" for c in chunks)

    (final,) = pub.of("chat.response.generated")
    assert final.response_message_id == RESP
    assert final.request_message_id == REQ
    assert final.full_content == text and final.status == "success"
    assert final.response_id != RESP, "opaque response_id must not be repurposed as the message UUID"


@pytest.mark.asyncio
async def test_final_event_keeps_citations_full_content_and_provider_fields():
    pub = CapturingPublisher()
    ctx = _ctx()
    ctx.selected_provider = "gemini"
    ctx.generation_fallback_used = True
    ctx.context_bundle = SimpleNamespace(
        memory=None,
        graph=None,
        retrieval=SimpleNamespace(
            chunks=[
                SimpleNamespace(file_id="file-1", source_name="handbook.pdf", page_number=3, section="Intro", content="x"),
                SimpleNamespace(file_id="file-1", source_name="handbook.pdf", page_number=3, section="Intro", content="y"),
            ]
        ),
    )
    await pub.publish_response(ctx, "The answer.")

    (final,) = pub.of("chat.response.generated")
    assert final.full_content == "The answer."
    assert final.provider == "gemini" and final.generation_fallback_used is True
    assert final.citations == [{"file_id": "file-1", "source_name": "handbook.pdf", "page_number": 3, "section": "Intro"}]
    assert final.response_message_id == RESP and final.request_message_id == REQ


def test_error_event_uses_the_request_supplied_response_id_and_never_invents_one():
    ctx = _ctx()
    err = ErrorHandler().build_error_response_event(ctx, RuntimeError("provider exploded"))
    assert err.status == "error"
    assert err.response_message_id == RESP
    assert err.request_message_id == REQ
    assert err.response_id != RESP


@pytest.mark.asyncio
async def test_cancellation_event_is_correlated_to_the_assistant_message():
    pub = CapturingPublisher()
    await pub.publish_cancellation(_ctx(), reason="client_disconnected")
    (cancelled,) = pub.of("chat.response.cancelled")
    assert cancelled.message_id == REQ and cancelled.response_message_id == RESP


@pytest.mark.asyncio
async def test_memory_update_event_contract_is_unchanged():
    pub = CapturingPublisher()
    await pub.publish_memory_update(_ctx(), "assistant text")
    (mem,) = pub.of("memory.update.requested")
    dumped = mem.model_dump()
    assert "response_message_id" not in dumped
    assert dumped["user_message"] == "What does the handbook say?" and dumped["assistant_response"] == "assistant text"
