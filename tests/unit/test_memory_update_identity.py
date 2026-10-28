"""memory.update.requested carries the real message ids and times (additive fields), so memory-service can store the
two messages of a turn in order and compare summary watermarks with conversation-service's UUIDv7 ids."""

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.models.pipeline_context import PipelineContext
from app.models.request import ChatMessageCreatedEvent
from app.producers.kafka_producer import KafkaPublisher


def uuid7_at(ts: datetime, rand: int = 0) -> str:
    ms = int(ts.timestamp() * 1000)
    value = (ms << 80) | (0x7 << 76) | ((rand & 0xFFF) << 64) | (0b10 << 62) | (rand & ((1 << 62) - 1))
    return str(uuid.UUID(int=value))


SENT_AT = datetime(2026, 10, 1, 9, 0, 0, 123000, tzinfo=UTC)


def make_ctx(message_id: str, response_message_id: str | None = None) -> PipelineContext:
    event = ChatMessageCreatedEvent(
        message_id=message_id, response_message_id=response_message_id, conversation_id="conv-1",
        user_id="user-1", content="hello",
    )
    return PipelineContext.from_event(event, request_id="req-1")


class CapturingPublisher(KafkaPublisher):
    def __init__(self):
        self.sent = []
        self.config = SimpleNamespace(kafka_chunk_topic="chat.response.chunk", kafka_output_topic="chat.response.generated")
        self.logger = MagicMock()

    async def _send(self, topic, key, payload, **kwargs):
        self.sent.append((topic, payload))


async def publish(ctx):
    pub = CapturingPublisher()
    await pub.publish_memory_update(ctx, "the answer")
    (event,) = [p for t, p in pub.sent if t == "memory.update.requested"]
    return event


@pytest.mark.asyncio
async def test_event_carries_real_ids_and_times():
    user_id, assistant_id = uuid7_at(SENT_AT, 1), uuid7_at(SENT_AT, 2)
    event = await publish(make_ctx(user_id, assistant_id))
    assert event.user_message_id == user_id and event.assistant_message_id == assistant_id
    assert event.user_created_at == SENT_AT.replace(microsecond=123000)       # the UUIDv7 time of the user message
    assert event.assistant_created_at >= event.user_created_at
    assert event.assistant_created_at > SENT_AT                               # completion time, after the send


@pytest.mark.asyncio
async def test_non_uuid7_message_id_falls_back_to_the_request_start_time():
    ctx = make_ctx(str(uuid.uuid4()), str(uuid.uuid4()))
    event = await publish(ctx)
    assert event.user_created_at == ctx.started_at
    assert event.assistant_created_at >= event.user_created_at


@pytest.mark.asyncio
async def test_missing_response_message_id_is_omitted_not_invented():
    event = await publish(make_ctx(uuid7_at(SENT_AT), None))
    assert event.assistant_message_id is None


@pytest.mark.asyncio
async def test_legacy_fields_and_topic_are_unchanged():
    event = await publish(make_ctx(uuid7_at(SENT_AT, 1), uuid7_at(SENT_AT, 2)))
    dumped = event.model_dump(mode="json")
    assert event.event_type == "memory.update.requested"
    for legacy_field in ("conversation_id", "user_id", "user_message", "assistant_response", "mode", "timestamp"):
        assert legacy_field in dumped
    assert dumped["user_message"] == "hello" and dumped["assistant_response"] == "the answer"


def test_new_fields_are_optional_so_old_producers_and_tests_still_validate():
    from app.models.response import MemoryUpdateRequestedEvent
    old = MemoryUpdateRequestedEvent(conversation_id="c", user_id="u", user_message="q", assistant_response="a")
    assert old.user_message_id is None and old.assistant_created_at is None
    assert datetime.now(UTC) - old.timestamp < timedelta(seconds=5)
