"""
Unit and mock tests for Kafka Consumer, Producer, Offset Commits, Idempotency, and DLQ.
Implements Phase 3 deliverables verification.
"""

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiokafka import ConsumerRecord, TopicPartition

from app.config.settings import LLMServiceConfig
from app.consumers.chat_consumer import ChatConsumer
from app.consumers.kafka_consumer import KafkaConsumerEngine
from app.context.schemas import ContextBundle
from app.exceptions.grpc import GRPCUnavailableError
from app.models.execution_plan import ExecutionPlan, UserMode
from app.models.pipeline_context import PipelineContext
from app.models.request import ChatMessageCreatedEvent
from app.models.response import (
    ChatMessageDLQEvent,
    ChatResponseChunkEvent,
    UsageMetrics,
)
from app.producers.kafka_producer import KafkaPublisher
from app.utils.cache import TTLCache
from app.utils.retry import RetryManager, RetryPolicy


@pytest.fixture
def mock_config() -> LLMServiceConfig:
    return LLMServiceConfig(
        kafka_bootstrap_servers="localhost:9092",
        kafka_consumer_group="test-group",
        kafka_input_topic="chat.message.created",
        kafka_output_topic="chat.response.generated",
        kafka_chunk_topic="chat.response.chunk",
        kafka_dlq_topic="chat.message.dlq",
    )


@pytest.mark.asyncio
async def test_ttl_cache_expiration_and_lru():
    """Verify TTLCache expiration and LRU eviction."""
    cache = TTLCache(max_size=2, default_ttl_seconds=1)
    cache.set("k1", "v1", ttl_seconds=1)
    cache.set("k2", "v2", ttl_seconds=1)
    assert cache.get("k1") == "v1"
    assert cache.contains("k2") is True

    # Eviction on exceeding max_size
    cache.set("k3", "v3", ttl_seconds=1)
    assert cache.size() <= 2

    # Expiration
    await asyncio.sleep(1.1)
    assert cache.get("k1") is None
    assert cache.get("k2") is None


@pytest.mark.asyncio
async def test_retry_manager_success_and_classification():
    """Verify retry manager executes successfully and respects retry classification."""
    manager = RetryManager(RetryPolicy(max_attempts=3, initial_delay_ms=10))

    call_count = 0

    async def flaky_call():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise GRPCUnavailableError("temporary network outage", service="memory")
        return "success"

    result = await manager.execute_with_retry(flaky_call, operation_name="test_flaky")
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_manager_permanent_failure():
    """Verify retry manager aborts immediately on non-retriable exceptions."""
    manager = RetryManager(RetryPolicy(max_attempts=3, initial_delay_ms=10))

    call_count = 0

    async def permanent_error_call():
        nonlocal call_count
        call_count += 1
        raise ValueError("Invalid schema")

    with pytest.raises(ValueError, match="Invalid schema"):
        await manager.execute_with_retry(permanent_error_call, operation_name="test_permanent")

    assert call_count == 1  # Should not retry ValueError


@pytest.mark.asyncio
async def test_kafka_publisher_publish_response_and_headers(mock_config: LLMServiceConfig):
    """Verify KafkaPublisher formats final response, headers, and produces to Kafka."""
    mock_aioproducer = AsyncMock()
    publisher = KafkaPublisher(config=mock_config, producer=mock_aioproducer)
    await publisher.start()

    ctx = PipelineContext(
        conversation_id="conv_100",
        user_id="user_200",
        message_id="msg_300",
        request_id="req_400",
        trace_id="00-testtrace-testspan-01",
        user_message="Hello",
        plan=ExecutionPlan(mode=UserMode.TUTOR),
        context_bundle=ContextBundle(),
        usage=UsageMetrics(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    await publisher.publish_response(ctx, full_response="Hello, world!")

    mock_aioproducer.send_and_wait.assert_called_once()
    call_kwargs = mock_aioproducer.send_and_wait.call_args.kwargs
    assert call_kwargs["topic"] == "chat.response.generated"
    assert call_kwargs["key"] == b"conv_100"

    # Verify serialized payload
    payload = json.loads(call_kwargs["value"].decode("utf-8"))
    assert payload["conversation_id"] == "conv_100"
    assert payload["full_content"] == "Hello, world!"
    assert payload["mode"] == "tutor"
    assert payload["usage"]["total_tokens"] == 15

    # Verify W3C headers
    headers_dict = dict(call_kwargs["headers"])
    assert headers_dict.get("traceparent") == b"00-testtrace-testspan-01"

    await publisher.stop()


@pytest.mark.asyncio
async def test_kafka_publisher_publish_chunk_and_dlq(mock_config: LLMServiceConfig):
    """Verify chunk and DLQ publishing."""
    mock_aioproducer = AsyncMock()
    publisher = KafkaPublisher(config=mock_config, producer=mock_aioproducer)
    await publisher.start()

    # Publish chunk
    chunk = ChatResponseChunkEvent(
        conversation_id="conv_100",
        user_id="user_200",
        message_id="msg_300",
        chunk_index=0,
        sequence_number=1,
        content="First token",
    )
    await publisher.publish_chunk(chunk)

    # Publish DLQ
    dlq = ChatMessageDLQEvent(
        original_topic="chat.message.created",
        error_type="DeserializationError",
        error_message="corrupt bytes",
        failed_at=datetime.now(UTC),
    )
    await publisher.publish_dlq(dlq)

    assert mock_aioproducer.send_and_wait.call_count == 2
    await publisher.stop()


@pytest.mark.asyncio
async def test_consumer_idempotency_duplicate_handling(mock_config: LLMServiceConfig):
    """Verify consumer skips duplicate message_id and commits offset."""
    mock_aioconsumer = AsyncMock()
    mock_publisher = AsyncMock()
    handler_mock = AsyncMock()

    consumer_engine = KafkaConsumerEngine(
        config=mock_config,
        publisher=mock_publisher,
        event_handler=handler_mock,
        consumer=mock_aioconsumer,
    )
    consumer_engine._running = True

    event_payload = {
        "event_type": "chat.message.created",
        "schema_version": "2.0",
        "message_id": "duplicate_msg_1",
        "conversation_id": "conv_1",
        "user_id": "user_1",
        "content": "Test content",
    }
    raw_bytes = json.dumps(event_payload).encode("utf-8")

    record = ConsumerRecord(
        topic="chat.message.created",
        partition=0,
        offset=10,
        timestamp=0,
        timestamp_type=0,
        key=b"conv_1",
        value=raw_bytes,
        checksum=None,
        serialized_key_size=6,
        serialized_value_size=len(raw_bytes),
        headers=[],
    )
    tp = TopicPartition("chat.message.created", 0)

    # First pass: normal processing
    await consumer_engine._process_single_message(record, tp)
    assert handler_mock.call_count == 1
    assert mock_aioconsumer.commit.call_count == 1

    # Second pass with same message_id: should skip handler, but still commit offset
    await consumer_engine._process_single_message(record, tp)
    assert handler_mock.call_count == 1  # Not called again
    assert mock_aioconsumer.commit.call_count == 2  # Offset committed for skip


@pytest.mark.asyncio
async def test_consumer_deserialization_failure_dlq_routing(mock_config: LLMServiceConfig):
    """Verify corrupted messages are routed to DLQ and offset committed to prevent poison pill."""
    mock_aioconsumer = AsyncMock()
    mock_publisher = AsyncMock()
    handler_mock = AsyncMock()

    consumer_engine = KafkaConsumerEngine(
        config=mock_config,
        publisher=mock_publisher,
        event_handler=handler_mock,
        consumer=mock_aioconsumer,
    )
    consumer_engine._running = True

    corrupt_record = ConsumerRecord(
        topic="chat.message.created",
        partition=0,
        offset=42,
        timestamp=0,
        timestamp_type=0,
        key=b"corrupt_key",
        value=b"NOT_A_VALID_JSON_STRING{{{",
        checksum=None,
        serialized_key_size=11,
        serialized_value_size=25,
        headers=[],
    )
    tp = TopicPartition("chat.message.created", 0)

    await consumer_engine._process_single_message(corrupt_record, tp)

    # Handler should not be invoked
    assert handler_mock.call_count == 0
    # DLQ event must be published
    mock_publisher.publish_dlq.assert_called_once()
    # Offset must be committed
    mock_aioconsumer.commit.assert_called_once()


@pytest.mark.asyncio
async def test_chat_consumer_event_handler():
    """Verify ChatConsumer delegates to pipeline."""
    mock_pipeline = AsyncMock()
    consumer = ChatConsumer(pipeline=mock_pipeline)

    event = ChatMessageCreatedEvent(
        message_id="m1",
        conversation_id="c1",
        user_id="u1",
        content="hello",
    )
    await consumer.handle(event)
    mock_pipeline.run.assert_called_once_with(event)
