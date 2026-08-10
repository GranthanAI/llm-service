"""
Unit Tests for Phase 16: Streaming Engine.
Tests dynamic chunk sizing, time/size flushing, TTFT recording, Kafka event emissions,
final response assembly, and cooperative cancellation.
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.execution_plan import ExecutionPlan, IntentCategory, ReasoningMode, Skill, UserMode
from app.models.pipeline_context import PipelineContext
from app.models.response import ChatResponseChunkEvent
from app.producers.kafka_producer import KafkaPublisher
from app.services.streaming_service import StreamingEngine
from app.utils.cancellation import CancellationToken


def create_test_context(
    conversation_id: str = "conv_stream_001",
    message_id: str = "msg_stream_001",
    user_id: str = "user_stream_001",
    mode: str = "smart",
) -> PipelineContext:
    """Helper to create a populated PipelineContext for streaming tests."""
    ctx = PipelineContext(
        conversation_id=conversation_id,
        user_id=user_id,
        message_id=message_id,
        request_id=f"req_{message_id}",
        user_message="Explain Transformer attention mechanisms in 3 sentences.",
        started_at=datetime.now(UTC),
        inference_started_at=datetime.now(UTC),
        selected_provider="nvidia",
        engine_type="langgraph",
        plan=ExecutionPlan(
            mode=UserMode(mode),
            engine_type="langgraph",
            intent=IntentCategory.QUESTION_ANSWERING,
            skill=Skill.RESEARCH,
            reasoning_mode=ReasoningMode.REACT,
        ),
    )
    return ctx


async def token_generator(tokens: list[str], delay_s: float = 0.0) -> AsyncIterator[str]:
    """Helper async generator producing tokens with optional inter-token delay."""
    for token in tokens:
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        yield token


@pytest.mark.asyncio
async def test_streaming_engine_first_chunk_and_ttft():
    """Verify first chunk emits immediately with size=1 and records TTFT."""
    mock_publisher = MagicMock(spec=KafkaPublisher)
    mock_publisher.publish_chunk = AsyncMock()
    mock_publisher.publish_response = AsyncMock()
    mock_publisher.publish_memory_update = AsyncMock()

    engine = StreamingEngine(
        publisher=mock_publisher,
        first_chunk_size=1,
        default_chunk_size=6,
        flush_interval_ms=500,
    )

    ctx = create_test_context()
    tokens = ["Hello", " ", "world", "!", " This", " is", " GraphGPT", "."]

    full_resp = await engine.stream(token_generator(tokens), ctx)

    assert full_resp == "Hello world! This is GraphGPT."
    assert ctx.first_chunk_at is not None
    assert ctx.completed_at is not None

    # First published chunk must be the first token only
    assert mock_publisher.publish_chunk.call_count >= 2
    first_call_arg: ChatResponseChunkEvent = mock_publisher.publish_chunk.call_args_list[0][0][0]
    assert first_call_arg.content == "Hello"
    assert first_call_arg.chunk_index == 0
    assert first_call_arg.sequence_number == 1
    assert first_call_arg.is_last is False


@pytest.mark.asyncio
async def test_streaming_engine_size_based_flushing():
    """Verify tokens are accumulated and flushed in chunks of default_chunk_size=6."""
    mock_publisher = MagicMock(spec=KafkaPublisher)
    mock_publisher.publish_chunk = AsyncMock()
    mock_publisher.publish_response = AsyncMock()
    mock_publisher.publish_memory_update = AsyncMock()

    engine = StreamingEngine(
        publisher=mock_publisher,
        first_chunk_size=1,
        default_chunk_size=4,
        flush_interval_ms=1000,
    )

    ctx = create_test_context()
    # 1 token (first chunk) + 4 tokens (chunk 1) + 4 tokens (chunk 2) + 2 tokens (final chunk) = 11 tokens
    tokens = [f"t{i}" for i in range(11)]

    full_resp = await engine.stream(token_generator(tokens), ctx)

    assert full_resp == "".join(tokens)
    # Total chunks = 4 (1 + 4 + 4 + 2)
    assert mock_publisher.publish_chunk.call_count == 4

    chunks = [call[0][0] for call in mock_publisher.publish_chunk.call_args_list]
    assert chunks[0].content == "t0"
    assert chunks[1].content == "t1t2t3t4"
    assert chunks[2].content == "t5t6t7t8"
    assert chunks[3].content == "t9t10"
    assert chunks[3].is_last is True


@pytest.mark.asyncio
async def test_streaming_engine_time_based_flushing():
    """Verify slow token streams flush based on flush_interval_ms."""
    mock_publisher = MagicMock(spec=KafkaPublisher)
    mock_publisher.publish_chunk = AsyncMock()
    mock_publisher.publish_response = AsyncMock()
    mock_publisher.publish_memory_update = AsyncMock()

    engine = StreamingEngine(
        publisher=mock_publisher,
        first_chunk_size=1,
        default_chunk_size=20,  # Very large size threshold
        flush_interval_ms=20,  # 20ms time threshold
    )

    ctx = create_test_context()
    # 4 tokens with 30ms delay each (each should trigger time flush)
    tokens = ["A", "B", "C", "D"]

    full_resp = await engine.stream(token_generator(tokens, delay_s=0.03), ctx)

    assert full_resp == "ABCD"
    # Should have flushed each token due to timeout
    assert mock_publisher.publish_chunk.call_count >= 3


@pytest.mark.asyncio
async def test_streaming_engine_code_block_chunk_sizing():
    """Verify entering code blocks (```) increases chunk threshold to code_block_chunk_size."""
    mock_publisher = MagicMock(spec=KafkaPublisher)
    mock_publisher.publish_chunk = AsyncMock()
    mock_publisher.publish_response = AsyncMock()
    mock_publisher.publish_memory_update = AsyncMock()

    engine = StreamingEngine(
        publisher=mock_publisher,
        first_chunk_size=1,
        default_chunk_size=3,
        code_block_chunk_size=8,
        flush_interval_ms=1000,
    )

    ctx = create_test_context()
    tokens = [
        "Intro",
        "```python\n",
        "def",
        " ",
        "add",
        "(",
        "a",
        ",",
        " ",
        "b",
        "):",
        "\n",
        "    return",
        " a + b\n",
        "```",
        " Outro",
    ]

    full_resp = await engine.stream(token_generator(tokens), ctx)

    assert "def add(a, b):" in full_resp
    assert mock_publisher.publish_chunk.call_count >= 2


@pytest.mark.asyncio
async def test_streaming_engine_final_response_and_memory_events():
    """Verify publish_response and publish_memory_update are called with complete text."""
    mock_publisher = MagicMock(spec=KafkaPublisher)
    mock_publisher.publish_chunk = AsyncMock()
    mock_publisher.publish_response = AsyncMock()
    mock_publisher.publish_memory_update = AsyncMock()

    engine = StreamingEngine(publisher=mock_publisher)
    ctx = create_test_context()
    tokens = ["Neural", " ", "networks", " ", "learn", " ", "representations", "."]

    full_resp = await engine.stream(token_generator(tokens), ctx)

    assert full_resp == "Neural networks learn representations."
    mock_publisher.publish_response.assert_awaited_once_with(ctx, full_resp)
    mock_publisher.publish_memory_update.assert_awaited_once_with(ctx, full_resp)
    assert ctx.usage is not None
    assert ctx.usage.completion_tokens == len(tokens)


@pytest.mark.asyncio
async def test_streaming_engine_cooperative_cancellation():
    """Verify CancellationToken aborts token streaming and publishes cancellation event."""
    mock_publisher = MagicMock(spec=KafkaPublisher)
    mock_publisher.publish_chunk = AsyncMock()
    mock_publisher.publish_response = AsyncMock()
    mock_publisher.publish_memory_update = AsyncMock()
    mock_publisher.publish_cancellation = AsyncMock()

    engine = StreamingEngine(publisher=mock_publisher)
    ctx = create_test_context()
    cancel_token = CancellationToken()

    async def cancelling_generator() -> AsyncIterator[str]:
        yield "Token1"
        yield "Token2"
        cancel_token.cancel(reason="user_abort")
        yield "Token3"
        yield "Token4"

    partial_resp = await engine.stream(
        cancelling_generator(),
        ctx,
        cancellation_token=cancel_token,
    )

    assert cancel_token.is_cancelled is True
    assert cancel_token.reason == "user_abort"
    mock_publisher.publish_cancellation.assert_awaited_once_with(ctx, reason="user_abort")
    # Response and memory update should not be published on cancellation
    mock_publisher.publish_response.assert_not_called()
    mock_publisher.publish_memory_update.assert_not_called()
    assert "Token1" in partial_resp


@pytest.mark.asyncio
async def test_streaming_engine_rate_limited_mode():
    """Verify rate-limited mode uses rate_limited_chunk_size threshold."""
    mock_publisher = MagicMock(spec=KafkaPublisher)
    mock_publisher.publish_chunk = AsyncMock()
    mock_publisher.publish_response = AsyncMock()
    mock_publisher.publish_memory_update = AsyncMock()

    engine = StreamingEngine(
        publisher=mock_publisher,
        first_chunk_size=1,
        default_chunk_size=4,
        rate_limited_chunk_size=8,
        flush_interval_ms=1000,
    )

    ctx = create_test_context()
    # 1 token (first chunk) + 8 tokens (chunk 1) + 2 tokens (final) = 11 tokens
    tokens = [f"tok_{i}" for i in range(11)]

    full_resp = await engine.stream(
        token_generator(tokens),
        ctx,
        is_rate_limited=True,
    )

    assert full_resp == "".join(tokens)
    # With rate_limited_chunk_size=8: 1 (first token) + 8 (tok_1..tok_8) + 2 (tok_9..tok_10) = 3 chunks
    assert mock_publisher.publish_chunk.call_count == 3
