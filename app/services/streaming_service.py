"""
Streaming Engine Service Implementation.
Implements LLD v2.0 Section 21 and HLD v2.0 Section 19.
Consumes normalized token streams, manages dynamic chunk buffering, publishes
Kafka streaming chunk events, records TTFT metrics, and handles final assembly and cancellation.
"""

import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import structlog

from app.config.logging import get_logger
from app.models.pipeline_context import PipelineContext
from app.models.response import (
    ChatResponseChunkEvent,
    ErrorType,  # noqa: F401
    UsageMetrics,
)
from app.producers.kafka_producer import KafkaPublisher
from app.utils.cancellation import CancellationToken
from app.utils.metrics import STREAMING_CHUNKS_TOTAL, TTFT
from app.utils.tracing import get_tracer

logger = get_logger("streaming_engine")


class StreamingEngine:
    """
    Consumes token streams from the Generation Router, applies dynamic chunk sizing
    and time-based buffer flushing, publishes real-time Kafka chunk events,
    records Time-To-First-Token (TTFT), and publishes final response and memory events.
    """

    def __init__(
        self,
        publisher: KafkaPublisher,
        default_chunk_size: int = 6,
        first_chunk_size: int = 1,
        code_block_chunk_size: int = 20,
        rate_limited_chunk_size: int = 12,
        flush_interval_ms: int = 50,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        self.publisher = publisher
        self.default_chunk_size = default_chunk_size
        self.first_chunk_size = first_chunk_size
        self.code_block_chunk_size = code_block_chunk_size
        self.rate_limited_chunk_size = rate_limited_chunk_size
        self.flush_interval_ms = flush_interval_ms
        self.logger = logger_instance or logger
        self.tracer = get_tracer()

    async def stream(
        self,
        token_iter: AsyncIterator[str],
        ctx: PipelineContext,
        cancellation_token: CancellationToken | None = None,
        is_rate_limited: bool = False,
    ) -> str:
        """
        Consumes tokens from token_iter, manages dynamic buffering, publishes
        ChatResponseChunkEvent messages to Kafka, records TTFT, and publishes
        the final response event.
        """
        with self.tracer.start_as_current_span("streaming_engine.stream") as span:
            span.set_attribute("conversation_id", ctx.conversation_id)
            span.set_attribute("message_id", ctx.message_id)
            span.set_attribute("provider", ctx.selected_provider or "nvidia")

            if ctx.inference_started_at is None:
                ctx.inference_started_at = datetime.now(UTC)

            buffer: list[str] = []
            buffer_tokens: int = 0
            chunk_index: int = 0
            full_content: list[str] = []
            last_flush_time: float = time.perf_counter()
            first_chunk_emitted: bool = False
            in_code_block: bool = False
            accumulated_text: str = ""

            provider = ctx.selected_provider or "nvidia"
            mode = (
                ctx.plan.mode.value
                if ctx.plan and hasattr(ctx.plan.mode, "value")
                else (str(ctx.plan.mode) if ctx.plan else "default")
            )
            engine_type = ctx.engine_type or "mode_handler"

            async for token in token_iter:
                # 1. Cooperative Cancellation Check
                if cancellation_token and cancellation_token.is_cancelled:
                    self.logger.warning(
                        "Streaming cancelled by cancellation token",
                        conversation_id=ctx.conversation_id,
                        message_id=ctx.message_id,
                        reason=cancellation_token.reason,
                    )
                    aclose = getattr(token_iter, "aclose", None)
                    if aclose is not None:               # stop the provider call; no further tokens are consumed
                        try:
                            await aclose()
                        except Exception:
                            pass
                    self._record_cancelled_usage(ctx, "".join(full_content))
                    await self.publisher.publish_cancellation(
                        ctx, reason=cancellation_token.reason or "client_disconnected"
                    )
                    return "".join(full_content)

                buffer.append(token)
                full_content.append(token)
                buffer_tokens += 1
                accumulated_text += token

                # Detect code blocks for larger chunk rendering
                if "```" in token:
                    in_code_block = not in_code_block

                # 2. Dynamic Chunk Size Strategy
                if not first_chunk_emitted:
                    target_chunk_size = self.first_chunk_size
                elif in_code_block:
                    target_chunk_size = self.code_block_chunk_size
                elif is_rate_limited:
                    target_chunk_size = self.rate_limited_chunk_size
                else:
                    target_chunk_size = self.default_chunk_size

                # 3. Size or Time-based Flush Condition
                now_perf = time.perf_counter()
                elapsed_flush = now_perf - last_flush_time
                should_flush = (buffer_tokens >= target_chunk_size) or (
                    elapsed_flush >= (self.flush_interval_ms / 1000.0) and buffer_tokens > 0
                )

                if should_flush:
                    await self._flush_buffer(
                        buffer=buffer,
                        ctx=ctx,
                        chunk_index=chunk_index,
                        provider=provider,
                        mode=mode,
                        is_last=False,
                    )

                    if not first_chunk_emitted:
                        first_chunk_emitted = True
                        ctx.first_chunk_at = datetime.now(UTC)
                        ttft_seconds = (
                            ctx.first_chunk_at - ctx.inference_started_at
                        ).total_seconds()
                        try:
                            TTFT.labels(
                                provider=provider,
                                mode=mode,
                                engine_type=engine_type,
                            ).observe(ttft_seconds)
                        except Exception:
                            pass
                        self.logger.debug(
                            "First token streamed (TTFT recorded)",
                            ttft_ms=ttft_seconds * 1000.0,
                            provider=provider,
                            mode=mode,
                        )

                    chunk_index += 1
                    buffer = []
                    buffer_tokens = 0
                    last_flush_time = time.perf_counter()

            # 4. Final Remaining Buffer Flush
            if buffer:
                if not first_chunk_emitted:
                    first_chunk_emitted = True
                    ctx.first_chunk_at = datetime.now(UTC)
                    ttft_seconds = (ctx.first_chunk_at - ctx.inference_started_at).total_seconds()
                    try:
                        TTFT.labels(
                            provider=provider,
                            mode=mode,
                            engine_type=engine_type,
                        ).observe(ttft_seconds)
                    except Exception:
                        pass

                await self._flush_buffer(
                    buffer=buffer,
                    ctx=ctx,
                    chunk_index=chunk_index,
                    provider=provider,
                    mode=mode,
                    is_last=True,
                )
                chunk_index += 1

            # 5. Final Response Assembly & Telemetry Publication
            full_response = "".join(full_content)
            ctx.completed_at = datetime.now(UTC)

            # Token usage is what the PROVIDERS reported for the answer call(s) (app/usage). Characters are not
            # tokens: when no call reported anything the figures stay 0 rather than being invented here.
            collector = getattr(ctx, "usage_collector", None)
            if collector is not None:
                prompt_t, completion_t = collector.totals("answer")
                ctx.usage = UsageMetrics(
                    prompt_tokens=prompt_t, completion_tokens=completion_t, total_tokens=prompt_t + completion_t)
            elif ctx.usage is None:
                ctx.usage = UsageMetrics()

            # Publish final chat.response.generated event
            await self.publisher.publish_response(ctx, full_response)

            # Publish memory.update.requested event
            await self.publisher.publish_memory_update(ctx, full_response)

            self.logger.info(
                "Streaming response completed and published",
                conversation_id=ctx.conversation_id,
                message_id=ctx.message_id,
                chunks_emitted=chunk_index,
                total_length=len(full_response),
            )

            return full_response

    @staticmethod
    def _record_cancelled_usage(ctx: PipelineContext, partial: str) -> None:
        """The answer call of a cancelled stream never reached its final usage chunk: record the partial output as an
        estimate (flagged), so billing sees that tokens were produced and the message counts as used."""
        collector = getattr(ctx, "usage_collector", None)
        if collector is None or not partial:
            return
        name = str(ctx.selected_provider or "")
        provider, _, model = name.partition(":")
        collector.record_provider_usage(
            provider=provider or "unknown", model=getattr(ctx, "answered_model", None) or model or provider or "unknown",
            input_tokens=None, output_tokens=None, messages=getattr(ctx, "final_prompt_messages", None),
            output_text=partial, purpose="answer")

    async def _flush_buffer(
        self,
        buffer: list[str],
        ctx: PipelineContext,
        chunk_index: int,
        provider: str,
        mode: str,
        is_last: bool = False,
    ) -> None:
        """Constructs and publishes a single ChatResponseChunkEvent."""
        content = "".join(buffer)
        if not content:
            return

        chunk_event = self._build_chunk_event(
            content=content,
            ctx=ctx,
            chunk_index=chunk_index,
            sequence_number=chunk_index + 1,
            is_last=is_last,
        )

        await self.publisher.publish_chunk(chunk_event)

        try:
            STREAMING_CHUNKS_TOTAL.labels(provider=provider, mode=mode).inc()
        except Exception:
            pass

    def _build_chunk_event(
        self,
        content: str,
        ctx: PipelineContext,
        chunk_index: int,
        sequence_number: int,
        is_last: bool = False,
    ) -> ChatResponseChunkEvent:
        """Constructs a ChatResponseChunkEvent matching LLD Section 21."""
        return ChatResponseChunkEvent(
            conversation_id=ctx.conversation_id,
            user_id=ctx.user_id,
            message_id=ctx.message_id,
            response_message_id=ctx.response_message_id,
            chunk_index=chunk_index,
            sequence_number=sequence_number,
            content=content,
            is_last=is_last,
            timestamp=datetime.now(UTC),
        )
