"""
Kafka Producer and Publisher Implementation.
Implements LLD v2.0 Section 7 and HLD v2.0 Section 26.3.
"""

import json
from datetime import UTC, datetime
from typing import Any

import structlog
from aiokafka import AIOKafkaProducer
from pydantic import BaseModel

from app.config.logging import get_logger
from app.config.settings import LLMServiceConfig, get_settings
from app.models.pipeline_context import PipelineContext
from app.models.request import TraceContext
from app.models.response import (
    ChatMessageDLQEvent,
    ChatResponseCancelledEvent,
    ChatResponseChunkEvent,
    ChatResponseGeneratedEvent,
    MemoryUpdateRequestedEvent,
    UsageMetrics,
)
from app.utils.retry import RetryManager, RetryPolicy


class KafkaPublisher:
    """
    Centralized Kafka Event Publisher.
    Handles message serialization, partitioning by conversation_id,
    W3C trace header propagation, and retries.
    """

    def __init__(
        self,
        config: LLMServiceConfig | None = None,
        producer: AIOKafkaProducer | None = None,
        retry_manager: RetryManager | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.config: LLMServiceConfig = config or get_settings()
        self._producer: AIOKafkaProducer | None = producer
        self._retry_manager: RetryManager = retry_manager or RetryManager(
            RetryPolicy(max_attempts=3)
        )
        self.logger = logger or get_logger("kafka_publisher")
        self._started: bool = False

    async def start(self) -> None:
        """Initialize and start the underlying AIOKafkaProducer."""
        if self._started:
            return

        if self._producer is None:
            producer_kwargs: dict[str, Any] = {
                "bootstrap_servers": self.config.kafka_bootstrap_servers,
                "acks": "all",
                "compression_type": "lz4",
                "max_request_size": 2097152,  # 2MB
                "linger_ms": 5,
                "enable_idempotence": True,
            }
            if self.config.kafka_username and self.config.kafka_password:
                producer_kwargs["security_protocol"] = "SASL_SSL"
                producer_kwargs["sasl_mechanism"] = "SCRAM-SHA-512"
                producer_kwargs["sasl_plain_username"] = self.config.kafka_username
                producer_kwargs["sasl_plain_password"] = (
                    self.config.kafka_password.get_secret_value()
                )

            self._producer = AIOKafkaProducer(**producer_kwargs)

        await self._producer.start()
        self._started = True
        self.logger.info(
            "KafkaPublisher started successfully",
            bootstrap_servers=self.config.kafka_bootstrap_servers,
        )

    async def stop(self) -> None:
        """Flush and stop the producer."""
        if self._producer and self._started:
            await self._producer.stop()
            self._started = False
            self.logger.info("KafkaPublisher stopped successfully")

    async def _send(
        self,
        topic: str,
        key: str,
        payload: BaseModel | dict[str, Any] | str | bytes,
        trace_context: TraceContext | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Internal method to send serialized bytes to Kafka with headers and retry."""
        if not self._started or self._producer is None:
            raise RuntimeError("KafkaPublisher is not started. Call start() before publishing.")

        # Serialize value to bytes
        if isinstance(payload, BaseModel):
            value_bytes = payload.model_dump_json().encode("utf-8")
        elif isinstance(payload, dict):
            value_bytes = json.dumps(payload, default=str).encode("utf-8")
        elif isinstance(payload, str):
            value_bytes = payload.encode("utf-8")
        elif isinstance(payload, bytes):
            value_bytes = payload
        else:
            value_bytes = str(payload).encode("utf-8")

        key_bytes = key.encode("utf-8")

        # Build W3C Trace headers
        headers: list[tuple[str, bytes]] = []
        if trace_context:
            if trace_context.traceparent:
                headers.append(("traceparent", trace_context.traceparent.encode("utf-8")))
            if trace_context.tracestate:
                headers.append(("tracestate", trace_context.tracestate.encode("utf-8")))
        if correlation_id:
            headers.append(("correlation_id", correlation_id.encode("utf-8")))

        async def _publish():
            await self._producer.send_and_wait(
                topic=topic,
                key=key_bytes,
                value=value_bytes,
                headers=headers,
            )

        await self._retry_manager.execute_with_retry(
            _publish,
            operation_name=f"kafka_publish_{topic}",
        )

    async def publish_chunk(self, event: ChatResponseChunkEvent) -> None:
        """Publish a single streaming token chunk to chat.response.chunk."""
        await self._send(
            topic=self.config.kafka_chunk_topic,
            key=event.conversation_id,
            payload=event,
            correlation_id=event.message_id,
        )

    async def publish_response(self, ctx: PipelineContext, full_response: str) -> None:
        """
        Publish final chat.response.generated event with complete telemetry.
        Called once generation and streaming complete.
        """
        plan = ctx.plan
        mode_val = (
            plan.mode.value
            if plan and hasattr(plan.mode, "value")
            else (str(plan.mode) if plan else "default")
        )
        skill_val = (
            plan.skill.value
            if plan and hasattr(plan.skill, "value")
            else (str(plan.skill) if plan else "general_chat")
        )

        final_event = ChatResponseGeneratedEvent(
            response_id=f"resp_{ctx.request_id[:12]}",
            conversation_id=ctx.conversation_id,
            user_id=ctx.user_id,
            request_message_id=ctx.message_id,
            full_content=full_response,
            provider=ctx.selected_provider or "nvidia",
            generation_fallback_used=ctx.generation_fallback_used,
            mode=mode_val,
            skill=skill_val,
            engine_type=ctx.engine_type or "mode_handler",
            usage=ctx.usage if isinstance(ctx.usage, UsageMetrics) else UsageMetrics(),
            cost_usd=ctx.cost_usd or 0.0,
            latency_ms=(datetime.now(UTC) - ctx.started_at).total_seconds() * 1000.0,
            ttft_ms=(
                (ctx.first_chunk_at - ctx.inference_started_at).total_seconds() * 1000.0
                if ctx.first_chunk_at and ctx.inference_started_at
                else 0.0
            ),
            finish_reason="stop",
            tools_used=[t.tool_name for t in (plan.tools if plan else [])],
            context_sources=list(
                filter(
                    None,
                    [
                        "memory" if ctx.context_bundle and ctx.context_bundle.memory else None,
                        "graph" if ctx.context_bundle and ctx.context_bundle.graph else None,
                        "retrieval"
                        if ctx.context_bundle and ctx.context_bundle.retrieval
                        else None,
                    ],
                )
            ),
            trace_context=TraceContext(traceparent=ctx.trace_id),
            status="success",
            timestamp=datetime.now(UTC),
        )

        await self._send(
            topic=self.config.kafka_output_topic,
            key=ctx.conversation_id,
            payload=final_event,
            trace_context=TraceContext(traceparent=ctx.trace_id),
            correlation_id=ctx.request_id,
        )
        self.logger.info(
            "Published final response event",
            conversation_id=ctx.conversation_id,
            message_id=ctx.message_id,
            engine_type=ctx.engine_type,
            provider=ctx.selected_provider,
        )

    async def publish_memory_update(self, ctx: PipelineContext, full_response: str) -> None:
        """Publish memory.update.requested event to trigger async memory synthesis."""
        mode_val = (
            ctx.plan.mode.value if ctx.plan and hasattr(ctx.plan.mode, "value") else "default"
        )
        memory_event = MemoryUpdateRequestedEvent(
            conversation_id=ctx.conversation_id,
            user_id=ctx.user_id,
            user_message=ctx.user_message,
            assistant_response=full_response,
            mode=mode_val,
            timestamp=datetime.now(UTC),
        )
        await self._send(
            topic="memory.update.requested",
            key=ctx.conversation_id,
            payload=memory_event,
            correlation_id=ctx.request_id,
        )

    async def publish_cancellation(
        self, ctx: PipelineContext, reason: str = "client_disconnected"
    ) -> None:
        """Publish chat.response.cancelled event on client abort."""
        cancel_event = ChatResponseCancelledEvent(
            conversation_id=ctx.conversation_id,
            user_id=ctx.user_id,
            message_id=ctx.message_id,
            reason=reason,
            timestamp=datetime.now(UTC),
        )
        await self._send(
            topic="chat.response.cancelled",
            key=ctx.conversation_id,
            payload=cancel_event,
            correlation_id=ctx.request_id,
        )

    async def publish_dlq(self, event: ChatMessageDLQEvent) -> None:
        """Publish unprocessable or permanently failed message to Dead Letter Queue."""
        key = event.original_key or "dlq"
        await self._send(
            topic=self.config.kafka_dlq_topic,
            key=key,
            payload=event,
        )
        self.logger.warning(
            "Published event to Dead Letter Queue",
            error_type=event.error_type,
            error_message=event.error_message,
            original_topic=event.original_topic,
        )
