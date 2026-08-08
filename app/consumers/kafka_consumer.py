"""
Kafka Consumer Engine with Manual Offset Commit, Idempotency, and DLQ.
Implements LLD v2.0 Section 6 and HLD v2.0 Section 26.2.
"""

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import structlog
from aiokafka import AIOKafkaConsumer, ConsumerRecord, TopicPartition
from aiokafka.structs import OffsetAndMetadata

from app.config.logging import (
    correlation_conversation_id,
    correlation_request_id,
    correlation_trace_id,
    get_logger,
)
from app.config.settings import LLMServiceConfig, get_settings
from app.models.request import ChatMessageCreatedEvent, TraceContext
from app.models.response import ChatMessageDLQEvent
from app.producers.kafka_producer import KafkaPublisher
from app.utils.cache import TTLCache
from app.utils.helpers import generate_request_id
from app.utils.retry import is_retriable


class KafkaConsumerEngine:
    """
    Asynchronous Kafka Consumer with at-least-once manual offset commit,
    TTL idempotency check, concurrency control, and Dead Letter Queue fallback.
    """

    def __init__(
        self,
        config: LLMServiceConfig | None = None,
        publisher: KafkaPublisher | None = None,
        event_handler: Callable[[ChatMessageCreatedEvent], Any] | None = None,
        consumer: AIOKafkaConsumer | None = None,
        max_concurrent_tasks: int = 50,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.config: LLMServiceConfig = config or get_settings()
        self.publisher: KafkaPublisher = publisher or KafkaPublisher(config=self.config)
        self.event_handler = event_handler
        self._consumer: AIOKafkaConsumer | None = consumer
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._processed_ids = TTLCache(max_size=10000, default_ttl_seconds=300)
        self.logger = logger or get_logger("kafka_consumer")
        self._running: bool = False
        self._consume_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start consumer and launch continuous consumption loop in background."""
        if self._running:
            return

        if self._consumer is None:
            consumer_kwargs: dict[str, Any] = {
                "bootstrap_servers": self.config.kafka_bootstrap_servers,
                "group_id": self.config.kafka_consumer_group,
                "auto_offset_reset": "earliest",
                "enable_auto_commit": False,
                "max_poll_interval_ms": self.config.kafka_max_poll_interval_ms,
                "session_timeout_ms": 45000,
                "heartbeat_interval_ms": 3000,
                "max_poll_records": 10,
            }
            if self.config.kafka_username and self.config.kafka_password:
                consumer_kwargs["security_protocol"] = "SASL_SSL"
                consumer_kwargs["sasl_mechanism"] = "SCRAM-SHA-512"
                consumer_kwargs["sasl_plain_username"] = self.config.kafka_username
                consumer_kwargs["sasl_plain_password"] = (
                    self.config.kafka_password.get_secret_value()
                )

            self._consumer = AIOKafkaConsumer(self.config.kafka_input_topic, **consumer_kwargs)

        await self._consumer.start()
        await self.publisher.start()
        self._running = True
        self._consume_task = asyncio.create_task(self._consume_loop())
        self.logger.info(
            "KafkaConsumer started successfully",
            topic=self.config.kafka_input_topic,
            group_id=self.config.kafka_consumer_group,
        )

    async def stop(self) -> None:
        """Gracefully stop consumer loop and underlying client."""
        self._running = False
        if self._consume_task and not self._consume_task.done():
            self._consume_task.cancel()
            try:
                await self._consume_task
            except asyncio.CancelledError:
                pass

        if self._consumer:
            await self._consumer.stop()

        self.logger.info("KafkaConsumer stopped")

    async def _consume_loop(self) -> None:
        """Continuous polling and dispatch loop."""
        while self._running and self._consumer:
            try:
                # Poll batch of records with 100ms timeout
                msg_batch = await self._consumer.getmany(timeout_ms=100, max_records=10)
                for tp, messages in msg_batch.items():
                    for msg in messages:
                        # Spawn worker task guarded by semaphore
                        asyncio.create_task(self._process_message_guarded(msg, tp))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error("Error in Kafka consume loop", error=str(exc), exc_info=True)
                await asyncio.sleep(0.5)

    async def _process_message_guarded(self, msg: ConsumerRecord, tp: TopicPartition) -> None:
        """Process record with concurrency throttling and commit on completion."""
        async with self._semaphore:
            await self._process_single_message(msg, tp)

    async def _process_single_message(self, msg: ConsumerRecord, tp: TopicPartition) -> None:
        """Parse, check idempotency, invoke pipeline with retries, and commit offset."""
        raw_val = msg.value.decode("utf-8") if isinstance(msg.value, bytes) else str(msg.value)
        key_str = msg.key.decode("utf-8") if isinstance(msg.key, bytes) else str(msg.key or "")

        # 1. Deserialize message
        try:
            payload_dict = json.loads(raw_val)
            event = ChatMessageCreatedEvent.model_validate(payload_dict)
        except Exception as exc:
            self.logger.warning(
                "Deserialization failed for incoming message -> routing to DLQ",
                error=str(exc),
                offset=msg.offset,
                partition=msg.partition,
            )
            dlq_event = ChatMessageDLQEvent(
                original_topic=msg.topic,
                original_partition=msg.partition,
                original_offset=msg.offset,
                original_key=key_str,
                original_value=raw_val,
                error_type="DeserializationError",
                error_message=f"Failed to parse ChatMessageCreatedEvent: {exc}",
                failed_at=datetime.now(UTC),
            )
            await self.publisher.publish_dlq(dlq_event)
            await self._commit_offset(tp, msg.offset + 1)
            return

        # 2. Extract TraceContext & Correlation IDs
        traceparent = ""
        tracestate = ""
        if msg.headers:
            for h_key, h_val in msg.headers:
                if h_key == "traceparent":
                    traceparent = h_val.decode("utf-8") if isinstance(h_val, bytes) else str(h_val)
                elif h_key == "tracestate":
                    tracestate = h_val.decode("utf-8") if isinstance(h_val, bytes) else str(h_val)

        if traceparent and not event.trace_context.traceparent:
            event.trace_context = TraceContext(traceparent=traceparent, tracestate=tracestate)

        correlation_trace_id.set(event.trace_context.traceparent)
        correlation_conversation_id.set(event.conversation_id)
        req_id = generate_request_id()
        correlation_request_id.set(req_id)

        # 3. Idempotency Check
        if self._processed_ids.contains(event.message_id):
            self.logger.debug(
                "Duplicate message skipped via idempotency cache",
                message_id=event.message_id,
                conversation_id=event.conversation_id,
            )
            await self._commit_offset(tp, msg.offset + 1)
            return

        self._processed_ids.set(event.message_id, True)

        # 4. Process event with retries
        success = await self._execute_with_consumer_retries(event)

        if success:
            await self._commit_offset(tp, msg.offset + 1)
        else:
            # Permanent failure after all retries exhausted -> send to DLQ
            dlq_event = ChatMessageDLQEvent(
                original_topic=msg.topic,
                original_partition=msg.partition,
                original_offset=msg.offset,
                original_key=key_str,
                original_value=raw_val,
                error_type="PipelineError",
                error_message=f"Processing failed after 3 consumer retries for message {event.message_id}",
                failed_at=datetime.now(UTC),
                retry_count=3,
            )
            await self.publisher.publish_dlq(dlq_event)
            await self._commit_offset(tp, msg.offset + 1)

    async def _execute_with_consumer_retries(self, event: ChatMessageCreatedEvent) -> bool:
        """Execute handler with exponential backoff (100ms, 200ms, 400ms) for retriable errors."""
        if self.event_handler is None:
            self.logger.warning("No event_handler configured on KafkaConsumerEngine")
            return True

        max_attempts = 3
        delays = [0.1, 0.2, 0.4]

        for attempt in range(1, max_attempts + 1):
            try:
                res = self.event_handler(event)
                if asyncio.iscoroutine(res):
                    await res
                return True
            except Exception as exc:
                self.logger.warning(
                    "Consumer handler failure",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    message_id=event.message_id,
                    error=str(exc),
                    retriable=is_retriable(exc),
                )
                if attempt < max_attempts and is_retriable(exc):
                    await asyncio.sleep(delays[attempt - 1])
                else:
                    return False

        return False

    async def _commit_offset(self, tp: TopicPartition, next_offset: int) -> None:
        """Manual offset commit to Kafka broker."""
        if self._consumer and self._running:
            try:
                await self._consumer.commit({tp: OffsetAndMetadata(next_offset, "")})
            except Exception as exc:
                self.logger.error(
                    "Failed to commit Kafka offset",
                    offset=next_offset,
                    partition=tp.partition,
                    error=str(exc),
                )
