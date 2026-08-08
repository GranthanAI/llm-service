"""
Chat Message Event Handler.
Implements LLD v2.0 Section 6.1.
"""

from typing import Any

import structlog

from app.config.logging import get_logger
from app.models.request import ChatMessageCreatedEvent
from app.utils.tracing import trace_span


class ChatConsumer:
    """
    Consumer handler for chat.message.created events.
    Receives validated events from KafkaConsumerEngine and triggers the RequestPipeline.
    """

    def __init__(
        self, pipeline: Any | None = None, logger: structlog.stdlib.BoundLogger | None = None
    ):
        self.pipeline = pipeline
        self.logger = logger or get_logger("chat_consumer")

    @trace_span("chat_consumer_handle")
    async def handle(self, event: ChatMessageCreatedEvent) -> None:
        """Process validated ChatMessageCreatedEvent."""
        self.logger.info(
            "Received chat message event",
            conversation_id=event.conversation_id,
            message_id=event.message_id,
            user_id=event.user_id,
            mode_hint=event.mode_hint,
        )

        if self.pipeline is not None:
            await self.pipeline.run(event)
        else:
            self.logger.debug(
                "No pipeline wired to ChatConsumer — event acknowledged",
                message_id=event.message_id,
            )
