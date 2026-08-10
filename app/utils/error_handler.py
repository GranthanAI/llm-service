"""
Error Handling and Taxonomy Classification Utility.
Implements LLD v2.0 Section 28.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from app.config.logging import get_logger
from app.exceptions.base import FatalError, PermanentError, RetriableError
from app.exceptions.grpc import GRPCError
from app.exceptions.provider import ProviderError
from app.exceptions.tool import ToolError
from app.models.response import ChatMessageDLQEvent, ChatResponseGeneratedEvent, ErrorType
from app.utils.metrics import ERRORS_TOTAL

if TYPE_CHECKING:
    from app.models.pipeline_context import PipelineContext

logger = get_logger("error_handler")


class ErrorClassification:
    """Classified error attributes."""

    def __init__(
        self,
        error_type: ErrorType,
        error_code: str,
        is_retriable: bool,
        is_fatal: bool,
        user_message: str,
    ):
        self.error_type = error_type
        self.error_code = error_code
        self.is_retriable = is_retriable
        self.is_fatal = is_fatal
        self.user_message = user_message


class ErrorHandler:
    """
    Centralized error classifier and event builder.
    Maps unhandled and domain exceptions into structured Kafka events and metrics.
    """

    def __init__(self, logger_instance: structlog.stdlib.BoundLogger | None = None):
        self.logger = logger_instance or logger

    def classify(self, exc: Exception) -> ErrorClassification:
        """Classifies any exception into category, retriability, and user-facing message."""
        exc_type = type(exc).__name__

        # 1. Check specific domain hierarchies
        if isinstance(exc, ToolError):
            return ErrorClassification(
                error_type=ErrorType.TOOL,
                error_code=getattr(exc, "error_code", "TOOL_ERROR"),
                is_retriable=isinstance(exc, RetriableError),
                is_fatal=False,
                user_message="A tool execution error occurred.",
            )

        if isinstance(exc, GRPCError):
            return ErrorClassification(
                error_type=ErrorType.GRPC,
                error_code=getattr(exc, "error_code", "GRPC_ERROR"),
                is_retriable=isinstance(exc, RetriableError),
                is_fatal=False,
                user_message="Upstream context service temporarily unavailable.",
            )

        if isinstance(exc, ProviderError):
            return ErrorClassification(
                error_type=ErrorType.INFERENCE,
                error_code=getattr(exc, "error_code", "PROVIDER_ERROR"),
                is_retriable=isinstance(exc, RetriableError),
                is_fatal=False,
                user_message="Upstream AI provider error occurred.",
            )

        # 2. Check general taxonomy markers
        if isinstance(exc, RetriableError):
            code = getattr(exc, "error_code", exc_type)
            return ErrorClassification(
                error_type=ErrorType.INFERENCE,
                error_code=code,
                is_retriable=True,
                is_fatal=False,
                user_message="Service is temporarily unavailable due to upstream delay. Retrying...",
            )

        if isinstance(exc, FatalError):
            code = getattr(exc, "error_code", exc_type)
            return ErrorClassification(
                error_type=ErrorType.UNEXPECTED,
                error_code=code,
                is_retriable=False,
                is_fatal=True,
                user_message="A critical service configuration error occurred.",
            )

        if isinstance(exc, PermanentError):
            code = getattr(exc, "error_code", exc_type)
            return ErrorClassification(
                error_type=ErrorType.ANALYSIS,
                error_code=code,
                is_retriable=False,
                is_fatal=False,
                user_message="Unable to complete request with provided input.",
            )

        # 3. Known error names
        if exc_type in ("RateLimitError", "ProviderRateLimitError", "TooManyRequests"):
            return ErrorClassification(
                error_type=ErrorType.INFERENCE,
                error_code="RATE_LIMIT_EXCEEDED",
                is_retriable=True,
                is_fatal=False,
                user_message="Rate limit reached. Please wait a moment and try again.",
            )

        if exc_type in (
            "TimeoutError",
            "ProviderTimeoutError",
            "GRPCTimeoutError",
            "ToolTimeoutError",
        ):
            return ErrorClassification(
                error_type=ErrorType.INFERENCE,
                error_code="OPERATION_TIMEOUT",
                is_retriable=True,
                is_fatal=False,
                user_message="The operation timed out. Please try again.",
            )

        if exc_type in ("ValidationError", "ToolValidationError", "PlanParseError"):
            return ErrorClassification(
                error_type=ErrorType.VALIDATION,
                error_code="VALIDATION_FAILED",
                is_retriable=False,
                is_fatal=False,
                user_message="Invalid request format.",
            )

        # Default unexpected error
        return ErrorClassification(
            error_type=ErrorType.UNEXPECTED,
            error_code="UNEXPECTED_ERROR",
            is_retriable=False,
            is_fatal=False,
            user_message="An unexpected error occurred while processing your request.",
        )

    def record_metrics(self, exc: Exception, stage: str = "pipeline") -> None:
        """Records error metric to Prometheus ERRORS_TOTAL counter."""
        classification = self.classify(exc)
        try:
            ERRORS_TOTAL.labels(
                error_type=classification.error_type.value,
                stage=stage,
            ).inc()
        except Exception:
            pass

    def build_error_response_event(
        self,
        ctx: PipelineContext,
        exc: Exception,
    ) -> ChatResponseGeneratedEvent:
        """Constructs an error-variant ChatResponseGeneratedEvent for publication to Kafka."""
        classification = self.classify(exc)
        self.record_metrics(exc, stage=ctx.engine_type or "pipeline")

        mode_str = (
            ctx.plan.mode.value
            if ctx.plan and hasattr(ctx.plan.mode, "value")
            else (str(ctx.plan.mode) if ctx.plan else "default")
        )
        skill_str = (
            ctx.plan.skill.value
            if ctx.plan and hasattr(ctx.plan.skill, "value")
            else (str(ctx.plan.skill) if ctx.plan else "general_chat")
        )

        return ChatResponseGeneratedEvent(
            response_id=f"resp_{ctx.message_id}",
            conversation_id=ctx.conversation_id,
            user_id=ctx.user_id,
            request_message_id=ctx.message_id,
            mode=mode_str,
            skill=skill_str,
            engine_type=ctx.engine_type or "mode_handler",
            provider=ctx.selected_provider,
            status="error",
            error_code=classification.error_code,
            error_message=classification.user_message,
            full_content=None,
            generation_fallback_used=ctx.generation_fallback_used,
            timestamp=datetime.now(UTC),
        )

    def build_dlq_event(
        self,
        raw_payload: dict[str, Any] | str,
        exc: Exception,
        original_topic: str = "chat.message.created",
        key: str | None = None,
    ) -> ChatMessageDLQEvent:
        """Constructs a ChatMessageDLQEvent for Dead Letter Queue routing."""
        classification = self.classify(exc)
        return ChatMessageDLQEvent(
            original_topic=original_topic,
            original_key=key,
            error_type=classification.error_type,
            error_message=str(exc),
            payload=raw_payload,
            timestamp=datetime.now(UTC),
        )
