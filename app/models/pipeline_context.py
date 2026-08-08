"""
Shared Pre-Dispatch Pipeline Context.
Implements LLD v2.0 Section 14.1 and HLD v2.0 Section 6.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from app.context.schemas import ContextBundle
from app.models.execution_plan import ExecutionPlan, UserMode

if TYPE_CHECKING:
    from app.models.request import ChatMessageCreatedEvent


class PipelineContext(BaseModel):
    """
    Shared pre-dispatch context passed through Context Collection,
    Request Analysis, and into Workflow Dispatch.
    """

    # Identity
    conversation_id: str
    user_id: str
    message_id: str
    request_id: str
    trace_id: str = ""
    span_id: str = ""

    # Input
    user_message: str
    mode_hint: UserMode | None = None
    file_ids: list[str] = Field(default_factory=list)

    # Context Bundle (set by ContextCollector)
    context_bundle: ContextBundle | None = None
    context_collected_at: datetime | None = None

    # Request Analysis (set by RequestAnalyzer)
    plan: ExecutionPlan | None = None

    # Post-dispatch (set during Generation / Streaming / Publishing)
    engine_type: str | None = None  # "mode_handler" | "langgraph"
    selected_provider: str | None = None  # "nvidia" | "gemini"
    generation_fallback_used: bool = False
    usage: Any | None = None  # UsageMetrics
    cost_usd: float | None = None

    # Metadata & Timestamps
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    inference_started_at: datetime | None = None
    first_chunk_at: datetime | None = None
    completed_at: datetime | None = None
    pii_detected: bool = False
    safety_check_failed: bool = False

    @classmethod
    def from_event(
        cls,
        event: "ChatMessageCreatedEvent",
        request_id: str,
        trace_id: str = "",
        span_id: str = "",
    ) -> "PipelineContext":
        """Construct a new PipelineContext from an incoming Kafka event."""
        mode_hint_val = None
        if event.mode_hint:
            try:
                mode_hint_val = UserMode(event.mode_hint.lower())
            except ValueError:
                mode_hint_val = None

        return cls(
            conversation_id=event.conversation_id,
            user_id=event.user_id,
            message_id=event.message_id,
            request_id=request_id,
            trace_id=trace_id or event.trace_context.traceparent,
            span_id=span_id,
            user_message=event.content,
            mode_hint=mode_hint_val,
            file_ids=event.file_ids,
            started_at=datetime.now(UTC),
        )
