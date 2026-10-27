"""
Shared Pre-Dispatch Pipeline Context.
Implements LLD v2.0 Section 14.1 and HLD v2.0 Section 6.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from app.context.schemas import ContextBundle
from app.models.execution_plan import ExecutionPlan, UserMode
from app.models.generation_preferences import GenerationPreferences, normalize_preferences
from app.models.request import CustomModeSnapshot

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
    # Assistant message this turn answers (see ChatMessageCreatedEvent). Echoed
    # unchanged on every chunk / final / error / cancellation event.
    response_message_id: str | None = None
    request_id: str
    trace_id: str = ""
    span_id: str = ""

    # Input
    user_message: str
    mode_hint: UserMode | None = None
    file_ids: list[str] = Field(default_factory=list)
    # Tool ids the user explicitly selected (validated against the registry before use).
    selected_tools: list[str] = Field(default_factory=list)
    include_diagrams: bool = False
    # Set when attached documents were searched but nothing relevant was found / some were skipped.
    document_notice: str | None = None
    # Standalone web-search query derived from the message + conversation (None = use the raw message).
    search_query: str | None = None
    # Earlier turns from the request event, used when the memory service cannot supply history.
    event_history: list[tuple[str, str]] = Field(default_factory=list)
    # Ancestor turns inherited by this branch (oldest first). Always used, ahead of the branch's own turns.
    inherited_history: list[tuple[str, str]] = Field(default_factory=list)
    # The user's custom mode for this request (immutable snapshot from Conversation Service), or None.
    custom_mode: CustomModeSnapshot | None = None
    # Branch lineage with boundaries, root first: [{conversation_id, upto_message_id}].
    ancestors: list[dict[str, str]] = Field(default_factory=list)
    # Prompt addendum for optional Mermaid/Manim blocks (set by the pipeline from the final mode; "" = none).
    visualization_guidance: str = ""
    # User generation preferences (length, tone, language, instructions, citations).
    preferences: GenerationPreferences = Field(default_factory=GenerationPreferences)

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
            response_message_id=event.response_message_id,
            request_id=request_id,
            trace_id=trace_id or event.trace_context.traceparent,
            span_id=span_id,
            user_message=event.content,
            mode_hint=mode_hint_val,
            file_ids=event.file_ids,
            selected_tools=list(event.selected_tools or []),
            include_diagrams=bool(event.include_diagrams),
            event_history=[(h.role, h.content) for h in event.history if not h.inherited],
            inherited_history=[(h.role, h.content) for h in event.history if h.inherited],
            ancestors=[a.model_dump() for a in event.ancestors],
            custom_mode=event.custom_mode,
            preferences=normalize_preferences(event.preferences),
            started_at=datetime.now(UTC),
        )
