"""
Request and Incoming Event Data Models.
Implements HLD v2.0 Section 26.2 and LLD v2.0 Section 6.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class TraceContext(BaseModel):
    """W3C Distributed Trace Context."""

    traceparent: str = ""
    tracestate: str = ""


class HistoryMessage(BaseModel):
    """One finished earlier turn, supplied by Conversation Service with the request."""

    role: str
    content: str
    # Earlier turns of an ANCESTOR conversation that this branch inherited (up to its branch point).
    inherited: bool = False
    source_conversation_id: str | None = None


class AncestorRef(BaseModel):
    """An ancestor conversation of a branch and the last of its messages in scope at the branch point."""

    conversation_id: str
    upto_message_id: str


class DefaultsApplied(BaseModel):
    """Which of mode_hint / selected_tools came from the user's saved defaults (not an
    explicit choice on this message). Saved defaults that are no longer available are
    skipped; explicit choices are validated strictly."""

    mode: bool = False
    tools: bool = False


class ChatMessageCreatedEvent(BaseModel):
    """
    Kafka event consumed from topic: chat.message.created.
    Produced by Conversation Service.
    """

    event_type: str = "chat.message.created"
    schema_version: str = "2.2"
    # Request/correlation identity. For a normal send this is the user's message
    # id; for a regeneration it is a fresh request id. It is NOT the id of the
    # assistant message that will hold the reply.
    message_id: str
    # Identity of the pending assistant message this request will fill, allocated
    # by Conversation Service before dispatch. Must be echoed unchanged on every
    # response event. Optional only so that a legacy event deserialises; such an
    # event is rejected to the DLQ (never answered under a guessed id).
    response_message_id: str | None = None
    conversation_id: str
    user_id: str
    content: str
    mode_hint: str | None = None
    selected_tools: list[str] = Field(default_factory=list)
    file_ids: list[str] = Field(default_factory=list)
    # Recent finished turns before this request, oldest first. Optional/additive (older producers omit it).
    history: list[HistoryMessage] = Field(default_factory=list)
    # Branch lineage (root first). Forwarded to the memory service so ancestor summaries are cut at the branch point.
    ancestors: list[AncestorRef] = Field(default_factory=list)
    # 2.2 (additive): generation preferences resolved by Conversation Service at send
    # time (see app/models/generation_preferences.py) and which choices came from saved defaults.
    preferences: dict[str, Any] = Field(default_factory=dict)
    defaults_applied: DefaultsApplied = Field(default_factory=DefaultsApplied)
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace_context: TraceContext = Field(default_factory=TraceContext)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AnalysisRequest(BaseModel):
    """Payload provided to Request Analyzer for generating an ExecutionPlan."""

    user_message: str
    mode_hint: str | None = None
    file_ids: list[str] = Field(default_factory=list)
    formatted_memory: str = ""
    formatted_graph: str = ""
    formatted_retrieval: str = ""
    available_tools_schema: list[dict[str, Any]] = Field(default_factory=list)


class UserRequest(BaseModel):
    """Direct HTTP API Chat Request payload."""

    conversation_id: str
    user_id: str
    message: str
    mode_hint: str | None = None
    file_ids: list[str] = Field(default_factory=list)
