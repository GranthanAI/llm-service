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


class ChatMessageCreatedEvent(BaseModel):
    """
    Kafka event consumed from topic: chat.message.created.
    Produced by Conversation Service.
    """

    event_type: str = "chat.message.created"
    schema_version: str = "2.0"
    message_id: str
    conversation_id: str
    user_id: str
    content: str
    mode_hint: str | None = None
    file_ids: list[str] = Field(default_factory=list)
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
