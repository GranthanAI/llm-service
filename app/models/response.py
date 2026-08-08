"""
Response and Outgoing Kafka Event Data Models.
Implements HLD v2.0 Section 26.3 and LLD v2.0 Section 7, 14.6, 21.
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from app.models.request import TraceContext


class UsageMetrics(BaseModel):
    """Token consumption metrics for request processing."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ErrorType(StrEnum):
    """Service error categorization matching LLD v2.0 Section 14.6."""

    VALIDATION = "VALIDATION"
    ANALYSIS = "ANALYSIS"
    TOOL = "TOOL"
    INFERENCE = "INFERENCE"
    PUBLISH = "PUBLISH"
    GRAPH_LOOP = "GRAPH_LOOP"
    GRPC = "GRPC"
    UNEXPECTED = "UNEXPECTED"


class ServiceError(BaseModel):
    """Structured service error representation."""

    error_type: ErrorType = ErrorType.UNEXPECTED
    error_code: str
    message: str
    retriable: bool = False
    provider: str | None = None
    tool_name: str | None = None
    mode: str | None = None
    engine_type: str | None = None
    traceback: str | None = None


class ChatResponseChunkEvent(BaseModel):
    """
    Streaming chunk event produced to Kafka topic: chat.response.chunk.
    Consumed by Conversation Service for realtime token delivery to client.
    """

    event_type: str = "chat.response.chunk"
    schema_version: str = "2.0"
    conversation_id: str
    user_id: str
    message_id: str
    chunk_index: int
    sequence_number: int
    content: str
    is_last: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatResponseGeneratedEvent(BaseModel):
    """
    Final response event produced to Kafka topic: chat.response.generated.
    Consumed by Conversation Service, Memory Service, and Analytics.
    """

    event_type: str = "chat.response.generated"
    schema_version: str = "2.0"
    response_id: str
    conversation_id: str
    user_id: str
    request_message_id: str
    full_content: str | None = None
    provider: str | None = "nvidia"
    generation_fallback_used: bool = False
    mode: str = "default"
    skill: str = "general_chat"
    engine_type: str = "mode_handler"  # "mode_handler" | "langgraph"
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    finish_reason: str = "stop"
    tools_used: list[str] = Field(default_factory=list)
    context_sources: list[str] = Field(default_factory=list)
    trace_context: TraceContext = Field(default_factory=TraceContext)
    status: str = "success"  # "success" | "error"
    error_code: str | None = None
    error_message: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryUpdateRequestedEvent(BaseModel):
    """
    Asynchronous event produced to Kafka topic: memory.update.requested.
    Signals Memory Service to synthesize long-term facts and conversation context.
    """

    event_type: str = "memory.update.requested"
    conversation_id: str
    user_id: str
    user_message: str
    assistant_response: str
    mode: str = "default"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatResponseCancelledEvent(BaseModel):
    """
    Event produced to Kafka topic: chat.response.cancelled
    when user or client cancels stream mid-generation.
    """

    event_type: str = "chat.response.cancelled"
    conversation_id: str
    user_id: str
    message_id: str
    reason: str = "client_disconnected"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatMessageDLQEvent(BaseModel):
    """
    Dead Letter Queue event produced to Kafka topic: chat.message.dlq.
    """

    original_topic: str
    original_partition: int = 0
    original_offset: int = 0
    original_key: str | None = None
    original_value: str | None = None
    error_type: str  # "DeserializationError" | "PipelineError"
    error_message: str
    failed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retry_count: int = 0
