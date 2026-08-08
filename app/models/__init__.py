"""
Core Models and Data Types for LLM Service.
"""

from app.models.execution_plan import (
    EngineType,
    ExecutionPlan,
    IntentCategory,
    ReasoningMode,
    Skill,
    ToolCall,
    UserMode,
)
from app.models.pipeline_context import PipelineContext
from app.models.provider import (
    CircuitState,
    ComposedPrompt,
    GenerationConfig,
    ModelLimits,
    PromptSection,
    ProviderSelection,
    ProviderType,
    TrimmedPrompt,
)
from app.models.request import (
    AnalysisRequest,
    ChatMessageCreatedEvent,
    TraceContext,
    UserRequest,
)
from app.models.response import (
    ChatMessageDLQEvent,
    ChatResponseCancelledEvent,
    ChatResponseChunkEvent,
    ChatResponseGeneratedEvent,
    ErrorType,
    MemoryUpdateRequestedEvent,
    ServiceError,
    UsageMetrics,
)
from app.models.tool import (
    ToolParams,
    ToolResult,
    ToolSchema,
    ValidationResult,
)

__all__ = [
    # Enums
    "IntentCategory",
    "UserMode",
    "Skill",
    "ReasoningMode",
    "EngineType",
    "ProviderType",
    "CircuitState",
    "ErrorType",
    # Request & Event
    "TraceContext",
    "ChatMessageCreatedEvent",
    "AnalysisRequest",
    "UserRequest",
    # Response & Event
    "UsageMetrics",
    "ChatResponseChunkEvent",
    "ChatResponseGeneratedEvent",
    "MemoryUpdateRequestedEvent",
    "ChatResponseCancelledEvent",
    "ChatMessageDLQEvent",
    "ServiceError",
    # Execution & Planning
    "ToolCall",
    "ExecutionPlan",
    # Context
    "PipelineContext",
    # Tools
    "ToolSchema",
    "ToolParams",
    "ToolResult",
    "ValidationResult",
    # Provider & Prompts
    "ModelLimits",
    "ProviderSelection",
    "GenerationConfig",
    "PromptSection",
    "ComposedPrompt",
    "TrimmedPrompt",
]
