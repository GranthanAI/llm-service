"""
Exceptions module exports.
Implements LLD v2.0 Section 28.1 error taxonomy.
"""

from app.exceptions.analysis import (
    AnalysisError,
    ContextOverflowError,
    CriticalAnalysisError,
    PlanParseError,
    UnknownModeError,
)
from app.exceptions.base import (
    BaseLLMServiceError,
    FatalError,
    PermanentError,
    RetriableError,
)
from app.exceptions.grpc import (
    GRPCError,
    GRPCTimeoutError,
    GRPCUnavailableError,
)
from app.exceptions.provider import (
    AllProvidersFailedError,
    CircuitOpenError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.exceptions.tool import (
    RequiredToolFailedError,
    ToolError,
    ToolTimeoutError,
    ToolValidationError,
    UnknownToolError,
)

__all__ = [
    "BaseLLMServiceError",
    "RetriableError",
    "PermanentError",
    "FatalError",
    "ProviderError",
    "AllProvidersFailedError",
    "ProviderTimeoutError",
    "ProviderRateLimitError",
    "CircuitOpenError",
    "ToolError",
    "RequiredToolFailedError",
    "ToolTimeoutError",
    "UnknownToolError",
    "ToolValidationError",
    "GRPCError",
    "GRPCUnavailableError",
    "GRPCTimeoutError",
    "AnalysisError",
    "PlanParseError",
    "CriticalAnalysisError",
    "UnknownModeError",
    "ContextOverflowError",
]
