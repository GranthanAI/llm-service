"""
Request Analysis Exceptions.
Implements LLD v2.0 Section 28.1.
"""

from app.exceptions.base import BaseLLMServiceError, PermanentError


class AnalysisError(BaseLLMServiceError):
    """Base exception for Request Analyzer failures."""

    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message, error_code=error_code or "ANALYSIS_ERROR")


class PlanParseError(PermanentError, AnalysisError):
    """Raised when JSON parsing of Groq analysis output fails."""

    def __init__(self, message: str = "Failed to parse analysis JSON output"):
        super().__init__(message, error_code="PLAN_PARSE_ERROR")


class CriticalAnalysisError(PermanentError, AnalysisError):
    """Raised when analysis fails critically and cannot recover."""

    def __init__(self, message: str = "Critical analysis error"):
        super().__init__(message, error_code="CRITICAL_ANALYSIS_ERROR")


class UnknownModeError(PermanentError, AnalysisError):
    """Raised when execution plan specifies an unregistered or unsupported mode."""

    def __init__(self, mode: str):
        super().__init__(f"Unknown or unsupported mode: '{mode}'", error_code="UNKNOWN_MODE")
        self.mode = mode


class ContextOverflowError(PermanentError, BaseLLMServiceError):
    """Raised when prompt cannot fit into model context window even after maximal trimming."""

    def __init__(self, message: str = "Prompt tokens exceed model context limit after trimming"):
        super().__init__(message, error_code="CONTEXT_OVERFLOW")
