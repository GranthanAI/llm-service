"""
Tool Exceptions.
Implements LLD v2.0 Section 28.1.
"""

from app.exceptions.base import BaseLLMServiceError, PermanentError, RetriableError


class ToolError(BaseLLMServiceError):
    """Base exception for tool execution failures."""

    def __init__(self, message: str, tool_name: str | None = None, error_code: str | None = None):
        super().__init__(message, error_code=error_code or "TOOL_ERROR")
        self.tool_name = tool_name


class RequiredToolFailedError(PermanentError, ToolError):
    """Raised when a tool marked required=True fails."""

    def __init__(self, message: str, tool_name: str | None = None):
        super().__init__(message, tool_name=tool_name, error_code="REQUIRED_TOOL_FAILED")


class ToolTimeoutError(RetriableError, ToolError):
    """Raised when a tool execution exceeds timeout limit."""

    def __init__(self, message: str, tool_name: str | None = None):
        super().__init__(message, tool_name=tool_name, error_code="TOOL_TIMEOUT")


class UnknownToolError(PermanentError, ToolError):
    """Raised when a requested tool is not registered in ToolRegistry."""

    def __init__(self, message: str, tool_name: str | None = None):
        super().__init__(message, tool_name=tool_name, error_code="UNKNOWN_TOOL")


class ToolValidationError(PermanentError, ToolError):
    """Raised when tool input parameter validation fails."""

    def __init__(self, message: str, tool_name: str | None = None):
        super().__init__(message, tool_name=tool_name, error_code="TOOL_VALIDATION_ERROR")
