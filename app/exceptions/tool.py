"""
Tool Exceptions.
"""


class ToolError(Exception):
    """Base exception for tool execution failures."""

    def __init__(self, message: str, tool_name: str | None = None):
        super().__init__(message)
        self.tool_name = tool_name


class RequiredToolFailedError(ToolError):
    """Raised when a tool marked required=True fails."""

    pass


class ToolTimeoutError(ToolError):
    """Raised when a tool execution exceeds timeout limit."""

    pass


class UnknownToolError(ToolError):
    """Raised when a requested tool is not registered in ToolRegistry."""

    pass


class ToolValidationError(ToolError):
    """Raised when tool input parameter validation fails."""

    pass
