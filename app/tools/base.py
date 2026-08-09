"""
Abstract Base Tool and Interfaces.
Implements LLD v2.0 Section 15.1 and HLD v2.0 Section 14.1.
"""

from abc import ABC, abstractmethod

from app.models.tool import (
    ToolParams,
    ToolResult,
    ToolSchema,
    ValidationResult,
)


class BaseTool(ABC):
    """
    Abstract Base Class for all tools in the LLM Service.
    Stateless, asynchronous, and self-describing via ToolSchema.
    """

    name: str
    description: str
    version: str = "1.0.0"
    timeout_ms: int = 5000
    is_async: bool = True

    @abstractmethod
    async def execute(self, params: ToolParams) -> ToolResult:
        """Execute tool operation asynchronously."""
        pass

    def validate_params(self, params: ToolParams) -> ValidationResult:
        """
        Validate tool parameters against expected schema.
        Default implementation checks that parameters dict is present.
        """
        if params.params is None or not isinstance(params.params, dict):
            return ValidationResult(valid=False, error="Tool parameters must be a dictionary")
        return ValidationResult(valid=True)

    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Return self-describing schema for registration and LLM tool calling."""
        pass

    def is_available(self) -> bool:
        """Return True if tool dependencies and API keys are available for execution."""
        return True
