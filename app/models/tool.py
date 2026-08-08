"""
Tool Framework Data Models and Schemas.
Implements LLD v2.0 Section 15.1 and Section 15.6.
"""

from typing import Any

from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """Tool metadata description and JSON parameter schema."""

    name: str
    description: str
    version: str = "1.0.0"
    parameters: dict[str, Any] = Field(default_factory=dict)
    examples: list[dict[str, Any]] = Field(default_factory=list)


class ToolParams(BaseModel):
    """Input payload delivered to a tool execution."""

    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    user_id: str | None = None
    conversation_id: str | None = None


class ToolResult(BaseModel):
    """Normalized output produced by any tool execution."""

    tool_name: str
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Result of parameter validation prior to tool execution."""

    valid: bool
    error: str | None = None
