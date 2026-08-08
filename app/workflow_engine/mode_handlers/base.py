"""
Mode Handler Protocol and Interfaces.
Implements LLD v2.0 Section 12.1.
Strictly plain Python Protocols with ZERO LangGraph dependencies.
"""

from typing import Any, Protocol

from app.models.execution_plan import ExecutionPlan, ToolCall
from app.models.pipeline_context import PipelineContext
from app.models.tool import ToolResult
from app.workflow_engine.workflow_result import ModeHandlerOutput


class ToolDispatcherProtocol(Protocol):
    """Protocol for dispatching tool calls from Mode Handlers."""

    async def dispatch(self, tools: list[ToolCall]) -> list[ToolResult]: ...


class PromptRegistryProtocol(Protocol):
    """Protocol for prompt registry access."""

    def get(self, name: str) -> Any: ...


class ModeHandler(Protocol):
    """Protocol implemented by all 5 deterministic Mode Handlers."""

    async def handle(self, plan: ExecutionPlan, ctx: PipelineContext) -> ModeHandlerOutput: ...
