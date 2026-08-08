"""
Web Search Mode Handler Implementation.
Implements LLD v2.0 Section 12.6.
"""

from typing import Any

import structlog

from app.config.logging import get_logger
from app.models.execution_plan import ExecutionPlan, ToolCall
from app.models.pipeline_context import PipelineContext
from app.models.tool import ToolResult
from app.workflow_engine.mode_handlers.base import (
    PromptRegistryProtocol,
    ToolDispatcherProtocol,
)
from app.workflow_engine.workflow_result import ModeHandlerOutput


class WebSearchHandler:
    """
    Web Search mode handler.
    Guarantees required web_search tool execution for current events / live web grounding.
    """

    def __init__(
        self,
        tool_dispatcher: ToolDispatcherProtocol | None = None,
        prompt_registry: PromptRegistryProtocol | Any | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.tool_dispatcher = tool_dispatcher
        self.prompt_registry = prompt_registry
        self.logger = logger or get_logger("web_search_handler")

    async def handle(self, plan: ExecutionPlan, ctx: PipelineContext) -> ModeHandlerOutput:
        """Execute web_search mode handling."""
        # Ensure web_search tool is present defensively (LLD2 Section 12.6)
        tools = list(plan.tools)
        has_web_search = any(t.tool_name == "web_search" for t in tools)
        if not has_web_search:
            tools.append(
                ToolCall(
                    tool_name="web_search",
                    params={"query": ctx.user_message},
                    parallel=False,
                    required=True,
                )
            )

        tool_outputs: list[ToolResult] = []
        if self.tool_dispatcher:
            self.logger.debug(
                "Dispatching required web search",
                tools_count=len(tools),
                conversation_id=ctx.conversation_id,
            )
            tool_outputs = await self.tool_dispatcher.dispatch(tools)

        history = (
            ctx.context_bundle.memory.short_term_messages
            if ctx.context_bundle and ctx.context_bundle.memory
            else []
        )

        return ModeHandlerOutput(
            mode="web_search",
            tool_outputs=tool_outputs,
            conversation_history=list(history),
            user_message=ctx.user_message,
        )
