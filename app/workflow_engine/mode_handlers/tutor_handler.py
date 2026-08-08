"""
Tutor Mode Handler Implementation.
Implements LLD v2.0 Section 12.3.
"""

from typing import Any

import structlog

from app.config.logging import get_logger
from app.models.execution_plan import ExecutionPlan
from app.models.pipeline_context import PipelineContext
from app.models.tool import ToolResult
from app.workflow_engine.mode_handlers.base import (
    PromptRegistryProtocol,
    ToolDispatcherProtocol,
)
from app.workflow_engine.workflow_result import ModeHandlerOutput


class TutorHandler:
    """
    Pedagogical tutor mode handler.
    Stateless, single-pass; pedagogical structure is formulated in the system prompt.
    """

    def __init__(
        self,
        tool_dispatcher: ToolDispatcherProtocol | None = None,
        prompt_registry: PromptRegistryProtocol | Any | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.tool_dispatcher = tool_dispatcher
        self.prompt_registry = prompt_registry
        self.logger = logger or get_logger("tutor_handler")

    async def handle(self, plan: ExecutionPlan, ctx: PipelineContext) -> ModeHandlerOutput:
        """Execute tutor mode handling."""
        tool_outputs: list[ToolResult] = []
        if plan.tools and self.tool_dispatcher:
            self.logger.debug(
                "Dispatching supplementary tools for tutor handler",
                tools_count=len(plan.tools),
                conversation_id=ctx.conversation_id,
            )
            tool_outputs = await self.tool_dispatcher.dispatch(plan.tools)

        history = (
            ctx.context_bundle.memory.short_term_messages
            if ctx.context_bundle and ctx.context_bundle.memory
            else []
        )

        return ModeHandlerOutput(
            mode="tutor",
            tool_outputs=tool_outputs,
            conversation_history=list(history),
            user_message=ctx.user_message,
        )
