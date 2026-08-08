"""
Ask Files Mode Handler Implementation.
Implements LLD v2.0 Section 12.5.
"""

from typing import Any

import structlog

from app.config.logging import get_logger
from app.models.execution_plan import ExecutionPlan
from app.models.pipeline_context import PipelineContext
from app.workflow_engine.mode_handlers.base import PromptRegistryProtocol
from app.workflow_engine.workflow_result import ModeHandlerOutput


class AskFilesHandler:
    """
    Document analysis and file Q&A mode handler.
    Grounding arrives pre-fetched via ctx.context_bundle.retrieval; no optional tools needed.
    """

    def __init__(
        self,
        prompt_registry: PromptRegistryProtocol | Any | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.prompt_registry = prompt_registry
        self.logger = logger or get_logger("ask_files_handler")

    async def handle(self, plan: ExecutionPlan, ctx: PipelineContext) -> ModeHandlerOutput:
        """Execute ask_files mode handling."""
        if not ctx.file_ids:
            self.logger.warning(
                "ask_files handler invoked with empty file_ids",
                conversation_id=ctx.conversation_id,
                request_id=ctx.request_id,
            )

        history = (
            ctx.context_bundle.memory.short_term_messages
            if ctx.context_bundle and ctx.context_bundle.memory
            else []
        )

        return ModeHandlerOutput(
            mode="ask_files",
            tool_outputs=[],
            conversation_history=list(history),
            user_message=ctx.user_message,
        )
