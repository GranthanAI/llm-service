"""
Workflow Engine Execution Coordinator.
Implements LLD v2.0 Section 11.1 and HLD v2.0 Section 10.
"""

import structlog

from app.config.logging import get_logger
from app.models.execution_plan import ExecutionPlan
from app.models.pipeline_context import PipelineContext
from app.utils.tracing import trace_span
from app.workflow_engine.mode_dispatcher import ModeDispatcher
from app.workflow_engine.workflow_result import WorkflowResult


class WorkflowEngine:
    """
    Primary workflow coordinator.
    Delegates planned execution to ModeDispatcher and produces normalized WorkflowResult.
    """

    def __init__(
        self,
        mode_dispatcher: ModeDispatcher,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.mode_dispatcher: ModeDispatcher = mode_dispatcher
        self.logger = logger or get_logger("workflow_engine")

    @trace_span("workflow_engine_execute")
    async def execute(self, plan: ExecutionPlan, ctx: PipelineContext) -> WorkflowResult:
        """Execute request according to ExecutionPlan through ModeDispatcher."""
        self.logger.info(
            "Executing workflow",
            mode=plan.mode.value if hasattr(plan.mode, "value") else str(plan.mode),
            conversation_id=ctx.conversation_id,
            request_id=ctx.request_id,
        )

        result: WorkflowResult = await self.mode_dispatcher.dispatch(plan, ctx)

        self.logger.info(
            "Workflow execution completed",
            mode=result.mode,
            engine_type=result.engine_type,
            tool_outputs_count=len(result.tool_outputs),
            has_draft=result.draft_content is not None,
            conversation_id=ctx.conversation_id,
        )
        return result
