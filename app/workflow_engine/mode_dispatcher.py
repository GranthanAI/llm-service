"""
Mode Dispatcher Engine.
Implements LLD v2.0 Section 11.3 and Section 11.4.
"""

from typing import Any

import structlog

from app.config.logging import get_logger
from app.exceptions.analysis import UnknownModeError
from app.models.execution_plan import ExecutionPlan, UserMode
from app.models.pipeline_context import PipelineContext
from app.utils.metrics import WORKFLOW_DISPATCH_TOTAL
from app.utils.tracing import trace_span
from app.workflow_engine.workflow_result import WorkflowResult

MODE_HANDLER_KEYS: set[str] = {"default", "tutor", "code", "ask_files", "web_search"}
LANGGRAPH_KEYS: set[str] = {"smart", "deep_research"}

# Verification of disjoint sets at module load (LLD v2.0 Section 11.4)
assert MODE_HANDLER_KEYS.isdisjoint(LANGGRAPH_KEYS), (
    "Mode Handler and LangGraph keys must be strictly disjoint"
)
assert (MODE_HANDLER_KEYS | LANGGRAPH_KEYS) == {m.value for m in UserMode}, (
    "All UserMode enum values must be registered in either handlers or graphs"
)


class ModeDispatcher:
    """
    Sole routing decision point in the system.
    Translates ExecutionPlan.mode into concrete engine execution (ModeHandler or LangGraph).
    """

    def __init__(
        self,
        handlers: dict[str, Any] | None = None,
        graphs: dict[str, Any] | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.handlers: dict[str, Any] = handlers or {}
        self.graphs: dict[str, Any] = graphs or {}
        self.logger = logger or get_logger("mode_dispatcher")

        # Invariant checked at construction: no mode key can appear in both maps
        overlap = set(self.handlers.keys()) & set(self.graphs.keys())
        if overlap:
            raise ValueError(f"Modes registered in both handlers and graphs: {overlap}")

    def engine_type_for(self, mode: str) -> str:
        """Resolve engine type for given mode ('mode_handler' or 'langgraph')."""
        if mode in self.handlers or mode in MODE_HANDLER_KEYS:
            return "mode_handler"
        if mode in self.graphs or mode in LANGGRAPH_KEYS:
            return "langgraph"
        raise UnknownModeError(mode)

    @trace_span("mode_dispatcher_dispatch")
    async def dispatch(self, plan: ExecutionPlan, ctx: PipelineContext) -> WorkflowResult:
        """Dispatch execution plan to the appropriate handler or graph workflow."""
        mode_val = plan.mode.value if hasattr(plan.mode, "value") else str(plan.mode)
        engine_type = self.engine_type_for(mode_val)

        # Update context metadata and telemetry metrics
        ctx.engine_type = engine_type
        ctx.plan = plan
        WORKFLOW_DISPATCH_TOTAL.labels(mode=mode_val, engine_type=engine_type).inc()

        self.logger.info(
            "Dispatching execution plan",
            mode=mode_val,
            engine_type=engine_type,
            skill=plan.skill.value if hasattr(plan.skill, "value") else str(plan.skill),
            conversation_id=ctx.conversation_id,
            request_id=ctx.request_id,
        )

        # 1. Deterministic Mode Handler execution
        if mode_val in self.handlers:
            handler = self.handlers[mode_val]
            handler_output = await handler.handle(plan, ctx)
            return WorkflowResult.from_mode_handler(handler_output)

        # 2. Iterative LangGraph Workflow execution
        if mode_val in self.graphs:
            graph = self.graphs[mode_val]
            initial_state = self._build_initial_graph_state(plan, ctx)
            final_state = await graph.ainvoke(initial_state)
            return WorkflowResult.from_graph_state(final_state)

        # Fallback if mode is recognized by enum but missing handler implementation
        self.logger.error("No handler or graph registered for mode", mode=mode_val)
        raise UnknownModeError(mode_val)

    def _build_initial_graph_state(
        self, plan: ExecutionPlan, ctx: PipelineContext
    ) -> dict[str, Any]:
        """Construct initial state payload for LangGraph workflow execution."""
        history = (
            ctx.context_bundle.memory.short_term_messages
            if ctx.context_bundle and ctx.context_bundle.memory
            else []
        )
        return {
            "mode": plan.mode.value if hasattr(plan.mode, "value") else str(plan.mode),
            "user_message": ctx.user_message,
            "conversation_history": list(history),
            "loop_iteration_count": 0,
            "max_iterations": plan.max_iterations,
            "tool_results": [],
            "draft_response": None,
        }
