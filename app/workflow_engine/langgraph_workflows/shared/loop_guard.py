"""
LangGraph Reusable Loop Guard.
Implements LLD v2.0 Section 13.1.4, Section 13.3, and HLD Section 12.2 & 31.2.
"""

from typing import Literal

import structlog

from app.config.logging import get_logger
from app.utils.metrics import LANGGRAPH_LOOP_CAPPED

logger = get_logger("loop_guard")


class LoopGuard:
    """
    Guards against runaway agentic loops in SmartGraph and DeepResearchGraph.
    Enforces deterministic iteration caps and emits telemetry on forced exits.
    """

    @staticmethod
    def evaluate(
        current_iteration: int,
        max_iterations: int,
        has_unsatisfied_tasks: bool,
        graph_name: str = "smart",
        log: structlog.stdlib.BoundLogger | None = None,
    ) -> Literal["loop", "proceed"]:
        """
        Evaluate whether the graph should continue looping or proceed to synthesis.
        """
        active_logger = log or logger

        if not has_unsatisfied_tasks:
            return "proceed"

        if current_iteration >= max_iterations:
            active_logger.warning(
                "LangGraph loop iteration cap reached — forcing termination to synthesis",
                graph=graph_name,
                iterations=current_iteration,
                max_iterations=max_iterations,
            )
            LANGGRAPH_LOOP_CAPPED.labels(graph=graph_name).inc()
            return "proceed"

        return "loop"
