"""
DeepResearchGraph Conditional Routing Edges.
Implements LLD v2.0 Section 13.2.4.
"""

from typing import Literal

from app.workflow_engine.langgraph_workflows.deep_research.state import (
    DeepResearchGraphState,
)
from app.workflow_engine.langgraph_workflows.shared.loop_guard import LoopGuard


def route_need_more_information(
    state: DeepResearchGraphState,
) -> Literal["search_again", "proceed"]:
    """
    Evaluates whether DeepResearchGraph needs additional research iterations
    or should proceed to source comparison, thematic summarization, and report generation.
    """
    coverage_sufficient = state.get("coverage_sufficient", False)
    current_iter = state.get("loop_iteration_count", 0)
    max_iter = state.get("max_iterations", 5)

    if coverage_sufficient:
        return "proceed"

    decision = LoopGuard.evaluate(
        current_iteration=current_iter,
        max_iterations=max_iter,
        has_unsatisfied_tasks=not coverage_sufficient,
        graph_name="deep_research",
    )

    if decision == "proceed":
        return "proceed"

    return "search_again"
