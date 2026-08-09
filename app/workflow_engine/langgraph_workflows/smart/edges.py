"""
SmartGraph Conditional Routing Edges.
Implements LLD v2.0 Section 13.1.4.
"""

from typing import Literal

from app.workflow_engine.langgraph_workflows.shared.loop_guard import LoopGuard
from app.workflow_engine.langgraph_workflows.smart.state import SmartGraphState


def route_after_execution(state: SmartGraphState) -> Literal["loop", "proceed"]:
    """
    Evaluates whether SmartGraph should cycle back to Dynamic Tool Selection
    or proceed to Prompt Assembly and Draft Generation.
    """
    sub_tasks = state.get("sub_tasks") or []
    satisfied = set(state.get("satisfied_sub_tasks") or [])

    unsatisfied = [t for t in sub_tasks if t.id not in satisfied]
    has_unsatisfied = len(unsatisfied) > 0

    return LoopGuard.evaluate(
        current_iteration=state.get("loop_iteration_count", 0),
        max_iterations=state.get("max_iterations", 5),
        has_unsatisfied_tasks=has_unsatisfied,
        graph_name="smart",
    )
