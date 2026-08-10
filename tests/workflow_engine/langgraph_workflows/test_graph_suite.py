"""
LangGraph Workflow Suite (Phase 20).
Extended coverage for SmartGraph and DeepResearchGraph using the actual API.
- LoopGuard: evaluate() static method
- SmartGraphState: TypedDict-based (accessed as dict)
- SmartGraphBuilder: compilation and ainvoke
- Edge routing: route_after_execution returns 'loop' or 'proceed'
- DeepResearchGraphBuilder: compilation and ainvoke
- Edge routing: route_need_more_information returns 'search_again' or 'proceed'
- ModeDispatcher routing
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config.settings import LLMServiceConfig
from app.models.execution_plan import (
    ExecutionPlan,
    IntentCategory,
    ReasoningMode,
    Skill,
    UserMode,
)
from app.models.pipeline_context import PipelineContext
from app.context.schemas import ContextBundle
from app.workflow_engine.langgraph_workflows.shared.loop_guard import LoopGuard
from app.workflow_engine.langgraph_workflows.smart.graph import SmartGraphBuilder
from app.workflow_engine.langgraph_workflows.smart.state import SmartGraphState
from app.workflow_engine.langgraph_workflows.smart.edges import route_after_execution
from app.workflow_engine.langgraph_workflows.deep_research.graph import DeepResearchGraphBuilder
from app.workflow_engine.langgraph_workflows.deep_research.state import DeepResearchGraphState
from app.workflow_engine.langgraph_workflows.deep_research.edges import route_need_more_information
from app.workflow_engine import ModeDispatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_dispatcher() -> MagicMock:
    td = MagicMock()
    td.dispatch = AsyncMock(return_value=[])
    return td


# ---------------------------------------------------------------------------
# LoopGuard (static evaluate method)
# ---------------------------------------------------------------------------


def test_loop_guard_proceed_when_no_unsatisfied_tasks():
    """LoopGuard.evaluate() returns 'proceed' when all tasks are satisfied."""
    result = LoopGuard.evaluate(
        current_iteration=2,
        max_iterations=5,
        has_unsatisfied_tasks=False,
    )
    assert result == "proceed"


def test_loop_guard_loop_when_unsatisfied_and_under_cap():
    """LoopGuard.evaluate() returns 'loop' when unsatisfied tasks remain under cap."""
    result = LoopGuard.evaluate(
        current_iteration=1,
        max_iterations=5,
        has_unsatisfied_tasks=True,
    )
    assert result == "loop"


def test_loop_guard_proceed_when_iteration_hits_cap():
    """LoopGuard.evaluate() returns 'proceed' when iteration >= max_iterations."""
    result = LoopGuard.evaluate(
        current_iteration=5,
        max_iterations=5,
        has_unsatisfied_tasks=True,
    )
    assert result == "proceed"


def test_loop_guard_proceed_when_iteration_exceeds_cap():
    """LoopGuard.evaluate() returns 'proceed' when iteration > max_iterations."""
    result = LoopGuard.evaluate(
        current_iteration=10,
        max_iterations=3,
        has_unsatisfied_tasks=True,
    )
    assert result == "proceed"


# ---------------------------------------------------------------------------
# SmartGraphState (TypedDict — accessed as dict)
# ---------------------------------------------------------------------------


def test_smart_graph_state_is_dict_compatible():
    """SmartGraphState instances can be created and accessed as dicts."""
    state: SmartGraphState = {
        "user_message": "Test",
        "conversation_id": "conv_1",
        "loop_iteration_count": 0,
        "max_iterations": 5,
        "sub_tasks": [],
        "satisfied_sub_tasks": [],
    }
    assert state["user_message"] == "Test"
    assert state["loop_iteration_count"] == 0


def test_smart_graph_state_get_with_default():
    """SmartGraphState dict.get() returns defaults for missing keys."""
    state: SmartGraphState = {
        "user_message": "Hello",
        "conversation_id": "conv_2",
    }
    assert state.get("max_iterations", 5) == 5
    assert state.get("sub_tasks", []) == []


# ---------------------------------------------------------------------------
# SmartGraph Construction
# ---------------------------------------------------------------------------


def test_smart_graph_builder_compiles_graph():
    """SmartGraphBuilder.build() returns a compiled LangGraph without errors."""
    td = _make_tool_dispatcher()
    builder = SmartGraphBuilder(tool_dispatcher=td, max_iterations=3)
    graph = builder.build()
    assert graph is not None
    assert hasattr(graph, "ainvoke")


@pytest.mark.asyncio
async def test_smart_graph_ainvoke_with_minimal_state():
    """SmartGraph.ainvoke() completes a cycle with a minimal initial state."""
    td = _make_tool_dispatcher()
    builder = SmartGraphBuilder(tool_dispatcher=td, max_iterations=2)
    graph = builder.build()

    initial_state: SmartGraphState = {
        "user_message": "What is LangGraph?",
        "conversation_id": "conv_smart_001",
        "loop_iteration_count": 0,
        "max_iterations": 2,
        "sub_tasks": [],
        "satisfied_sub_tasks": [],
    }

    result = await graph.ainvoke(initial_state)
    assert result is not None


# ---------------------------------------------------------------------------
# SmartGraph Edge Routing
# ---------------------------------------------------------------------------


def test_smart_graph_route_proceed_when_max_reached():
    """route_after_execution forces 'proceed' when max iterations hit."""
    state: SmartGraphState = {
        "user_message": "Test",
        "conversation_id": "conv1",
        "loop_iteration_count": 5,
        "max_iterations": 5,
        "sub_tasks": [MagicMock(id="task_1")],
        "satisfied_sub_tasks": [],
    }
    decision = route_after_execution(state)
    assert decision == "proceed"


def test_smart_graph_route_proceed_when_no_unsatisfied():
    """route_after_execution returns 'proceed' when all sub-tasks are satisfied."""
    task = MagicMock()
    task.id = "task_1"
    state: SmartGraphState = {
        "user_message": "Test",
        "conversation_id": "conv1",
        "loop_iteration_count": 1,
        "max_iterations": 5,
        "sub_tasks": [task],
        "satisfied_sub_tasks": ["task_1"],
    }
    decision = route_after_execution(state)
    assert decision == "proceed"


def test_smart_graph_route_loop_when_tasks_unsatisfied():
    """route_after_execution returns 'loop' when unsatisfied tasks exist under cap."""
    task = MagicMock()
    task.id = "task_1"
    state: SmartGraphState = {
        "user_message": "Search for news",
        "conversation_id": "conv1",
        "loop_iteration_count": 1,
        "max_iterations": 5,
        "sub_tasks": [task],
        "satisfied_sub_tasks": [],  # task_1 not satisfied
    }
    decision = route_after_execution(state)
    assert decision == "loop"


# ---------------------------------------------------------------------------
# DeepResearchGraph State
# ---------------------------------------------------------------------------


def test_deep_research_state_is_dict_compatible():
    """DeepResearchGraphState can be created and accessed as a dict."""
    state: DeepResearchGraphState = {
        "user_message": "Research climate change",
        "conversation_id": "conv_dr_001",
        "loop_iteration_count": 0,
        "max_iterations": 4,
        "coverage_sufficient": False,
        "findings": [],
    }
    assert state["conversation_id"] == "conv_dr_001"
    assert state["loop_iteration_count"] == 0


# ---------------------------------------------------------------------------
# DeepResearchGraph Construction
# ---------------------------------------------------------------------------


def test_deep_research_graph_builder_compiles():
    """DeepResearchGraphBuilder.build() returns a compiled LangGraph."""
    td = _make_tool_dispatcher()
    builder = DeepResearchGraphBuilder(tool_dispatcher=td, max_iterations=4)
    graph = builder.build()
    assert graph is not None
    assert hasattr(graph, "ainvoke")


@pytest.mark.asyncio
async def test_deep_research_graph_ainvoke():
    """DeepResearchGraph.ainvoke() completes with minimal initial state."""
    td = _make_tool_dispatcher()
    builder = DeepResearchGraphBuilder(tool_dispatcher=td, max_iterations=2)
    graph = builder.build()

    initial_state: DeepResearchGraphState = {
        "user_message": "Summarize the impact of transformers on NLP",
        "conversation_id": "conv_dr_002",
        "loop_iteration_count": 0,
        "max_iterations": 2,
        "coverage_sufficient": False,
        "findings": [],
    }

    result = await graph.ainvoke(initial_state)
    assert result is not None


# ---------------------------------------------------------------------------
# DeepResearchGraph Edge Routing
# ---------------------------------------------------------------------------


def test_deep_research_edge_returns_search_again_when_not_covered():
    """route_need_more_information returns 'search_again' when coverage is insufficient."""
    state: DeepResearchGraphState = {
        "user_message": "Research quantum computing",
        "conversation_id": "conv_dr_003",
        "loop_iteration_count": 1,
        "max_iterations": 4,
        "coverage_sufficient": False,
        "findings": [],
    }
    decision = route_need_more_information(state)
    assert decision == "search_again"


def test_deep_research_edge_proceeds_when_coverage_sufficient():
    """route_need_more_information returns 'proceed' when coverage is sufficient."""
    state: DeepResearchGraphState = {
        "user_message": "Research quantum computing",
        "conversation_id": "conv_dr_003",
        "loop_iteration_count": 2,
        "max_iterations": 4,
        "coverage_sufficient": True,  # ← covered!
        "findings": ["Source A", "Source B"],
    }
    decision = route_need_more_information(state)
    assert decision == "proceed"


def test_deep_research_edge_proceeds_when_iteration_cap_hit():
    """route_need_more_information returns 'proceed' when max iterations exceeded."""
    state: DeepResearchGraphState = {
        "user_message": "Research quantum computing",
        "conversation_id": "conv_dr_004",
        "loop_iteration_count": 4,  # At max
        "max_iterations": 4,
        "coverage_sufficient": False,
        "findings": [],
    }
    decision = route_need_more_information(state)
    assert decision == "proceed"


# ---------------------------------------------------------------------------
# ModeDispatcher → Graph / Handler Routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mode_dispatcher_routes_smart_to_langgraph():
    """ModeDispatcher routes 'smart' mode to the langgraph engine type."""
    td = _make_tool_dispatcher()
    smart_graph = SmartGraphBuilder(tool_dispatcher=td, max_iterations=2).build()
    dr_graph = DeepResearchGraphBuilder(tool_dispatcher=td, max_iterations=2).build()

    dispatcher = ModeDispatcher(
        handlers={},
        graphs={"smart": smart_graph, "deep_research": dr_graph},
    )
    assert dispatcher.engine_type_for("smart") == "langgraph"
    assert dispatcher.engine_type_for("deep_research") == "langgraph"


@pytest.mark.asyncio
async def test_mode_dispatcher_routes_default_to_mode_handler():
    """ModeDispatcher routes 'default' mode to the mode_handler engine type."""
    from app.workflow_engine.mode_handlers import DefaultHandler

    handler = DefaultHandler()
    dispatcher = ModeDispatcher(
        handlers={"default": handler},
        graphs={},
    )
    assert dispatcher.engine_type_for("default") == "mode_handler"
