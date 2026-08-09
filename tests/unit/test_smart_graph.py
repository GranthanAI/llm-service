"""
Unit tests for SmartGraph LangGraph Orchestration (Phase 11).
Tests State, Nodes, Edges, LoopGuard, Builder Compilation, and Dispatcher Routing.
"""

from unittest.mock import AsyncMock

import pytest

from app.context.schemas import ContextBundle
from app.models.execution_plan import ExecutionPlan, UserMode
from app.models.pipeline_context import PipelineContext
from app.models.tool import ToolResult
from app.tools.dispatcher import ToolDispatcher
from app.workflow_engine.langgraph_workflows.shared.loop_guard import LoopGuard
from app.workflow_engine.langgraph_workflows.smart import (
    SmartGraphBuilder,
    SmartGraphState,
    SubTask,
    route_after_execution,
)
from app.workflow_engine.langgraph_workflows.smart.nodes import (
    make_execution_node,
    make_generation_node,
    make_planner_node,
    make_prompt_node,
    make_tool_selection_node,
)
from app.workflow_engine.mode_dispatcher import ModeDispatcher

# --- 1. Loop Guard & Conditional Edge Tests ---


def test_loop_guard_evaluation():
    """Verify LoopGuard returns 'loop', 'proceed', and caps iterations."""
    # Unsatisfied tasks within limit -> loop
    assert (
        LoopGuard.evaluate(current_iteration=1, max_iterations=5, has_unsatisfied_tasks=True)
        == "loop"
    )

    # All tasks satisfied -> proceed
    assert (
        LoopGuard.evaluate(current_iteration=1, max_iterations=5, has_unsatisfied_tasks=False)
        == "proceed"
    )

    # Unsatisfied tasks but reached iteration cap -> forced proceed
    assert (
        LoopGuard.evaluate(current_iteration=5, max_iterations=5, has_unsatisfied_tasks=True)
        == "proceed"
    )
    assert (
        LoopGuard.evaluate(current_iteration=6, max_iterations=5, has_unsatisfied_tasks=True)
        == "proceed"
    )


def test_route_after_execution():
    """Verify route_after_execution edge condition."""
    state_loop: SmartGraphState = {
        "sub_tasks": [SubTask(id="task_1", description="search")],
        "satisfied_sub_tasks": [],
        "loop_iteration_count": 1,
        "max_iterations": 3,
    }  # type: ignore
    assert route_after_execution(state_loop) == "loop"

    state_done: SmartGraphState = {
        "sub_tasks": [SubTask(id="task_1", description="search")],
        "satisfied_sub_tasks": ["task_1"],
        "loop_iteration_count": 1,
        "max_iterations": 3,
    }  # type: ignore
    assert route_after_execution(state_done) == "proceed"

    state_capped: SmartGraphState = {
        "sub_tasks": [SubTask(id="task_1", description="search")],
        "satisfied_sub_tasks": [],
        "loop_iteration_count": 3,
        "max_iterations": 3,
    }  # type: ignore
    assert route_after_execution(state_capped) == "proceed"


# --- 2. SmartGraph Individual Node Tests ---


@pytest.mark.asyncio
async def test_smart_graph_nodes():
    """Verify planner, tool selection, execution, prompt, and generation nodes."""
    # 1. Planner Node
    planner = make_planner_node()
    state_in: SmartGraphState = {
        "user_message": "What is the latest quantum computing algorithm?",
        "context_bundle": ContextBundle.empty(),
        "conversation_history": [],
    }  # type: ignore
    plan_out = await planner(state_in)
    assert len(plan_out["sub_tasks"]) >= 1
    assert plan_out["loop_iteration_count"] == 0

    # 2. Tool Selection Node
    tool_select = make_tool_selection_node()
    state_select: SmartGraphState = {
        "user_message": "What is the latest quantum computing algorithm?",
        "sub_tasks": plan_out["sub_tasks"],
        "satisfied_sub_tasks": [],
    }  # type: ignore
    select_out = await tool_select(state_select)
    assert select_out["next_tool_call"] is not None
    assert select_out["next_tool_call"].tool_name == "web_search"

    # 3. Execution Node
    mock_dispatcher = AsyncMock(spec=ToolDispatcher)
    mock_dispatcher.dispatch.return_value = [
        ToolResult(
            tool_name="web_search",
            success=True,
            data={
                "results": [
                    {"title": "Quantum Paper", "url": "https://arxiv.org", "snippet": "Finding"}
                ]
            },
        )
    ]
    executor = make_execution_node(tool_dispatcher=mock_dispatcher)
    state_exec: SmartGraphState = {
        "next_tool_call": select_out["next_tool_call"],
        "loop_iteration_count": 0,
    }  # type: ignore
    exec_out = await executor(state_exec)
    assert len(exec_out["tool_results"]) == 1
    assert exec_out["loop_iteration_count"] == 1
    assert "subtask_search_1" in exec_out["satisfied_sub_tasks"]

    # 4. Prompt Node
    prompt_node = make_prompt_node()
    state_prompt: SmartGraphState = {
        "user_message": "What is the latest quantum computing algorithm?",
        "tool_results": exec_out["tool_results"],
    }  # type: ignore
    prompt_out = await prompt_node(state_prompt)
    assert "Quantum Paper" in prompt_out["_intermediate_prompt"]

    # 5. Generation Node
    generation_node = make_generation_node()
    state_gen: SmartGraphState = {
        "user_message": "What is the latest quantum computing algorithm?",
        "tool_results": exec_out["tool_results"],
        "loop_iteration_count": 1,
    }  # type: ignore
    gen_out = await generation_node(state_gen)
    assert "Smart reasoning completed" in gen_out["draft_response"]


# --- 3. End-to-End SmartGraph Invocation & Compilation Tests ---


@pytest.mark.asyncio
async def test_smart_graph_builder_and_ainvoke():
    """Verify SmartGraph compiles and successfully executes end-to-end via ainvoke."""
    mock_dispatcher = AsyncMock(spec=ToolDispatcher)
    mock_dispatcher.dispatch.return_value = [
        ToolResult(
            tool_name="web_search",
            success=True,
            data={
                "results": [
                    {"title": "Doc", "url": "https://example.com", "snippet": "Sample snippet"}
                ]
            },
        )
    ]

    builder = SmartGraphBuilder(tool_dispatcher=mock_dispatcher, max_iterations=3)
    graph = builder.build()

    initial_state: SmartGraphState = {
        "conversation_id": "conv_123",
        "user_id": "user_456",
        "request_id": "req_789",
        "mode": "smart",
        "user_message": "Explain quantum computing algorithms in depth",
        "context_bundle": ContextBundle.empty(),
        "conversation_history": [],
        "sub_tasks": [],
        "satisfied_sub_tasks": [],
        "tool_results": [],
        "next_tool_call": None,
        "loop_iteration_count": 0,
        "max_iterations": 3,
        "draft_response": None,
    }

    final_state = await graph.ainvoke(initial_state)

    assert final_state["draft_response"] is not None
    assert final_state["loop_iteration_count"] >= 1
    assert len(final_state["tool_results"]) >= 1


# --- 4. ModeDispatcher Routing to SmartGraph ---


@pytest.mark.asyncio
async def test_mode_dispatcher_routes_to_smart_graph():
    """Verify ModeDispatcher invokes compiled SmartGraph when mode='smart'."""
    mock_dispatcher = AsyncMock(spec=ToolDispatcher)
    mock_dispatcher.dispatch.return_value = [
        ToolResult(
            tool_name="web_search",
            success=True,
            data={
                "results": [
                    {"title": "Knowledge", "url": "https://wikidata.org", "snippet": "Fact"}
                ]
            },
        )
    ]
    smart_graph = SmartGraphBuilder(tool_dispatcher=mock_dispatcher, max_iterations=2).build()

    dispatcher = ModeDispatcher(
        handlers={},
        graphs={"smart": smart_graph},
    )

    ctx = PipelineContext(
        conversation_id="conv_smart",
        user_id="user_1",
        message_id="msg_1",
        request_id="req_1",
        user_message="Find research papers on neural networks",
        mode_hint=UserMode.SMART,
        context_bundle=ContextBundle.empty(),
    )
    plan = ExecutionPlan(
        mode=UserMode.SMART,
        tools=[],
        max_iterations=2,
    )

    wf_result = await dispatcher.dispatch(plan, ctx)

    assert wf_result.mode == "smart"
    assert wf_result.engine_type == "langgraph"
    assert wf_result.draft_content is not None
    assert len(wf_result.tool_outputs) >= 1
