"""
Unit tests for SmartGraph LangGraph Orchestration (Phase 11).
Tests State, Nodes, Edges, LoopGuard, Builder Compilation, and Dispatcher Routing.
"""

from unittest.mock import AsyncMock

import pytest

from app.context.schemas import ContextBundle
from app.models.execution_plan import ExecutionPlan, ToolCall, UserMode
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
        # the LLM Request Analyzer's decision: which tool to call, with what arguments
        "planned_tools": [ToolCall(tool_name="web_search", params={"query": "latest quantum computing algorithm"})],
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
    assert "subtask_tool_1" in exec_out["satisfied_sub_tasks"]

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
    assert "Analysis & Synthesis" in gen_out["draft_response"]
    assert "Executive Summary" in gen_out["draft_response"]


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
        "planned_tools": [ToolCall(tool_name="web_search", params={"query": "quantum computing algorithms"})],
        "plan_is_fallback": False,
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
        tools=[ToolCall(tool_name="web_search", params={"query": "neural network research papers"})],
        max_iterations=2,
    )

    wf_result = await dispatcher.dispatch(plan, ctx)

    assert wf_result.mode == "smart"
    assert wf_result.engine_type == "langgraph"
    assert wf_result.draft_content is not None
    assert len(wf_result.tool_outputs) >= 1


def test_smart_generation_prompt_requires_source_links_and_limits_tables():
    import yaml
    from pathlib import Path
    d = yaml.safe_load((Path(__file__).resolve().parents[2] / "app/prompts/smart/generation_v1.yaml").read_text(encoding="utf-8"))
    text = d["system"]
    assert "[Title](URL)" in text and '"Sources" list' in text
    assert "never put a single answer or an explanation in a table" in text
    assert "Never mention which search engine" in text
    assert d["version"] == "1.2"


# --- 5. Who decides which tool to call: the LLM plan first, the rules only as fallback ---------------------------------


async def _plan(message, planned=None, fallback=False):
    state = {"user_message": message, "context_bundle": ContextBundle.empty(), "conversation_history": [],
             "planned_tools": planned or [], "plan_is_fallback": fallback}
    return (await make_planner_node()(state))["sub_tasks"]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_llm_plan_with_no_tools_means_no_search_even_for_a_long_how_question():
    # the real case: "embedding model kya hota hai" used to trigger a web search through keyword matching
    assert await _plan("Embedding model kya hota hai, how does it work and why is it used?") == []


@pytest.mark.asyncio
async def test_llm_planned_tools_and_their_arguments_are_used_as_decided():
    calls = [ToolCall(tool_name="web_search", params={"query": "gpt-6 astra release"}),
             ToolCall(tool_name="web_search", params={"query": "gpt-6 astra paper arxiv"}, required=True)]
    tasks = await _plan("tell me about GPT-6 Astra", planned=calls)
    assert [t.params["query"] for t in tasks] == ["gpt-6 astra release", "gpt-6 astra paper arxiv"]
    assert [t.tool_name for t in tasks] == ["web_search", "web_search"] and tasks[1].required


@pytest.mark.asyncio
async def test_planned_tools_are_capped():
    from app.workflow_engine.langgraph_workflows.smart.nodes import MAX_PLANNED_TOOLS
    many = [ToolCall(tool_name="web_search", params={"query": f"q{i}"}) for i in range(10)]
    assert len(await _plan("x", planned=many)) == MAX_PLANNED_TOOLS


@pytest.mark.asyncio
async def test_rules_decide_only_when_the_llm_planner_failed():
    assert await _plan("What is a Python dictionary?", fallback=True) == []                       # stable: still no search
    fresh = await _plan("What changed in the latest Python release?", fallback=True)
    assert [t.tool_name for t in fresh] == ["web_search"]


@pytest.mark.asyncio
async def test_an_explicit_request_to_search_is_honoured_even_if_the_llm_planned_nothing():
    tasks = await _plan("search for the Raft consensus paper and give me links")
    assert [t.tool_name for t in tasks] == ["web_search"]


@pytest.mark.asyncio
async def test_bookkeeping_id_is_not_sent_to_the_tool_and_planner_arguments_are_kept():
    mock = AsyncMock(spec=ToolDispatcher)
    mock.dispatch.return_value = [ToolResult(tool_name="web_search", success=True, data={"results": []})]
    task = SubTask(id="subtask_tool_1", description="d", tool_name="web_search", params={"query": "exact planner query", "max_results": 3})
    select = await make_tool_selection_node()({"sub_tasks": [task], "satisfied_sub_tasks": [], "user_message": "m"})  # type: ignore[arg-type]
    call = select["next_tool_call"]
    assert call.params["query"] == "exact planner query" and call.params["max_results"] == 3
    out = await make_execution_node(mock)({"next_tool_call": call, "loop_iteration_count": 0})  # type: ignore[arg-type]
    sent = mock.dispatch.call_args.args[0][0]
    assert "sub_task_id" not in sent.params and sent.params["query"] == "exact planner query"
    assert out["satisfied_sub_tasks"] == ["subtask_tool_1"]


def test_dispatcher_passes_the_llm_tool_plan_and_flags_a_fallback_plan():
    from app.request_analyzer.safe_default import SafeDefaultFactory
    d = ModeDispatcher(handlers={}, graphs={})
    ctx = PipelineContext(conversation_id="c", user_id="u", message_id="m", request_id="r", user_message="hi",
                          mode_hint=UserMode.SMART, context_bundle=ContextBundle.empty())
    llm_plan = ExecutionPlan(mode=UserMode.SMART, tools=[ToolCall(tool_name="web_search", params={"query": "q"})])
    state = d._build_initial_graph_state(llm_plan, ctx)
    assert [t.tool_name for t in state["planned_tools"]] == ["web_search"] and state["plan_is_fallback"] is False
    safe = SafeDefaultFactory.create(latency_ms=0.0, reason="x")
    assert d._build_initial_graph_state(safe, ctx)["plan_is_fallback"] is True
