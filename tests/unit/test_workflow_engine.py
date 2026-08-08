"""
Unit tests for WorkflowEngine, ModeDispatcher, and WorkflowResult Normalization.
Implements Phase 7 deliverables verification.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.context.schemas import ContextBundle, Message, Role
from app.exceptions.analysis import UnknownModeError
from app.models.execution_plan import ExecutionPlan, UserMode
from app.models.pipeline_context import PipelineContext
from app.models.tool import ToolResult
from app.workflow_engine.engine import WorkflowEngine
from app.workflow_engine.mode_dispatcher import (
    LANGGRAPH_KEYS,
    MODE_HANDLER_KEYS,
    ModeDispatcher,
)
from app.workflow_engine.workflow_result import (
    ModeHandlerOutput,
    WorkflowResult,
)


def test_mode_disjoint_invariants():
    """Verify Mode Handler and LangGraph mode key sets are strictly disjoint and cover all UserModes."""
    assert MODE_HANDLER_KEYS.isdisjoint(LANGGRAPH_KEYS)
    assert (MODE_HANDLER_KEYS | LANGGRAPH_KEYS) == {m.value for m in UserMode}


def test_mode_dispatcher_overlap_rejection():
    """Verify ModeDispatcher raises ValueError if same mode is passed to handlers and graphs."""
    with pytest.raises(ValueError, match="Modes registered in both handlers and graphs"):
        ModeDispatcher(
            handlers={"smart": MagicMock()},
            graphs={"smart": MagicMock()},
        )


def test_engine_type_resolution():
    """Verify engine_type_for correctly classifies all valid and invalid modes."""
    dispatcher = ModeDispatcher(
        handlers={"default": MagicMock(), "tutor": MagicMock()},
        graphs={"smart": MagicMock(), "deep_research": MagicMock()},
    )
    # Mode handler modes
    assert dispatcher.engine_type_for("default") == "mode_handler"
    assert dispatcher.engine_type_for("tutor") == "mode_handler"
    assert dispatcher.engine_type_for("code") == "mode_handler"
    assert dispatcher.engine_type_for("ask_files") == "mode_handler"
    assert dispatcher.engine_type_for("web_search") == "mode_handler"

    # LangGraph modes
    assert dispatcher.engine_type_for("smart") == "langgraph"
    assert dispatcher.engine_type_for("deep_research") == "langgraph"

    # Unknown mode
    with pytest.raises(UnknownModeError, match="Unknown or unsupported mode: 'unsupported_mode'"):
        dispatcher.engine_type_for("unsupported_mode")


@pytest.mark.asyncio
async def test_mode_dispatcher_routes_to_mode_handler():
    """Verify ModeDispatcher routes deterministic modes to registered ModeHandler."""
    mock_handler = AsyncMock()
    mock_handler.handle.return_value = ModeHandlerOutput(
        mode="tutor",
        tool_outputs=[ToolResult(tool_name="web_search", success=True, data={"res": []})],
        conversation_history=[Message(role=Role.USER, content="Teach me calculus")],
        user_message="Teach me calculus",
    )

    dispatcher = ModeDispatcher(handlers={"tutor": mock_handler})

    plan = ExecutionPlan(mode=UserMode.TUTOR)
    ctx = PipelineContext(
        conversation_id="conv_001",
        user_id="user_001",
        message_id="msg_001",
        request_id="req_001",
        user_message="Teach me calculus",
        context_bundle=ContextBundle(),
    )

    result: WorkflowResult = await dispatcher.dispatch(plan, ctx)

    # Handler must be called with plan and ctx
    mock_handler.handle.assert_called_once_with(plan, ctx)

    # Result must be normalized
    assert result.mode == "tutor"
    assert result.engine_type == "mode_handler"
    assert result.draft_content is None
    assert len(result.tool_outputs) == 1
    assert result.user_message == "Teach me calculus"
    assert len(result.conversation_history) == 1

    # PipelineContext must be updated
    assert ctx.engine_type == "mode_handler"
    assert ctx.plan == plan


@pytest.mark.asyncio
async def test_mode_dispatcher_routes_to_langgraph():
    """Verify ModeDispatcher routes agentic loop modes to registered LangGraph graph."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "mode": "smart",
        "user_message": "Solve this multi-step problem",
        "draft_response": "Draft synthesized solution",
        "tool_results": [ToolResult(tool_name="web_search", success=True, data={"data": "info"})],
        "conversation_history": [Message(role=Role.USER, content="Solve this")],
        "loop_iteration_count": 3,
    }

    dispatcher = ModeDispatcher(graphs={"smart": mock_graph})

    plan = ExecutionPlan(mode=UserMode.SMART, max_iterations=4)
    ctx = PipelineContext(
        conversation_id="conv_002",
        user_id="user_002",
        message_id="msg_002",
        request_id="req_002",
        user_message="Solve this multi-step problem",
        context_bundle=ContextBundle(),
    )

    result: WorkflowResult = await dispatcher.dispatch(plan, ctx)

    mock_graph.ainvoke.assert_called_once()
    assert result.mode == "smart"
    assert result.engine_type == "langgraph"
    assert result.draft_content == "Draft synthesized solution"
    assert len(result.tool_outputs) == 1
    assert result.metadata["loop_iterations"] == 3
    assert ctx.engine_type == "langgraph"


@pytest.mark.asyncio
async def test_workflow_engine_coordination():
    """Verify WorkflowEngine coordinates execution through ModeDispatcher."""
    mock_dispatcher = AsyncMock()
    expected_result = WorkflowResult(
        mode="code",
        engine_type="mode_handler",
        draft_content=None,
        tool_outputs=[],
        conversation_history=[],
        user_message="Write a python function",
    )
    mock_dispatcher.dispatch.return_value = expected_result

    engine = WorkflowEngine(mode_dispatcher=mock_dispatcher)

    plan = ExecutionPlan(mode=UserMode.CODE)
    ctx = PipelineContext(
        conversation_id="conv_003",
        user_id="user_003",
        message_id="msg_003",
        request_id="req_003",
        user_message="Write a python function",
        context_bundle=ContextBundle(),
    )

    result = await engine.execute(plan, ctx)

    mock_dispatcher.dispatch.assert_called_once_with(plan, ctx)
    assert result == expected_result
    assert result.mode == "code"


@pytest.mark.asyncio
async def test_mode_dispatcher_unregistered_mode_raises():
    """Verify ModeDispatcher raises UnknownModeError when no handler/graph is configured for mode."""
    dispatcher = ModeDispatcher(handlers={}, graphs={})

    plan = ExecutionPlan(mode=UserMode.CODE)
    ctx = PipelineContext(
        conversation_id="conv_004",
        user_id="user_004",
        message_id="msg_004",
        request_id="req_004",
        user_message="test",
    )

    with pytest.raises(UnknownModeError, match="Unknown or unsupported mode: 'code'"):
        await dispatcher.dispatch(plan, ctx)
