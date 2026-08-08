"""
Unit tests for all 5 deterministic Mode Handlers (Default, Tutor, Code, AskFiles, WebSearch).
Implements Phase 8 deliverables verification.
"""

from unittest.mock import AsyncMock

import pytest

from app.context.schemas import (
    ContextBundle,
    DocumentChunk,
    MemoryContext,
    Message,
    RetrievalContext,
    Role,
)
from app.models.execution_plan import ExecutionPlan, ToolCall, UserMode
from app.models.pipeline_context import PipelineContext
from app.models.tool import ToolResult
from app.workflow_engine.mode_handlers import (
    AskFilesHandler,
    CodeHandler,
    DefaultHandler,
    TutorHandler,
    WebSearchHandler,
)


@pytest.fixture
def sample_context() -> PipelineContext:
    bundle = ContextBundle(
        memory=MemoryContext(
            short_term_messages=[
                Message(role=Role.USER, content="Earlier query"),
                Message(role=Role.ASSISTANT, content="Earlier answer"),
            ]
        ),
        retrieval=RetrievalContext(
            chunks=[DocumentChunk(chunk_id="c1", file_id="f1", content="doc snippet", score=0.9)]
        ),
    )
    return PipelineContext(
        conversation_id="conv_100",
        user_id="user_100",
        message_id="msg_100",
        request_id="req_100",
        user_message="Hello mode test",
        file_ids=["file_1"],
        context_bundle=bundle,
    )


@pytest.mark.asyncio
async def test_default_handler(sample_context: PipelineContext):
    """Verify DefaultHandler handles simple queries and optional tool dispatches."""
    mock_dispatcher = AsyncMock()
    mock_dispatcher.dispatch.return_value = [
        ToolResult(tool_name="web_search", success=True, data={"res": []})
    ]

    handler = DefaultHandler(tool_dispatcher=mock_dispatcher)

    # 1. Without tools
    plan_no_tools = ExecutionPlan(mode=UserMode.DEFAULT, tools=[])
    out1 = await handler.handle(plan_no_tools, sample_context)
    assert out1.mode == "default"
    assert len(out1.tool_outputs) == 0
    assert len(out1.conversation_history) == 2
    assert out1.user_message == "Hello mode test"
    mock_dispatcher.dispatch.assert_not_called()

    # 2. With optional tool
    plan_with_tools = ExecutionPlan(
        mode=UserMode.DEFAULT,
        tools=[ToolCall(tool_name="web_search", params={"query": "test"})],
    )
    out2 = await handler.handle(plan_with_tools, sample_context)
    assert out2.mode == "default"
    assert len(out2.tool_outputs) == 1
    mock_dispatcher.dispatch.assert_called_once()


@pytest.mark.asyncio
async def test_tutor_handler(sample_context: PipelineContext):
    """Verify TutorHandler formats pedagogical context."""
    handler = TutorHandler()
    plan = ExecutionPlan(mode=UserMode.TUTOR)
    out = await handler.handle(plan, sample_context)

    assert out.mode == "tutor"
    assert len(out.tool_outputs) == 0
    assert len(out.conversation_history) == 2
    assert out.user_message == "Hello mode test"


@pytest.mark.asyncio
async def test_code_handler(sample_context: PipelineContext):
    """Verify CodeHandler handles code requests."""
    mock_dispatcher = AsyncMock()
    mock_dispatcher.dispatch.return_value = [
        ToolResult(tool_name="web_search", success=True, data={"docs": "python syntax"})
    ]

    handler = CodeHandler(tool_dispatcher=mock_dispatcher)
    plan = ExecutionPlan(
        mode=UserMode.CODE,
        tools=[ToolCall(tool_name="web_search", params={"query": "python"})],
    )
    out = await handler.handle(plan, sample_context)

    assert out.mode == "code"
    assert len(out.tool_outputs) == 1
    assert out.tool_outputs[0].data == {"docs": "python syntax"}


@pytest.mark.asyncio
async def test_ask_files_handler(sample_context: PipelineContext):
    """Verify AskFilesHandler operates without tool_dispatcher and accesses retrieval context."""
    handler = AskFilesHandler()
    plan = ExecutionPlan(mode=UserMode.ASK_FILES)

    # 1. With files attached
    out = await handler.handle(plan, sample_context)
    assert out.mode == "ask_files"
    assert len(out.tool_outputs) == 0
    assert len(out.conversation_history) == 2

    # 2. Without files attached (logs warning, succeeds)
    ctx_no_files = PipelineContext(
        conversation_id="conv_101",
        user_id="user_101",
        message_id="msg_101",
        request_id="req_101",
        user_message="Summarize doc",
        file_ids=[],
        context_bundle=ContextBundle(),
    )
    out_no_files = await handler.handle(plan, ctx_no_files)
    assert out_no_files.mode == "ask_files"
    assert len(out_no_files.conversation_history) == 0


@pytest.mark.asyncio
async def test_web_search_handler_defensive_injection(sample_context: PipelineContext):
    """Verify WebSearchHandler defensively injects web_search tool if missing in plan."""
    mock_dispatcher = AsyncMock()
    mock_dispatcher.dispatch.return_value = [
        ToolResult(tool_name="web_search", success=True, data={"query": "weather"})
    ]

    handler = WebSearchHandler(tool_dispatcher=mock_dispatcher)

    # Plan with empty tools
    plan_empty_tools = ExecutionPlan(mode=UserMode.WEB_SEARCH, tools=[])
    out = await handler.handle(plan_empty_tools, sample_context)

    assert out.mode == "web_search"
    assert len(out.tool_outputs) == 1
    mock_dispatcher.dispatch.assert_called_once()
    calls = mock_dispatcher.dispatch.call_args[0][0]
    assert len(calls) == 1
    assert calls[0].tool_name == "web_search"
    assert calls[0].params == {"query": "Hello mode test"}
    assert calls[0].required is True
