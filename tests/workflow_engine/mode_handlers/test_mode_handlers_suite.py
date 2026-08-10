"""
Workflow Engine Mode Handler Suite (Phase 20).
Extended coverage for all 5 Mode Handlers using the actual API.
Tests ModeHandlerOutput shape, tool dispatching, history, and defensive fallbacks.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.models.execution_plan import (
    ExecutionPlan,
    IntentCategory,
    ReasoningMode,
    Skill,
    ToolCall,
    UserMode,
)
from app.models.pipeline_context import PipelineContext
from app.models.tool import ToolResult
from app.context.schemas import ContextBundle, MemoryContext, Message
from app.workflow_engine.mode_handlers import (
    AskFilesHandler,
    CodeHandler,
    DefaultHandler,
    TutorHandler,
    WebSearchHandler,
)
from app.workflow_engine.workflow_result import ModeHandlerOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(
    mode: UserMode = UserMode.DEFAULT,
    skill: Skill = Skill.GENERAL_CHAT,
    tools: list[str] | None = None,
) -> ExecutionPlan:
    return ExecutionPlan(
        intent=IntentCategory.QUESTION_ANSWERING,
        mode=mode,
        skill=skill,
        reasoning=ReasoningMode.CHAIN_OF_THOUGHT,
        tools=[ToolCall(tool_name=t) for t in (tools or [])],
        max_iterations=1,
        suggested_temperature=0.7,
        analysis_confidence=0.9,
        groq_model_used="llama-3.3-70b-versatile",
    )


def _make_ctx(
    message: str = "Explain attention mechanisms",
    file_ids: list[str] | None = None,
    history: list[Message] | None = None,
) -> PipelineContext:
    msgs = history or []
    return PipelineContext(
        conversation_id="conv_mode_001",
        user_id="user_mode_001",
        message_id="msg_mode_001",
        request_id="req_mode_001",
        user_message=message,
        file_ids=file_ids or [],
        context_bundle=ContextBundle(
            conversation_history=msgs,
            graph_context=None,
            retrieved_chunks=[],
            sources=[],
        ),
    )


def _make_tool_dispatcher(tool_results: list[ToolResult] | None = None):
    dispatcher = AsyncMock()
    dispatcher.dispatch = AsyncMock(return_value=tool_results or [])
    return dispatcher


# ---------------------------------------------------------------------------
# DefaultHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_handler_no_tools_produces_output():
    """DefaultHandler with no tools returns a valid ModeHandlerOutput."""
    handler = DefaultHandler()
    ctx = _make_ctx(message="What is Python?")
    plan = _make_plan(mode=UserMode.DEFAULT, skill=Skill.GENERAL_CHAT)

    result = await handler.handle(plan=plan, ctx=ctx)
    assert isinstance(result, ModeHandlerOutput)
    assert result.mode == "default"
    assert result.user_message == "What is Python?"
    assert result.tool_outputs == []


@pytest.mark.asyncio
async def test_default_handler_dispatches_tools_when_present():
    """DefaultHandler calls tool_dispatcher.dispatch() when plan has tools."""
    tool_result = ToolResult(tool_name="web_search", success=True, content="Search result", raw={})
    dispatcher = _make_tool_dispatcher([tool_result])

    handler = DefaultHandler(tool_dispatcher=dispatcher)
    ctx = _make_ctx(message="Latest AI news")
    plan = _make_plan(mode=UserMode.DEFAULT, tools=["web_search"])

    result = await handler.handle(plan=plan, ctx=ctx)
    dispatcher.dispatch.assert_called_once()
    assert len(result.tool_outputs) == 1
    assert result.tool_outputs[0].tool_name == "web_search"


@pytest.mark.asyncio
async def test_default_handler_includes_conversation_history():
    """DefaultHandler includes memory short_term_messages in output."""
    history = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]
    ctx = _make_ctx(message="Continue")
    ctx.context_bundle.memory = MemoryContext(short_term_messages=history, user_profile=None)

    handler = DefaultHandler()
    plan = _make_plan(mode=UserMode.DEFAULT)
    result = await handler.handle(plan=plan, ctx=ctx)

    assert len(result.conversation_history) == 2
    assert result.conversation_history[0].content == "Hello"


@pytest.mark.asyncio
async def test_default_handler_empty_context_bundle():
    """DefaultHandler handles None context_bundle without crashing."""
    handler = DefaultHandler()
    ctx = _make_ctx(message="Simple question")
    ctx.context_bundle = None  # No context at all

    plan = _make_plan(mode=UserMode.DEFAULT)
    result = await handler.handle(plan=plan, ctx=ctx)

    assert isinstance(result, ModeHandlerOutput)
    assert result.conversation_history == []


# ---------------------------------------------------------------------------
# TutorHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tutor_handler_returns_mode_tutor():
    """TutorHandler output has mode='tutor'."""
    handler = TutorHandler()
    ctx = _make_ctx(message="Explain gradient descent step by step")
    plan = _make_plan(mode=UserMode.TUTOR, skill=Skill.TUTOR)

    result = await handler.handle(plan=plan, ctx=ctx)
    assert isinstance(result, ModeHandlerOutput)
    assert result.mode == "tutor"


@pytest.mark.asyncio
async def test_tutor_handler_passes_user_message():
    """TutorHandler preserves user_message in output."""
    handler = TutorHandler()
    ctx = _make_ctx(message="Teach me backpropagation")
    plan = _make_plan(mode=UserMode.TUTOR, skill=Skill.TUTOR)

    result = await handler.handle(plan=plan, ctx=ctx)
    assert result.user_message == "Teach me backpropagation"


# ---------------------------------------------------------------------------
# CodeHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_handler_returns_mode_code():
    """CodeHandler output has mode='code'."""
    handler = CodeHandler()
    ctx = _make_ctx(message="Write a binary search in Python")
    plan = _make_plan(mode=UserMode.CODE, skill=Skill.CODING)

    result = await handler.handle(plan=plan, ctx=ctx)
    assert isinstance(result, ModeHandlerOutput)
    assert result.mode == "code"


@pytest.mark.asyncio
async def test_code_handler_handles_debug_requests():
    """CodeHandler works for debugging intent requests."""
    handler = CodeHandler()
    ctx = _make_ctx(message="Why does this Python code raise IndexError?")
    plan = _make_plan(mode=UserMode.CODE, skill=Skill.CODING)

    result = await handler.handle(plan=plan, ctx=ctx)
    assert result is not None
    assert result.mode == "code"


# ---------------------------------------------------------------------------
# AskFilesHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_files_handler_with_file_ids_returns_mode():
    """AskFilesHandler returns mode='ask_files' when file_ids present."""
    handler = AskFilesHandler()
    ctx = _make_ctx(message="Summarize this document", file_ids=["file_aaa", "file_bbb"])
    plan = _make_plan(mode=UserMode.ASK_FILES, skill=Skill.RESEARCH)

    result = await handler.handle(plan=plan, ctx=ctx)
    assert isinstance(result, ModeHandlerOutput)
    assert result.mode == "ask_files"


@pytest.mark.asyncio
async def test_ask_files_handler_with_no_files_degrades_gracefully():
    """AskFilesHandler handles missing file_ids without crashing."""
    handler = AskFilesHandler()
    ctx = _make_ctx(message="Summarize the document", file_ids=[])
    plan = _make_plan(mode=UserMode.ASK_FILES, skill=Skill.RESEARCH)

    result = await handler.handle(plan=plan, ctx=ctx)
    assert result is not None
    assert result.mode == "ask_files"


# ---------------------------------------------------------------------------
# WebSearchHandler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_handler_with_tool_dispatcher():
    """WebSearchHandler calls tool dispatcher for web_search tool."""
    web_result = ToolResult(
        tool_name="web_search",
        success=True,
        content="LLMs are large neural networks...",
        raw=[{"url": "https://example.com", "title": "LLMs", "snippet": "..."}],
    )
    dispatcher = _make_tool_dispatcher([web_result])
    handler = WebSearchHandler(tool_dispatcher=dispatcher)
    ctx = _make_ctx(message="What is a large language model?")
    plan = _make_plan(mode=UserMode.WEB_SEARCH, skill=Skill.RESEARCH, tools=["web_search"])

    result = await handler.handle(plan=plan, ctx=ctx)
    assert isinstance(result, ModeHandlerOutput)
    assert result.mode == "web_search"
    dispatcher.dispatch.assert_called_once()


@pytest.mark.asyncio
async def test_web_search_handler_injects_web_search_if_missing():
    """WebSearchHandler defensively injects web_search tool even if not in plan."""
    dispatcher = _make_tool_dispatcher([])
    handler = WebSearchHandler(tool_dispatcher=dispatcher)
    ctx = _make_ctx(message="Latest news on AI")
    # Plan has NO tools
    plan = _make_plan(mode=UserMode.WEB_SEARCH, skill=Skill.RESEARCH, tools=[])

    result = await handler.handle(plan=plan, ctx=ctx)
    # Dispatcher should still be called with the injected tool
    dispatcher.dispatch.assert_called_once()
    call_args = dispatcher.dispatch.call_args[0][0]  # First positional arg = tools list
    tool_names = [t.tool_name for t in call_args]
    assert "web_search" in tool_names


@pytest.mark.asyncio
async def test_web_search_handler_no_dispatcher_returns_output():
    """WebSearchHandler without a dispatcher still returns output (no crash)."""
    handler = WebSearchHandler()  # No dispatcher
    ctx = _make_ctx(message="Latest news on AI")
    plan = _make_plan(mode=UserMode.WEB_SEARCH, skill=Skill.RESEARCH, tools=["web_search"])

    result = await handler.handle(plan=plan, ctx=ctx)
    assert result is not None
    assert result.mode == "web_search"
    assert result.tool_outputs == []
