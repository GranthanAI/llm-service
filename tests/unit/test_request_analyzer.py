"""
Unit tests for RequestAnalyzer, PromptBuilder, GroqClient, CircuitBreaker, and SafeDefaultPlan.
Implements Phase 6 deliverables verification.
"""

import json
from unittest.mock import AsyncMock

import pytest

from app.config.settings import LLMServiceConfig
from app.context.schemas import (
    ContextBundle,
    DocumentChunk,
    EntityNode,
    Fact,
    GraphContext,
    MemoryContext,
    Message,
    RetrievalContext,
    Role,
)
from app.exceptions.provider import CircuitOpenError, ProviderError
from app.models.execution_plan import (
    ExecutionPlan,
    IntentCategory,
    ReasoningMode,
    Skill,
    UserMode,
)
from app.models.pipeline_context import PipelineContext
from app.models.provider import CircuitState
from app.request_analyzer.analyzer import RequestAnalyzer
from app.request_analyzer.prompt_template import AnalysisPromptBuilder
from app.request_analyzer.schemas import SafeDefaultPlan
from app.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


@pytest.fixture
def sample_pipeline_ctx() -> PipelineContext:
    bundle = ContextBundle(
        memory=MemoryContext(
            short_term_messages=[Message(role=Role.USER, content="Hello there")],
            long_term_facts=[Fact(fact_id="f1", statement="User writes Python")],
        ),
        graph=GraphContext(
            entities=[EntityNode(id="e1", name="FastAPI", type="Framework")],
            subgraph_summary="FastAPI is a modern web framework.",
        ),
        retrieval=RetrievalContext(
            chunks=[
                DocumentChunk(chunk_id="c1", file_id="f1", content="FastAPI doc snippet", score=0.9)
            ],
        ),
    )
    return PipelineContext(
        conversation_id="conv_1",
        user_id="user_1",
        message_id="msg_1",
        request_id="req_1",
        user_message="Build an async endpoint in FastAPI",
        mode_hint=UserMode.CODE,
        file_ids=[],
        context_bundle=bundle,
    )


def test_analysis_prompt_builder(sample_pipeline_ctx: PipelineContext):
    """Verify prompt builder formats all sections properly."""
    builder = AnalysisPromptBuilder()
    messages = builder.build_prompt(
        sample_pipeline_ctx,
        tools_schema=[{"name": "web_search", "description": "Search web"}],
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Build an async endpoint in FastAPI"

    sys_text = messages[0]["content"]
    assert "User writes Python" in sys_text
    assert "FastAPI is a modern web framework." in sys_text
    assert "FastAPI doc snippet" in sys_text
    assert "web_search" in sys_text
    assert "code" in sys_text  # mode hint


@pytest.mark.asyncio
async def test_request_analyzer_success(sample_pipeline_ctx: PipelineContext):
    """Verify successful Groq analysis and ExecutionPlan construction."""
    mock_groq = AsyncMock()
    valid_plan_json = json.dumps(
        {
            "intent": "CODE_GENERATION",
            "mode": "code",
            "skill": "coding",
            "reasoning": "CHAIN_OF_THOUGHT",
            "tools": [],
            "max_iterations": 1,
            "suggested_temperature": 0.2,
            "analysis_confidence": 0.95,
        }
    )
    mock_groq.complete.return_value = valid_plan_json
    mock_groq.model = "llama-3.3-70b-versatile"

    analyzer = RequestAnalyzer(groq_client=mock_groq)
    plan = await analyzer.analyze(sample_pipeline_ctx)

    assert isinstance(plan, ExecutionPlan)
    assert plan.intent == IntentCategory.CODE_GENERATION
    assert plan.mode == UserMode.CODE
    assert plan.skill == Skill.CODING
    assert plan.reasoning == ReasoningMode.CHAIN_OF_THOUGHT
    assert plan.suggested_temperature == 0.2
    assert plan.analysis_confidence == 0.95
    assert plan.groq_model_used == "llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_safety_override_ask_files_without_files(sample_pipeline_ctx: PipelineContext):
    """Safety Rule 1: ask_files without file_ids reverts to default mode."""
    mock_groq = AsyncMock()
    plan_json = json.dumps(
        {
            "intent": "DOCUMENT_ANALYSIS",
            "mode": "ask_files",
            "skill": "research",
            "reasoning": "DIRECT",
            "tools": [],
            "max_iterations": 1,
        }
    )
    mock_groq.complete.return_value = plan_json
    mock_groq.model = "llama-3.3-70b-versatile"

    # sample_pipeline_ctx has file_ids = []
    analyzer = RequestAnalyzer(groq_client=mock_groq)
    plan = await analyzer.analyze(sample_pipeline_ctx)

    assert plan.mode == UserMode.DEFAULT  # Reverted to default


@pytest.mark.asyncio
async def test_safety_override_disabled_web_search(sample_pipeline_ctx: PipelineContext):
    """Safety Rule 2: web_search tool stripped when enable_web_search is False."""
    mock_groq = AsyncMock()
    plan_json = json.dumps(
        {
            "intent": "WEB_SEARCH",
            "mode": "web_search",
            "skill": "research",
            "reasoning": "DIRECT",
            "tools": [{"tool_name": "web_search", "params": {"query": "latest news"}}],
            "max_iterations": 1,
        }
    )
    mock_groq.complete.return_value = plan_json
    mock_groq.model = "llama-3.3-70b-versatile"

    config = LLMServiceConfig(enable_web_search=False)
    analyzer = RequestAnalyzer(groq_client=mock_groq, config=config)
    plan = await analyzer.analyze(sample_pipeline_ctx)

    assert len(plan.tools) == 0  # web_search stripped


@pytest.mark.asyncio
async def test_safety_override_iteration_capping(sample_pipeline_ctx: PipelineContext):
    """Safety Rules 3, 4, 5: Iteration caps per mode."""
    mock_groq = AsyncMock()
    mock_groq.model = "llama-3.3-70b-versatile"

    config = LLMServiceConfig(
        langgraph_max_loop_iterations_smart=4,
        langgraph_max_loop_iterations_deep_research=8,
    )
    analyzer = RequestAnalyzer(groq_client=mock_groq, config=config)

    # 1. Smart mode capped at 4
    mock_groq.complete.return_value = json.dumps({"mode": "smart", "max_iterations": 10})
    plan_smart = await analyzer.analyze(sample_pipeline_ctx)
    assert plan_smart.max_iterations == 4

    # 2. Deep research mode capped at 8
    mock_groq.complete.return_value = json.dumps({"mode": "deep_research", "max_iterations": 20})
    plan_deep = await analyzer.analyze(sample_pipeline_ctx)
    assert plan_deep.max_iterations == 8

    # 3. Deterministic mode (e.g. tutor) forced to 1
    mock_groq.complete.return_value = json.dumps({"mode": "tutor", "max_iterations": 5})
    plan_tutor = await analyzer.analyze(sample_pipeline_ctx)
    assert plan_tutor.max_iterations == 1


@pytest.mark.asyncio
async def test_request_analyzer_fallback_on_json_parse_error(sample_pipeline_ctx: PipelineContext):
    """Verify fallback to SafeDefaultPlan on invalid JSON."""
    mock_groq = AsyncMock()
    mock_groq.complete.return_value = "NOT_VALID_JSON_AT_ALL{{{"
    mock_groq.model = "llama-3.3-70b-versatile"

    analyzer = RequestAnalyzer(groq_client=mock_groq)
    plan = await analyzer.analyze(sample_pipeline_ctx)

    assert isinstance(plan, SafeDefaultPlan)
    assert plan.mode == UserMode.DEFAULT
    assert plan.skill == Skill.GENERAL_CHAT
    assert plan.analysis_confidence == 0.0
    assert plan.groq_model_used == "fallback_safe_default"


@pytest.mark.asyncio
async def test_request_analyzer_fallback_on_groq_outage(sample_pipeline_ctx: PipelineContext):
    """Verify fallback to SafeDefaultPlan on Groq network/API failure."""
    mock_groq = AsyncMock()
    mock_groq.complete.side_effect = ProviderError("Groq service outage", provider="groq")

    analyzer = RequestAnalyzer(groq_client=mock_groq)
    plan = await analyzer.analyze(sample_pipeline_ctx)

    assert isinstance(plan, SafeDefaultPlan)
    assert plan.mode == UserMode.DEFAULT
    assert plan.analysis_confidence == 0.0


@pytest.mark.asyncio
async def test_circuit_breaker_transitions():
    """Verify CircuitBreaker state transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
    cb = CircuitBreaker(
        name="test_cb",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            recovery_timeout_seconds=1,
        ),
    )
    assert cb.state == CircuitState.CLOSED

    # Trigger 3 failures to open circuit
    for _ in range(3):
        await cb.on_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.is_open() is True

    # Immediate call should raise CircuitOpenError
    with pytest.raises(CircuitOpenError):
        async with cb.call():
            pass

    # Wait for recovery window to allow HALF_OPEN probe
    cb.last_failure_time -= 2.0  # Simulate 2 seconds passed
    assert cb.is_open() is False

    # First probe success
    async with cb.call():
        pass
    assert cb.state == CircuitState.HALF_OPEN

    # Second probe success -> closes circuit
    async with cb.call():
        pass
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
