"""
Unit tests for Phase 2 Core Models, Schemas, and Enums.
"""

import json

from app.context.schemas import (
    ContextBundle,
    DocumentChunk,
    EntityNode,
    Fact,
    GraphContext,
    MemoryContext,
    Message,
    RelationshipEdge,
    RetrievalContext,
    Role,
)
from app.models.execution_plan import (
    ExecutionPlan,
    IntentCategory,
    ReasoningMode,
    Skill,
    ToolCall,
    UserMode,
)
from app.models.pipeline_context import PipelineContext
from app.models.provider import (
    ProviderSelection,
    ProviderType,
)
from app.models.request import ChatMessageCreatedEvent, TraceContext
from app.models.response import (
    ChatResponseGeneratedEvent,
    UsageMetrics,
)
from app.models.tool import ToolResult, ToolSchema, ValidationResult
from app.request_analyzer.schemas import SafeDefaultPlan
from app.workflow_engine.langgraph_workflows.deep_research.state import Finding
from app.workflow_engine.langgraph_workflows.smart.state import SubTask
from app.workflow_engine.workflow_result import ModeHandlerOutput, WorkflowResult


def test_kafka_consumed_event_schema_compatibility():
    """Verify deserialization of canonical chat.message.created Kafka event."""
    raw_payload = {
        "event_type": "chat.message.created",
        "schema_version": "2.0",
        "message_id": "msg_abc123",
        "conversation_id": "conv_xyz789",
        "user_id": "user_def456",
        "content": "Explain quantum entanglement",
        "mode_hint": "tutor",
        "file_ids": ["file_1", "file_2"],
        "metadata": {
            "client_timestamp": "2026-08-07T14:47:51Z",
            "client_version": "3.2.1",
        },
        "trace_context": {
            "traceparent": "00-abc123-def456-01",
            "tracestate": "",
        },
        "timestamp": "2026-08-07T14:47:51.234Z",
    }
    event = ChatMessageCreatedEvent.model_validate(raw_payload)
    assert event.message_id == "msg_abc123"
    assert event.mode_hint == "tutor"
    assert len(event.file_ids) == 2
    assert event.trace_context.traceparent == "00-abc123-def456-01"


def test_kafka_produced_event_schema_compatibility():
    """Verify serialization of canonical chat.response.generated Kafka event."""
    event = ChatResponseGeneratedEvent(
        response_id="resp_ghi789",
        conversation_id="conv_xyz789",
        user_id="user_def456",
        request_message_id="msg_abc123",
        full_content="Quantum entanglement is...",
        provider="nvidia",
        generation_fallback_used=False,
        mode="tutor",
        skill="tutor",
        engine_type="mode_handler",
        usage=UsageMetrics(prompt_tokens=1200, completion_tokens=320, total_tokens=1520),
        cost_usd=0.0076,
        latency_ms=1840.0,
        ttft_ms=520.0,
        context_sources=["memory", "graph", "retrieval"],
        trace_context=TraceContext(traceparent="00-abc123-def456-01"),
    )
    dumped = json.loads(event.model_dump_json())
    assert dumped["event_type"] == "chat.response.generated"
    assert dumped["usage"]["total_tokens"] == 1520
    assert dumped["engine_type"] == "mode_handler"


def test_pipeline_context_from_event():
    """Test PipelineContext instantiation from incoming event."""
    event = ChatMessageCreatedEvent(
        message_id="msg_001",
        conversation_id="conv_001",
        user_id="user_001",
        content="What is Python?",
        mode_hint="code",
        file_ids=[],
        trace_context=TraceContext(traceparent="00-trace1-span1-01"),
    )
    ctx = PipelineContext.from_event(event=event, request_id="req_uuid_123")
    assert ctx.request_id == "req_uuid_123"
    assert ctx.conversation_id == "conv_001"
    assert ctx.mode_hint == UserMode.CODE
    assert ctx.user_message == "What is Python?"
    assert ctx.context_bundle is None
    assert ctx.plan is None


def test_context_bundle_and_sources():
    """Test ContextBundle with memory, graph, and retrieval sub-contexts."""
    memory = MemoryContext(
        short_term_messages=[Message(role=Role.USER, content="Hello")],
        long_term_facts=[Fact(fact_id="f1", statement="User likes Python")],
    )
    graph = GraphContext(
        entities=[EntityNode(id="e1", name="Python", type="Language")],
        relationships=[RelationshipEdge(source_id="e1", target_id="e2", relation_type="USES")],
        subgraph_summary="Python programming language",
    )
    retrieval = RetrievalContext(
        chunks=[DocumentChunk(chunk_id="c1", file_id="f1", content="Chunk text", score=0.95)],
        total_chunks=1,
    )
    bundle = ContextBundle(
        memory=memory,
        graph=graph,
        retrieval=retrieval,
        degraded=False,
    )
    assert len(bundle.memory.short_term_messages) == 1
    assert len(bundle.graph.entities) == 1
    assert len(bundle.retrieval.chunks) == 1
    assert bundle.degraded is False


def test_execution_plan_and_safe_default():
    """Test ExecutionPlan and SafeDefaultPlan fallback behavior."""
    plan = ExecutionPlan(
        intent=IntentCategory.CODE_GENERATION,
        mode=UserMode.CODE,
        skill=Skill.CODING,
        reasoning=ReasoningMode.CHAIN_OF_THOUGHT,
        tools=[ToolCall(tool_name="web_search", required=False)],
        max_iterations=1,
    )
    assert plan.mode == UserMode.CODE
    assert len(plan.tools) == 1

    safe = SafeDefaultPlan()
    assert safe.mode == UserMode.DEFAULT
    assert safe.skill == Skill.GENERAL_CHAT
    assert safe.analysis_confidence == 0.0
    assert len(safe.tools) == 0


def test_workflow_result_normalization():
    """Verify WorkflowResult normalizes from both ModeHandlerOutput and LangGraph states."""
    # 1. Mode Handler output normalization
    tool_res = ToolResult(tool_name="web_search", success=True, data={"results": []})
    handler_out = ModeHandlerOutput(
        mode="web_search",
        tool_outputs=[tool_res],
        conversation_history=[Message(role=Role.USER, content="Search")],
        user_message="Search",
    )
    res1 = WorkflowResult.from_mode_handler(handler_out)
    assert res1.engine_type == "mode_handler"
    assert res1.draft_content is None
    assert len(res1.tool_outputs) == 1

    # 2. LangGraph state normalization
    graph_state = {
        "mode": "smart",
        "user_message": "Solve this",
        "draft_response": "Draft solution",
        "tool_results": [tool_res],
        "conversation_history": [],
        "loop_iteration_count": 2,
    }
    res2 = WorkflowResult.from_graph_state(graph_state)
    assert res2.engine_type == "langgraph"
    assert res2.draft_content == "Draft solution"
    assert res2.metadata["loop_iterations"] == 2


def test_tool_and_provider_schemas():
    """Test Tool and Provider schemas."""
    t_schema = ToolSchema(
        name="web_search",
        description="Search web",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    assert t_schema.name == "web_search"

    v_res = ValidationResult(valid=True)
    assert v_res.valid is True

    p_sel = ProviderSelection(
        provider=ProviderType.NVIDIA,
        model="meta/llama-3.1-70b-instruct",
        is_fallback=False,
    )
    assert p_sel.provider == ProviderType.NVIDIA


def test_subtask_and_finding_models():
    """Test SubTask and Finding models for SmartGraph and DeepResearchGraph."""
    st = SubTask(
        id="task_1", description="Search documentation", tool_name="web_search", required=True
    )
    assert st.id == "task_1"
    assert st.completed is False

    finding = Finding(source="web", snippet="Discovered fact", relevance_score=0.9)
    assert finding.source == "web"
    assert finding.relevance_score == 0.9
