"""
Unit Tests for DeepResearchGraph Workflow, State, Nodes, Edges, LoopGuard, and ModeDispatcher.
Implements test coverage for Phase 12 as per LLD v2.0 Section 13.2.
"""

from unittest.mock import AsyncMock

import pytest

from app.context.schemas import ContextBundle
from app.models.execution_plan import ExecutionPlan, UserMode
from app.models.pipeline_context import PipelineContext
from app.models.tool import ToolResult
from app.tools.dispatcher import ToolDispatcher
from app.workflow_engine.langgraph_workflows.deep_research import (
    DeepResearchGraphBuilder,
    DeepResearchGraphState,
    Finding,
    make_analyze_node,
    make_compare_sources_node,
    make_generate_report_node,
    make_search_node,
    make_summarize_node,
    route_need_more_information,
)
from app.workflow_engine.mode_dispatcher import ModeDispatcher

# --- 1. Loop Guard and Edge Routing Tests ---


def test_route_need_more_information():
    """Verify route_need_more_information conditional transitions."""
    # Case 1: Coverage sufficient -> proceed
    state_covered: DeepResearchGraphState = {
        "coverage_sufficient": True,
        "loop_iteration_count": 1,
        "max_iterations": 5,
    }  # type: ignore
    assert route_need_more_information(state_covered) == "proceed"

    # Case 2: Coverage insufficient but iteration count < max -> search_again
    state_need_more: DeepResearchGraphState = {
        "coverage_sufficient": False,
        "loop_iteration_count": 1,
        "max_iterations": 5,
    }  # type: ignore
    assert route_need_more_information(state_need_more) == "search_again"

    # Case 3: Loop cap reached -> force proceed
    state_capped: DeepResearchGraphState = {
        "coverage_sufficient": False,
        "loop_iteration_count": 5,
        "max_iterations": 5,
    }  # type: ignore
    assert route_need_more_information(state_capped) == "proceed"


# --- 2. Node Execution Tests ---


@pytest.mark.asyncio
async def test_deep_research_nodes():
    """Verify search, analyze, compare_sources, summarize, and generate_report nodes."""
    # 1. Search Node
    mock_dispatcher = AsyncMock(spec=ToolDispatcher)
    mock_dispatcher.dispatch.return_value = [
        ToolResult(
            tool_name="web_search",
            success=True,
            data={
                "results": [
                    {
                        "source": "arxiv",
                        "title": "Deep Learning Research",
                        "url": "https://arxiv.org/abs/123",
                        "snippet": "Breakthrough neural architectures",
                    },
                    {
                        "source": "wikipedia",
                        "title": "Machine Learning",
                        "url": "https://en.wikipedia.org/wiki/ML",
                        "snippet": "Overview of computational learning",
                    },
                ]
            },
        )
    ]
    search_node = make_search_node(tool_dispatcher=mock_dispatcher)
    state_in: DeepResearchGraphState = {
        "user_message": "Recent advances in generative AI architectures",
        "queries_issued": [],
        "loop_iteration_count": 0,
    }  # type: ignore
    search_out = await search_node(state_in)
    assert len(search_out["search_results"]) == 1
    assert search_out["loop_iteration_count"] == 1
    assert "Recent advances in generative AI architectures" in search_out["queries_issued"]

    # 2. Analyze Node
    analyze_node = make_analyze_node()
    state_analyze: DeepResearchGraphState = {
        "user_message": "Recent advances in generative AI architectures",
        "search_results": search_out["search_results"],
        "findings": [],
        "loop_iteration_count": 1,
        "max_iterations": 3,
    }  # type: ignore
    analyze_out = await analyze_node(state_analyze)
    assert len(analyze_out["findings"]) == 2
    assert isinstance(analyze_out["findings"][0], Finding)

    # 3. Compare Sources Node
    compare_node = make_compare_sources_node()
    state_compare: DeepResearchGraphState = {
        "findings": analyze_out["findings"],
    }  # type: ignore
    compare_out = await compare_node(state_compare)
    assert len(compare_out["cross_referenced_findings"]) == 2

    # 4. Summarize Node
    summarize_node = make_summarize_node()
    state_summarize: DeepResearchGraphState = {
        "user_message": "Recent advances in generative AI architectures",
        "cross_referenced_findings": compare_out["cross_referenced_findings"],
    }  # type: ignore
    summarize_out = await summarize_node(state_summarize)
    assert "Thematic research synthesis" in summarize_out["synthesis"]

    # 5. Generate Report Node
    report_node = make_generate_report_node()
    state_report: DeepResearchGraphState = {
        "user_message": "Recent advances in generative AI architectures",
        "cross_referenced_findings": compare_out["cross_referenced_findings"],
        "loop_iteration_count": 1,
    }  # type: ignore
    report_out = await report_node(state_report)
    assert "# Deep Research Report:" in report_out["structured_report"]
    assert "## 1. Executive Summary" in report_out["structured_report"]
    assert "## 6. Grounded References & Source Citations" in report_out["structured_report"]


# --- 3. End-to-End DeepResearchGraph Invocation & Builder Tests ---


@pytest.mark.asyncio
async def test_deep_research_graph_builder_and_ainvoke():
    """Verify compiling DeepResearchGraph and running end-to-end ainvoke."""
    mock_dispatcher = AsyncMock(spec=ToolDispatcher)
    mock_dispatcher.dispatch.return_value = [
        ToolResult(
            tool_name="web_search",
            success=True,
            data={
                "results": [
                    {
                        "source": "arxiv",
                        "title": "Quantum Deep Learning",
                        "url": "https://arxiv.org/abs/quantum",
                        "snippet": "Quantum variational circuits for deep learning.",
                    },
                    {
                        "source": "wikipedia",
                        "title": "Quantum computing",
                        "url": "https://en.wikipedia.org/wiki/Quantum_computing",
                        "snippet": "Information processing using quantum states.",
                    },
                    {
                        "source": "duckduckgo",
                        "title": "Quantum AI Overview",
                        "url": "https://duckduckgo.com/qai",
                        "snippet": "Recent survey on quantum machine learning algorithms.",
                    },
                    {
                        "source": "openalex",
                        "title": "Scholarly Quantum Analysis",
                        "url": "https://openalex.org/w123",
                        "snippet": "Peer-reviewed analysis of hybrid classical-quantum models.",
                    },
                    {
                        "source": "wikidata",
                        "title": "Quantum Neural Network Entity",
                        "url": "https://www.wikidata.org/wiki/Q100",
                        "snippet": "Structured entity for quantum neural networks.",
                    },
                    {
                        "source": "stackexchange",
                        "title": "Qiskit Optimizer Question",
                        "url": "https://quantumcomputing.stackexchange.com/q/1",
                        "snippet": "Implementation of VQE optimizer in Qiskit.",
                    },
                ]
            },
        )
    ]

    builder = DeepResearchGraphBuilder(tool_dispatcher=mock_dispatcher, max_iterations=2)
    compiled_graph = builder.build()

    initial_state: DeepResearchGraphState = {
        "conversation_id": "conv_dr_test_1",
        "user_id": "user_researcher",
        "request_id": "req_dr_1",
        "mode": "deep_research",
        "user_message": "Research quantum neural network optimizers",
        "context_bundle": ContextBundle.empty(),
        "conversation_history": [],
        "queries_issued": [],
        "search_results": [],
        "findings": [],
        "coverage_sufficient": False,
        "loop_iteration_count": 0,
        "max_iterations": 2,
        "cross_referenced_findings": None,
        "synthesis": None,
        "structured_report": None,
    }

    final_state = await compiled_graph.ainvoke(initial_state)

    assert final_state["structured_report"] is not None
    assert "# Deep Research Report:" in final_state["structured_report"]
    assert "## 1. Executive Summary" in final_state["structured_report"]
    assert len(final_state["findings"]) >= 6
    assert final_state["loop_iteration_count"] >= 1


# --- 4. ModeDispatcher Integration Tests for Deep Research ---


@pytest.mark.asyncio
async def test_mode_dispatcher_routes_to_deep_research_graph():
    """Verify ModeDispatcher resolves DEEP_RESEARCH to LangGraph and produces WorkflowResult."""
    mock_dispatcher = AsyncMock(spec=ToolDispatcher)
    mock_dispatcher.dispatch.return_value = [
        ToolResult(
            tool_name="web_search",
            success=True,
            data={
                "results": [
                    {
                        "source": "arxiv",
                        "title": "Supervised Learning",
                        "url": "https://arxiv.org",
                        "snippet": "Supervised learning bounds.",
                    }
                ]
            },
        )
    ]

    dr_graph = DeepResearchGraphBuilder(tool_dispatcher=mock_dispatcher, max_iterations=1).build()
    mode_dispatcher = ModeDispatcher(handlers={}, graphs={"deep_research": dr_graph})

    ctx = PipelineContext(
        conversation_id="conv_dr_dispatch",
        user_id="user_researcher",
        message_id="msg_dr_1",
        request_id="req_dr_1",
        user_message="Comprehensive deep dive on reinforcement learning from human feedback",
        mode_hint=UserMode.DEEP_RESEARCH,
        context_bundle=ContextBundle.empty(),
    )
    plan = ExecutionPlan(
        mode=UserMode.DEEP_RESEARCH,
        tools=[],
        max_iterations=1,
    )

    result = await mode_dispatcher.dispatch(plan, ctx)

    assert result.mode == "deep_research"
    assert result.engine_type == "langgraph"
    assert result.draft_content is not None
    assert "# Deep Research Report:" in result.draft_content
    assert len(result.tool_outputs) >= 1
