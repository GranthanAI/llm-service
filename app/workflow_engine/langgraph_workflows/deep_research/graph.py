"""
DeepResearchGraph Builder and StateGraph Compilation.
Implements LLD v2.0 Section 13.2.1 and Section 14.4.
"""

from typing import Any

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config.settings import LLMServiceConfig
from app.tools.dispatcher import ToolDispatcher
from app.workflow_engine.langgraph_workflows.deep_research.edges import (
    route_need_more_information,
)
from app.workflow_engine.langgraph_workflows.deep_research.nodes import (
    make_analyze_node,
    make_compare_sources_node,
    make_generate_report_node,
    make_search_node,
    make_summarize_node,
)
from app.workflow_engine.langgraph_workflows.deep_research.state import (
    DeepResearchGraphState,
)


class DeepResearchGraphBuilder:
    """
    Builder for the DeepResearchGraph LangGraph workflow.
    Constructs an iterative multi-step research loop with coverage analysis,
    source cross-referencing, thematic summarization, and publication report generation.
    """

    def __init__(
        self,
        tool_dispatcher: ToolDispatcher,
        prompt_registry: Any | None = None,
        config: LLMServiceConfig | None = None,
        max_iterations: int = 5,
    ):
        self.tool_dispatcher = tool_dispatcher
        self.prompt_registry = prompt_registry
        self.max_iterations = (
            config.langgraph_max_loop_iterations_deep_research
            if config is not None
            else max_iterations
        )

    def build(self) -> CompiledStateGraph:
        """Constructs and compiles the DeepResearchGraph StateGraph."""
        graph = StateGraph(DeepResearchGraphState)

        # 1. Register Nodes
        graph.add_node("search", make_search_node(self.tool_dispatcher))
        graph.add_node("analyze", make_analyze_node(self.prompt_registry))
        graph.add_node("compare_sources", make_compare_sources_node(self.prompt_registry))
        graph.add_node("summarize", make_summarize_node(self.prompt_registry))
        graph.add_node("generate_report", make_generate_report_node(self.prompt_registry))

        # 2. Configure Entry & Edges
        graph.set_entry_point("search")
        graph.add_edge("search", "analyze")

        # 3. Conditional Multi-Step Search Loop Edge
        graph.add_conditional_edges(
            "analyze",
            route_need_more_information,
            {
                "search_again": "search",
                "proceed": "compare_sources",
            },
        )

        # 4. Downstream Synthesis Pipeline
        graph.add_edge("compare_sources", "summarize")
        graph.add_edge("summarize", "generate_report")
        graph.set_finish_point("generate_report")

        return graph.compile()
