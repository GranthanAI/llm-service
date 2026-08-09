"""
SmartGraph StateGraph Builder and Compilation.
Implements LLD v2.0 Section 13.1.1 and Section 13.1.5.
"""

from typing import Any

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config.settings import LLMServiceConfig
from app.tools.dispatcher import ToolDispatcher
from app.workflow_engine.langgraph_workflows.smart.edges import route_after_execution
from app.workflow_engine.langgraph_workflows.smart.nodes import (
    make_execution_node,
    make_generation_node,
    make_planner_node,
    make_prompt_node,
    make_tool_selection_node,
)
from app.workflow_engine.langgraph_workflows.smart.state import SmartGraphState


class SmartGraphBuilder:
    """
    Constructs and compiles the SmartGraph LangGraph instance for iterative
    agentic planning, dynamic tool selection, execution, and synthesis.
    """

    def __init__(
        self,
        tool_dispatcher: ToolDispatcher,
        config: LLMServiceConfig | None = None,
        max_iterations: int = 5,
        llm_client: Any | None = None,
    ):
        self.tool_dispatcher = tool_dispatcher
        self.max_iterations = (
            config.langgraph_max_loop_iterations_smart if config is not None else max_iterations
        )
        self.llm_client = llm_client

    def build(self) -> CompiledStateGraph:
        """Construct StateGraph, wire nodes, edges, conditional loop guard, and compile."""
        graph = StateGraph(SmartGraphState)

        # 1. Register Nodes
        graph.add_node("planner", make_planner_node(llm_client=self.llm_client))
        graph.add_node("dynamic_tool_selection", make_tool_selection_node())
        graph.add_node("execution", make_execution_node(tool_dispatcher=self.tool_dispatcher))
        graph.add_node("prompt", make_prompt_node())
        graph.add_node("generation", make_generation_node())

        # 2. Wire Entry and Static Edges
        graph.set_entry_point("planner")
        graph.add_edge("planner", "dynamic_tool_selection")
        graph.add_edge("dynamic_tool_selection", "execution")

        # 3. Wire Conditional Loop Edge (Section 13.1.4)
        graph.add_conditional_edges(
            "execution",
            route_after_execution,
            {
                "loop": "dynamic_tool_selection",
                "proceed": "prompt",
            },
        )

        # 4. Wire Terminal Synthesis Edges
        graph.add_edge("prompt", "generation")
        graph.set_finish_point("generation")

        return graph.compile()
