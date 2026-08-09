"""
Live Integration Test Script for DeepResearchGraph.
Executes an end-to-end multi-step deep research investigation against real live internet data.
"""

import asyncio

from app.context.schemas import ContextBundle
from app.models.execution_plan import ExecutionPlan, UserMode
from app.models.pipeline_context import PipelineContext
from app.tools.dispatcher import ToolDispatcher
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool
from app.workflow_engine.langgraph_workflows.deep_research import (
    DeepResearchGraphBuilder,
)
from app.workflow_engine.mode_dispatcher import ModeDispatcher


async def main():
    print("\n" + "=" * 70)
    print("LIVE REAL-SITUATION INTEGRATION TEST: DEEP RESEARCH GRAPH")
    print("=" * 70 + "\n")

    query = (
        "Comprehensive architectural evolution of Mixture of Experts (MoE) in Large Language Models: "
        "Routing mechanisms, expert capacity factors, load balancing loss, and fine-tuning challenges."
    )

    print(f"Target Deep Research Investigation:\n'{query}'\n")

    # 1. Initialize Tool Framework & Dispatcher
    registry = ToolRegistry()
    web_search = WebSearchTool(timeout_ms=10000)
    registry.register(web_search, enabled=True)
    dispatcher = ToolDispatcher(registry=registry, executor=ToolExecutor())

    # 2. Build DeepResearchGraph with max_iterations=3
    print("Stage 1: Building and Compiling DeepResearchGraph...")
    deep_research_graph = DeepResearchGraphBuilder(
        tool_dispatcher=dispatcher, max_iterations=3
    ).build()
    mode_dispatcher = ModeDispatcher(handlers={}, graphs={"deep_research": deep_research_graph})
    print("   DeepResearchGraph compiled successfully.\n")

    # 3. Formulate Pipeline Context & Execution Plan
    ctx = PipelineContext(
        conversation_id="conv_live_deep_research",
        user_id="user_principal_researcher",
        message_id="msg_dr_live_1",
        request_id="req_dr_live_1",
        user_message=query,
        mode_hint=UserMode.DEEP_RESEARCH,
        context_bundle=ContextBundle.empty(),
    )
    plan = ExecutionPlan(
        mode=UserMode.DEEP_RESEARCH,
        tools=[],
        max_iterations=3,
    )

    # 4. Dispatch and Run Multi-Step Deep Research
    print("Stage 2: Initiating Deep Multi-Iteration Research Execution...")
    result = await mode_dispatcher.dispatch(plan, ctx)

    print("\n" + "=" * 70)
    print("DEEP RESEARCH EXECUTION METRICS")
    print("=" * 70)
    print(f"Mode: {result.mode}")
    print(f"Engine Type: {result.engine_type}")
    print(f"Total Search Iterations Dispatched: {len(result.tool_outputs)}")

    print("\n" + "=" * 70)
    print("FINAL PUBLICATION-GRADE DEEP RESEARCH REPORT")
    print("=" * 70 + "\n")
    print(result.draft_content)
    print("\n" + "=" * 70)
    print("Deep Research live test completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
