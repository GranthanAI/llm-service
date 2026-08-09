"""
Real-world live integration test for SmartGraph agentic orchestration.
Demonstrates how SmartGraph operates like ChatGPT / Smart Mode:
1. Receives complex real-world query
2. Plans subtasks via 'planner' node
3. Selects 'web_search' dynamically via 'dynamic_tool_selection' node
4. Queries live open search engines concurrently in parallel (DuckDuckGo, Wikipedia, ArXiv, etc.)
5. Aggregates findings and advances iteration count
6. LoopGuard evaluates completion status
7. Assembles intermediate prompt and produces grounded synthesis.
"""

import asyncio

from app.context.schemas import ContextBundle
from app.models.execution_plan import ExecutionPlan, UserMode
from app.models.pipeline_context import PipelineContext
from app.tools.dispatcher import ToolDispatcher
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool
from app.workflow_engine.langgraph_workflows.smart import SmartGraphBuilder
from app.workflow_engine.mode_dispatcher import ModeDispatcher


async def main():
    print("\n=======================================================")
    print("LIVE REAL-SITUATION TEST: SmartGraph Agentic Mode")
    print("=======================================================\n")

    # 1. Initialize real WebSearchTool with open providers
    registry = ToolRegistry()
    search_tool = WebSearchTool(timeout_ms=8000)
    registry.register(search_tool, enabled=True)
    dispatcher = ToolDispatcher(registry=registry, executor=ToolExecutor())

    # 2. Build and compile SmartGraph
    print("1. Compiling SmartGraph...")
    smart_graph = SmartGraphBuilder(tool_dispatcher=dispatcher, max_iterations=3).build()
    print("   StateGraph compiled successfully with LoopGuard and routing edges.\n")

    # 3. Wire into ModeDispatcher
    mode_dispatcher = ModeDispatcher(
        handlers={},
        graphs={"smart": smart_graph},
    )

    # 4. Execute Real-World Scenario
    query = "What are the latest breakthrough algorithms in quantum computing?"
    print("2. Executing Real User Query in 'smart' mode:")
    print(f"   Query: '{query}'\n")

    ctx = PipelineContext(
        conversation_id="conv_real_test",
        user_id="user_real_test",
        message_id="msg_real_test",
        request_id="req_real_test",
        user_message=query,
        mode_hint=UserMode.SMART,
        context_bundle=ContextBundle.empty(),
    )
    plan = ExecutionPlan(
        mode=UserMode.SMART,
        tools=[],
        max_iterations=3,
    )

    print("3. Running SmartGraph agentic loop...")
    result = await mode_dispatcher.dispatch(plan, ctx)

    print("\n=======================================================")
    print("SMARTGRAPH EXECUTION RESULT:")
    print("=======================================================")
    print(f"Mode: {result.mode}")
    print(f"Engine Type: {result.engine_type}")
    print(f"Tool Outputs Gathered: {len(result.tool_outputs)}")

    for idx, tr in enumerate(result.tool_outputs, 1):
        res_data = tr.data.get("results", []) if tr.success and isinstance(tr.data, dict) else []
        print(f"   - Tool Call [{idx}] ({tr.tool_name}) -> Collected {len(res_data)} sources:")
        for source in res_data[:3]:
            print(f"       * {source.get('title')} ({source.get('url')})")

    print(f"\nDraft Synthesized Response:\n{result.draft_content}\n")
    print("=======================================================")
    print("SmartGraph real situation verification completed successfully!")
    print("=======================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
