"""
Tough Prompt Real-Situation Stress Test for Smart Mode (SmartGraph).
Tests multi-stage agentic reasoning with a complex multi-facet query:
'Compare the mathematical convergence guarantees of AdamW vs Lion optimizer in deep learning, provide code implementation differences, and analyze their benchmark performance on Vision Transformers.'
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
    print("\n" + "=" * 70)
    print("DEMANDING TOUGH PROMPT STRESS TEST: SMARTGRAPH AGENTIC MODE")
    print("=" * 70 + "\n")

    tough_query = (
        "Compare the mathematical convergence guarantees of AdamW vs Lion optimizer in deep learning, "
        "provide code implementation differences, and analyze their benchmark performance on Vision Transformers."
    )

    print(f"Target Tough Query:\n'{tough_query}'\n")

    # 1. Initialize Tool Framework & Dispatcher
    registry = ToolRegistry()
    web_search = WebSearchTool(timeout_ms=10000)
    registry.register(web_search, enabled=True)
    dispatcher = ToolDispatcher(registry=registry, executor=ToolExecutor())

    # 2. Build SmartGraph with max_iterations=5
    print("Stage 1: Building and Compiling SmartGraph with LoopGuard (max_iterations=5)...")
    smart_graph = SmartGraphBuilder(tool_dispatcher=dispatcher, max_iterations=5).build()
    mode_dispatcher = ModeDispatcher(handlers={}, graphs={"smart": smart_graph})
    print("   SmartGraph compiled successfully.\n")

    # 3. Formulate Pipeline Context & Execution Plan
    ctx = PipelineContext(
        conversation_id="conv_tough_stress",
        user_id="user_researcher",
        message_id="msg_tough_1",
        request_id="req_tough_1",
        user_message=tough_query,
        mode_hint=UserMode.SMART,
        context_bundle=ContextBundle.empty(),
    )
    plan = ExecutionPlan(
        mode=UserMode.SMART,
        tools=[],
        max_iterations=5,
    )

    # 4. Dispatch and Run Multi-Stage Agentic Loop
    print("Stage 2: Initiating Multi-Stage Execution in SmartGraph...")
    result = await mode_dispatcher.dispatch(plan, ctx)

    print("\n" + "=" * 70)
    print("TOUGH PROMPT EXECUTION BREAKDOWN & STAGES OBSERVED")
    print("=" * 70)
    print(f"Mode: {result.mode}")
    print(f"Engine Type: {result.engine_type}")
    print(f"Total Tool Executions Dispatched: {len(result.tool_outputs)}")

    for idx, to in enumerate(result.tool_outputs, 1):
        res_list = to.data.get("results", []) if to.success and isinstance(to.data, dict) else []
        print(f"\n[Tool Execution Step {idx}]:")
        print(f"   Tool Name: {to.tool_name}")
        print(f"   Latency: {to.latency_ms:.2f}ms")
        print(f"   Sources Retrieved: {len(res_list)}")
        for s_idx, src in enumerate(res_list[:3], 1):
            print(f"      ({s_idx}) [{src.get('source', 'web').upper()}] {src.get('title')}")
            print(f"          URL: {src.get('url')}")
            print(f"          Snippet: {src.get('snippet', '')[:110]}...")

    print("\n" + "=" * 70)
    print("FINAL USER RESPONSE (SYNTHESIZED & GROUNDED)")
    print("=" * 70 + "\n")
    print(result.draft_content)
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
