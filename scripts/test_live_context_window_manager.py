"""
Live Verification Script for Phase 14: Context Window Manager.
Tests token calibration, budget calculation, priority trimming, and model limit enforcement.
"""

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
)
from app.context_window.budget import BudgetCalculator
from app.context_window.manager import ContextWindowManager
from app.context_window.token_counter import TokenCounter
from app.models.pipeline_context import PipelineContext
from app.models.tool import ToolResult
from app.prompts.builder import PromptBuilder
from app.prompts.loader import PromptLoader
from app.prompts.registry import PromptRegistry
from app.workflow_engine.workflow_result import WorkflowResult


def main():
    print("\n" + "=" * 70)
    print("LIVE VERIFICATION: PHASE 14 CONTEXT WINDOW MANAGER")
    print("=" * 70 + "\n")

    # 1. Initialize Components
    print("Stage 1: Initializing TokenCounter and ContextWindowManager...")
    counter = TokenCounter()
    manager = ContextWindowManager(token_counter=counter)

    sample_text = (
        "GraphGPT orchestrates multi-agent graph workflows, deterministic mode handlers, "
        "and multi-provider LLM adapters with strict priority-based context window enforcement."
    )
    c_nvidia = counter.count_text(sample_text, "nvidia")
    c_groq = counter.count_text(sample_text, "groq")
    c_gemini = counter.count_text(sample_text, "gemini")
    print(f"   Calibrated Tokens -> NVIDIA: {c_nvidia} | Groq: {c_groq} | Gemini: {c_gemini}")

    # 2. Model Limits and Budget Calculation
    print("\nStage 2: Model Context Limits & Budget Allocation...")
    models = ["meta/llama-3.1-405b-instruct", "gemini-1.5-pro", "llama-3.3-70b-versatile"]
    for m in models:
        limits = manager.get_model_limits(m)
        budget = BudgetCalculator.calculate(limits)
        print(f"   Model: '{m}'")
        print(
            f"     Context Window: {limits.context_window:,} tokens | Reserved Output: {limits.max_output_tokens:,} tokens"
        )
        print(f"     Effective Input Budget: {budget.input_budget:,} tokens")

    # 3. Prompt Construction via Prompt Engine
    print("\nStage 3: Building multi-section Prompt via Prompt Engine...")
    loader = PromptLoader()
    registry = PromptRegistry(loader=loader)
    builder = PromptBuilder(registry=registry)

    bundle = ContextBundle(
        memory=MemoryContext(
            long_term_facts=[
                Fact(
                    fact_id="f1",
                    statement="User is leading an AI infrastructure team",
                    confidence=0.99,
                ),
                Fact(
                    fact_id="f2", statement="Deploying on NVIDIA H100 GPU clusters", confidence=0.97
                ),
            ],
            short_term_messages=[
                Message(role="user", content="How do we maximize throughput for Llama-3.1-405B?"),
                Message(
                    role="assistant",
                    content="We should utilize Tensor Parallelism = 8 and FP8 quantization.",
                ),
            ],
        ),
        graph=GraphContext(
            entities=[
                EntityNode(id="e1", name="TensorParallelism", type="Technique"),
                EntityNode(id="e2", name="FP8Quantization", type="Technique"),
            ],
            relationships=[
                RelationshipEdge(
                    source_id="FP8Quantization",
                    relation_type="complements",
                    target_id="TensorParallelism",
                ),
            ],
        ),
        retrieval=RetrievalContext(
            chunks=[
                DocumentChunk(
                    chunk_id="chk_perf_01",
                    file_id="nim_serving_guide.pdf",
                    content="NVIDIA NIM with TensorRT-LLM achieves 2.3x higher token throughput using FP8 GEMMs on Hopper architecture."
                    * 4,
                    score=0.96,
                ),
                DocumentChunk(
                    chunk_id="chk_perf_02",
                    file_id="vllm_benchmarks.pdf",
                    content="Chunked prefill and PagedAttention reduce KV cache fragmentation by 96% under heavy concurrent loads."
                    * 4,
                    score=0.91,
                ),
            ]
        ),
    )

    ctx = PipelineContext(
        request_id="req_ctx_test_001",
        message_id="msg_ctx_test_001",
        conversation_id="conv_context_window_test",
        user_id="Alex",
        user_message="Provide recommendations for low-latency batch inference on Hopper GPUs.",
        context_bundle=bundle,
    )

    workflow_result = WorkflowResult(
        mode="smart",
        engine_type="langgraph",
        user_message="Provide recommendations for low-latency batch inference on Hopper GPUs.",
        tool_outputs=[
            ToolResult(
                tool_name="web_search",
                success=True,
                data={
                    "query": "Hopper FP8 inference throughput benchmarks",
                    "sources": [
                        {
                            "title": "Hopper Architecture Deep Dive",
                            "url": "https://developer.nvidia.com/blog/hopper-inference",
                            "snippet": "Transformer Engine dynamically chooses between FP8 and FP16 formats to maximize compute density.",
                        }
                    ],
                },
            )
        ],
        draft_content=(
            "### Agentic Synthesis:\n"
            "1. FP8 Execution: Halves bandwidth demands without perplexity degradation.\n"
            "2. Tensor Parallelism: TP=8 across NVLink domain ensures minimal inter-node latency."
        ),
    )

    composed_prompt = builder.build(workflow_result, ctx)
    print(
        f"   Original Composed Prompt: {composed_prompt.total_tokens} tokens across {len(composed_prompt.sections)} sections."
    )

    # 4. Context Window Management - Normal vs Constrained
    print("\nStage 4: Enforcing Context Window Management...")

    # Case A: Standard budget (NVIDIA 405B - 128k input)
    trimmed_normal = manager.manage(composed_prompt, target_model="meta/llama-3.1-405b-instruct")
    print("   [Normal Case] Model: meta/llama-3.1-405b-instruct")
    print(
        f"     Was Trimmed: {trimmed_normal.was_trimmed} | Final Tokens: {trimmed_normal.total_tokens}"
    )

    # Case B: Constrained budget (e.g. 350 token budget to trigger priority trimming)
    print("\n   [Constrained Budget Case] Simulating 350-token strict budget limit...")
    trimmed_constrained = manager.manage(
        composed_prompt,
        target_model="meta/llama-3.1-405b-instruct",
        custom_input_budget=350,
    )
    print(f"     Was Trimmed: {trimmed_constrained.was_trimmed}")
    print(
        f"     Original Tokens: {trimmed_constrained.original_tokens} -> Final Tokens: {trimmed_constrained.total_tokens}"
    )
    print(f"     Sections Trimmed: {trimmed_constrained.trimmed_sections}")

    # Case C: Extreme constraint testing ContextOverflowError
    print("\n   [Extreme Overflow Test] Simulating 100-token impossible limit...")
    try:
        manager.manage(
            composed_prompt, target_model="meta/llama-3.1-405b-instruct", custom_input_budget=100
        )
    except Exception as e:
        print(f"     Successfully caught expected overflow protection: {type(e).__name__} - {e}")

    print("\n" + "=" * 70)
    print("TRIMMED PROMPT SECTIONS AFTER PRIORITY BUDGET ENFORCEMENT")
    print("=" * 70)
    for idx, s in enumerate(trimmed_constrained.sections, 1):
        status = "LOCKED" if not s.trimmable else "TRIMMABLE"
        print(
            f"[{idx}] Section '{s.name}' | Priority: {s.priority}/10 | Tokens: {s.token_count} | {status}"
        )

    print("\n" + "=" * 70)
    print("Context Window Manager live verification completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
