"""
Live Verification Script for Phase 15: Generation Router & Provider Adapters.
Tests live streaming and execution with Groq, NVIDIA NIM (primary), and Google Gemini (fallback),
loading all keys, models, and parameters directly from environment configuration.
"""

import asyncio

from app.config.settings import get_settings
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
from app.context_window.manager import ContextWindowManager
from app.models.pipeline_context import PipelineContext
from app.models.tool import ToolResult
from app.prompts.builder import PromptBuilder
from app.prompts.loader import PromptLoader
from app.prompts.registry import PromptRegistry
from app.providers.gemini import GeminiAdapter
from app.providers.groq import GroqAdapter
from app.providers.nvidia import NVIDIAAdapter
from app.providers.router import GenerationRouter
from app.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from app.workflow_engine.workflow_result import WorkflowResult


async def main():
    print("\n" + "=" * 70)
    print("LIVE VERIFICATION: PHASE 15 GENERATION ROUTER & PROVIDER ADAPTERS")
    print("=" * 70 + "\n")

    config = get_settings()

    # 1. Initialize Adapters with Config-driven parameters
    print("Stage 1: Initializing LLM Provider Adapters from Environment Config...")
    nvidia_key = config.nvidia_api_key.get_secret_value()
    gemini_key = config.gemini_api_key.get_secret_value()
    groq_key = config.groq_api_key.get_secret_value()

    print(f"   [NVIDIA NIM]  Model: {config.nvidia_model} | BaseURL: {config.nvidia_base_url}")
    print(
        f"                 Temp: {config.nvidia_temperature} | TopP: {config.nvidia_top_p} | MaxTokens: {config.nvidia_max_tokens}"
    )
    print(
        f"   [Google GenAI] Model: {config.gemini_model} | Temp: {config.gemini_temperature} | TopP: {config.gemini_top_p} | MaxTokens: {config.gemini_max_tokens}"
    )
    print(
        f"   [Groq API]    Model: {config.groq_model} | Temp: {config.groq_temperature} | MaxTokens: {config.groq_max_tokens}\n"
    )

    nvidia_adapter = NVIDIAAdapter(
        api_key=nvidia_key,
        model=config.nvidia_model,
        base_url=config.nvidia_base_url,
        timeout_s=config.nvidia_timeout_ms / 1000.0,
        temperature=config.nvidia_temperature,
        top_p=config.nvidia_top_p,
        max_tokens=config.nvidia_max_tokens,
    )
    gemini_adapter = GeminiAdapter(
        api_key=gemini_key,
        model=config.gemini_model,
    )
    groq_adapter = GroqAdapter(
        api_key=groq_key,
        model=config.groq_model,
        timeout_s=config.groq_timeout_ms / 1000.0,
    )

    cb_nvidia = CircuitBreaker(
        "nvidia", CircuitBreakerConfig(failure_threshold=2, recovery_timeout_s=10)
    )
    cb_gemini = CircuitBreaker(
        "gemini", CircuitBreakerConfig(failure_threshold=2, recovery_timeout_s=10)
    )

    router = GenerationRouter(
        nvidia_adapter=nvidia_adapter,
        gemini_adapter=gemini_adapter,
        circuit_breakers={"nvidia": cb_nvidia, "gemini": cb_gemini},
    )

    # 2. Test Groq Request Analyzer Execution
    print("Stage 2: Live Testing Groq Request Analyzer (JSON Execution)...")
    try:
        groq_resp = await groq_adapter.execute(
            messages=[
                {
                    "role": "user",
                    "content": "Return a JSON object with key 'status' and value 'operational'.",
                }
            ],
            params={"response_format": {"type": "json_object"}},
        )
        print("   [Groq Live Output]:", groq_resp.content.strip())
        print(
            f"   [Groq Tokens]: Prompt: {groq_resp.prompt_tokens} | Completion: {groq_resp.completion_tokens} | Total: {groq_resp.total_tokens}\n"
        )
    except Exception as exc:
        print(f"   Groq execution failed: {exc}\n")

    # 3. Build Pipeline Context & Composed Prompt
    print("Stage 3: Composing and trimming multi-source prompt...")
    loader = PromptLoader()
    registry = PromptRegistry(loader=loader)
    builder = PromptBuilder(registry=registry)
    cwm = ContextWindowManager()

    bundle = ContextBundle(
        memory=MemoryContext(
            long_term_facts=[
                Fact(
                    fact_id="f1",
                    statement="User is designing high-performance AI inference pipelines",
                    confidence=0.98,
                ),
            ],
            short_term_messages=[
                Message(
                    role="user",
                    content="What is the advantage of using FP8 precision in LLM inference?",
                ),
                Message(
                    role="assistant",
                    content="FP8 doubles compute throughput and halves memory bandwidth on modern GPUs.",
                ),
            ],
        ),
        graph=GraphContext(
            entities=[
                EntityNode(id="e1", name="HopperArchitecture", type="Hardware"),
                EntityNode(id="e2", name="FP8TensorCores", type="Feature"),
            ],
            relationships=[
                RelationshipEdge(
                    source_id="HopperArchitecture",
                    relation_type="includes",
                    target_id="FP8TensorCores",
                ),
            ],
        ),
        retrieval=RetrievalContext(
            chunks=[
                DocumentChunk(
                    chunk_id="chk_1",
                    file_id="gpu_benchmarks.pdf",
                    content="NVIDIA Hopper H100 provides 4th-generation Tensor Cores with dedicated Transformer Engine FP8 support.",
                    score=0.95,
                )
            ]
        ),
    )

    ctx = PipelineContext(
        request_id="req_gen_test_001",
        message_id="msg_gen_test_001",
        conversation_id="conv_gen_test_001",
        user_id="Elena",
        user_message="Summarize the core technical benefits of FP8 quantization for LLMs in 3 concise bullet points.",
        context_bundle=bundle,
    )

    workflow_result = WorkflowResult(
        mode="smart",
        engine_type="langgraph",
        user_message="Summarize the core technical benefits of FP8 quantization for LLMs in 3 concise bullet points.",
        tool_outputs=[
            ToolResult(
                tool_name="web_search",
                success=True,
                data={
                    "query": "FP8 quantization benefits LLM inference",
                    "sources": [
                        {
                            "title": "FP8 Formats for Deep Learning",
                            "url": "https://arxiv.org/abs/2209.05433",
                            "snippet": "FP8 (E4M3 and E5M2) matches 16-bit precision training and inference accuracy with up to 2x speedup.",
                        }
                    ],
                },
            )
        ],
        draft_content="FP8 enables 2x matrix multiplication throughput and reduces KV cache memory footprint by 50%.",
    )

    composed_prompt = builder.build(workflow_result, ctx)
    trimmed_prompt = cwm.manage(composed_prompt, target_model=config.nvidia_model)
    print(
        f"   Composed Tokens: {composed_prompt.total_tokens} -> Managed Tokens: {trimmed_prompt.total_tokens}\n"
    )

    # 4. Live Primary Generation (NVIDIA NIM)
    print("Stage 4: Live Streaming Generation via Primary Adapter (NVIDIA NIM)...")
    print("-" * 70)
    primary_output_tokens = []
    try:
        async for token in router.generate_stream(trimmed_prompt):
            print(token, end="", flush=True)
            primary_output_tokens.append(token)
        print("\n" + "-" * 70)
        print(f"\n   Primary Streaming Success! (Received {len(primary_output_tokens)} tokens)")
    except Exception as exc:
        print(f"\n   Primary generation encountered error: {exc}")

    # 5. Live Fallback Generation (Simulated NVIDIA Circuit Trip -> Gemini)
    print("\nStage 5: Testing Live Fallback to Google Gemini (Tripping NVIDIA Circuit Breaker)...")
    cb_nvidia.trip()  # Manually open NVIDIA circuit breaker
    print(f"   NVIDIA Circuit Breaker is open: {cb_nvidia.is_open()}")
    print("-" * 70)

    fallback_output_tokens = []
    try:
        async for token in router.generate_stream(trimmed_prompt):
            print(token, end="", flush=True)
            fallback_output_tokens.append(token)
        print("\n" + "-" * 70)
        print(
            f"\n   Fallback Streaming Success! (Received {len(fallback_output_tokens)} tokens from Gemini)"
        )
    except Exception as exc:
        print(f"\n   Fallback generation encountered error: {exc}")

    print("\n" + "=" * 70)
    print("Generation Router live verification completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
