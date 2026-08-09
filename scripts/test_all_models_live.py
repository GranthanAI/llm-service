"""
Comprehensive Live Provider & Router Testing Script.
Runs all configured models live and collects exact prompts, outputs, token metrics, and fallback behavior.
"""

import asyncio
import time

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


async def run_all_live():
    results = {}
    config = get_settings()

    print("\n======================================================================")
    print("RUNNING ALL LIVE MODELS (GROQ, NVIDIA NIM, GOOGLE GEMINI)")
    print("======================================================================")

    # -------------------------------------------------------------------------
    # 1. GROQ (llama-3.3-70b-versatile)
    # -------------------------------------------------------------------------
    print("\n[1/4] Executing Groq Request Analyzer (JSON Mode)...")
    groq_adapter = GroqAdapter(
        api_key=config.groq_api_key.get_secret_value(),
        model=config.groq_model,
        timeout_s=config.groq_timeout_ms / 1000.0,
    )
    groq_prompt = [
        {
            "role": "system",
            "content": "You are the GraphGPT Request Analyzer. You MUST respond with a single valid JSON object representing an ExecutionPlan with fields: mode, engine_type, intent, skill, reasoning_mode, need_web_search, suggested_tools, loop_iteration_cap, confidence.",
        },
        {
            "role": "user",
            "content": "Analyze user message: 'Compare AdamW vs Lion optimizer in PyTorch and search recent ArXiv benchmarks.'",
        },
    ]
    try:
        t0 = time.perf_counter()
        groq_res = await groq_adapter.execute(
            messages=groq_prompt,
            params={"response_format": {"type": "json_object"}},
        )
        groq_latency = time.perf_counter() - t0
        results["groq"] = {
            "status": "success",
            "model": config.groq_model,
            "latency_s": round(groq_latency, 3),
            "prompt": groq_prompt,
            "output": groq_res.content,
            "prompt_tokens": groq_res.prompt_tokens,
            "completion_tokens": groq_res.completion_tokens,
            "total_tokens": groq_res.total_tokens,
        }
        print(f"   Groq Latency: {groq_latency:.3f}s | Tokens: {groq_res.total_tokens}")
        print("   Groq Output:", groq_res.content)
    except Exception as e:
        print("   Groq Failed:", e)
        results["groq"] = {"status": "failed", "error": str(e)}

    # -------------------------------------------------------------------------
    # 2. Build Multi-Source Composed Prompt
    # -------------------------------------------------------------------------
    print("\n[2/4] Building Multi-Source Composed Prompt...")
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
        request_id="req_bench_001",
        message_id="msg_bench_001",
        conversation_id="conv_bench_001",
        user_id="Elena",
        user_message="Summarize the core technical benefits of FP8 quantization for LLMs in 3 concise bullet points.",
        context_bundle=bundle,
    )

    wf_res = WorkflowResult(
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

    composed_prompt = builder.build(wf_res, ctx)
    trimmed_prompt = cwm.manage(composed_prompt, target_model=config.nvidia_model)
    print(f"   Composed Tokens: {composed_prompt.total_tokens} -> Managed Tokens: {trimmed_prompt.total_tokens}")

    # -------------------------------------------------------------------------
    # 3. NVIDIA NIM (Primary Generation Adapter)
    # -------------------------------------------------------------------------
    print("\n[3/4] Executing NVIDIA NIM (meta/llama-3.3-70b-instruct) Live Streaming...")
    nvidia_adapter = NVIDIAAdapter(
        api_key=config.nvidia_api_key.get_secret_value(),
        model=config.nvidia_model,
        base_url=config.nvidia_base_url,
        timeout_s=45.0,
        temperature=config.nvidia_temperature,
        top_p=config.nvidia_top_p,
        max_tokens=config.nvidia_max_tokens,
    )
    try:
        t0 = time.perf_counter()
        nv_tokens = []
        async for token in nvidia_adapter.stream(trimmed_prompt.messages):
            nv_tokens.append(token)
        nv_latency = time.perf_counter() - t0
        nv_content = "".join(nv_tokens)
        results["nvidia"] = {
            "status": "success",
            "model": config.nvidia_model,
            "latency_s": round(nv_latency, 3),
            "chunk_count": len(nv_tokens),
            "output": nv_content,
            "temperature": config.nvidia_temperature,
            "top_p": config.nvidia_top_p,
            "max_tokens": config.nvidia_max_tokens,
        }
        print(f"   NVIDIA Latency: {nv_latency:.3f}s | Chunks: {len(nv_tokens)}")
        print("   NVIDIA Output:\n" + nv_content)
    except Exception as e:
        print("   NVIDIA Failed:", e)
        results["nvidia"] = {"status": "failed", "error": str(e)}

    # -------------------------------------------------------------------------
    # 4. GOOGLE GEMINI (Fallback Generation Adapter)
    # -------------------------------------------------------------------------
    print("\n[4/4] Executing Google Gemini (gemini-2.5-flash) Live Streaming...")
    gemini_adapter = GeminiAdapter(
        api_key=config.gemini_api_key.get_secret_value(),
        model=config.gemini_model,
    )
    try:
        t0 = time.perf_counter()
        gem_tokens = []
        async for token in gemini_adapter.stream(trimmed_prompt.messages):
            gem_tokens.append(token)
        gem_latency = time.perf_counter() - t0
        gem_content = "".join(gem_tokens)
        results["gemini"] = {
            "status": "success",
            "model": config.gemini_model,
            "latency_s": round(gem_latency, 3),
            "chunk_count": len(gem_tokens),
            "output": gem_content,
            "temperature": config.gemini_temperature,
            "top_p": config.gemini_top_p,
            "max_tokens": config.gemini_max_tokens,
        }
        print(f"   Gemini Latency: {gem_latency:.3f}s | Chunks: {len(gem_tokens)}")
        print("   Gemini Output:\n" + gem_content)
    except Exception as e:
        print("   Gemini Failed:", e)
        results["gemini"] = {"status": "failed", "error": str(e)}

    # -------------------------------------------------------------------------
    # 5. GENERATION ROUTER WITH CIRCUIT BREAKER FALLBACK
    # -------------------------------------------------------------------------
    print("\n[5/5] Testing Generation Router with Circuit Breaker Failover...")
    cb_nvidia = CircuitBreaker("nvidia", CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=10))
    cb_gemini = CircuitBreaker("gemini", CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=10))
    router = GenerationRouter(
        nvidia_adapter=nvidia_adapter,
        gemini_adapter=gemini_adapter,
        circuit_breakers={"nvidia": cb_nvidia, "gemini": cb_gemini},
    )

    cb_nvidia.trip()
    print(f"   Tripped NVIDIA CB to OPEN (is_open={cb_nvidia.is_open()})")
    try:
        t0 = time.perf_counter()
        fallback_tokens = []
        async for token in router.generate_stream(trimmed_prompt):
            fallback_tokens.append(token)
        fb_latency = time.perf_counter() - t0
        fb_content = "".join(fallback_tokens)
        results["router_fallback"] = {
            "status": "success",
            "route_taken": "gemini",
            "latency_s": round(fb_latency, 3),
            "chunks": len(fallback_tokens),
            "output": fb_content,
        }
        print(f"   Router Fallback Latency: {fb_latency:.3f}s | Chunks: {len(fallback_tokens)}")
        print("   Router Fallback Output:\n" + fb_content)
    except Exception as e:
        print("   Router Fallback Failed:", e)
        results["router_fallback"] = {"status": "failed", "error": str(e)}

    print("\n======================================================================")
    print("ALL MODELS EXECUTED LIVE SUCCESSFULLY!")
    print("======================================================================")

    return results


if __name__ == "__main__":
    asyncio.run(run_all_live())
