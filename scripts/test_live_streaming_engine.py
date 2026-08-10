"""
Live Verification Script for Phase 16: Streaming Engine.
Streams live tokens from Generation Router through StreamingEngine, validating dynamic chunk buffering,
Kafka chunk events generation, TTFT measurement, and final response assembly.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

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
from app.models.execution_plan import ExecutionPlan, IntentCategory, ReasoningMode, Skill, UserMode
from app.models.pipeline_context import PipelineContext
from app.models.response import (
    ChatResponseChunkEvent,
)
from app.models.tool import ToolResult
from app.producers.kafka_producer import KafkaPublisher
from app.prompts.builder import PromptBuilder
from app.prompts.loader import PromptLoader
from app.prompts.registry import PromptRegistry
from app.providers.gemini import GeminiAdapter
from app.providers.nvidia import NVIDIAAdapter
from app.providers.router import GenerationRouter
from app.services.streaming_service import StreamingEngine
from app.utils.cancellation import CancellationToken
from app.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from app.workflow_engine.workflow_result import WorkflowResult


async def main():
    print("\n" + "=" * 70)
    print("LIVE VERIFICATION: PHASE 16 STREAMING ENGINE & KAFKA CHUNKS")
    print("=" * 70 + "\n")

    config = get_settings()

    # 1. Initialize Adapters & Router
    print("Stage 1: Initializing Generation Router with Config-driven settings...")
    nvidia_adapter = NVIDIAAdapter(
        api_key=config.nvidia_api_key.get_secret_value(),
        model=config.nvidia_model,
        base_url=config.nvidia_base_url,
        timeout_s=config.nvidia_timeout_ms / 1000.0,
        temperature=config.nvidia_temperature,
        top_p=config.nvidia_top_p,
        max_tokens=config.nvidia_max_tokens,
    )
    gemini_adapter = GeminiAdapter(
        api_key=config.gemini_api_key.get_secret_value(),
        model=config.gemini_model,
    )
    cb_nvidia = CircuitBreaker(
        "nvidia", CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=10)
    )
    cb_gemini = CircuitBreaker(
        "gemini", CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=10)
    )
    router = GenerationRouter(
        nvidia_adapter=nvidia_adapter,
        gemini_adapter=gemini_adapter,
        circuit_breakers={"nvidia": cb_nvidia, "gemini": cb_gemini},
    )

    # 2. Build Pipeline Context & Composed Prompt
    print("\nStage 2: Composing prompt and pipeline context...")
    loader = PromptLoader()
    registry = PromptRegistry(loader=loader)
    builder = PromptBuilder(registry=registry)
    cwm = ContextWindowManager()

    bundle = ContextBundle(
        memory=MemoryContext(
            long_term_facts=[
                Fact(
                    fact_id="f1",
                    statement="User is an AI engineer evaluating streaming inference pipelines",
                    confidence=0.98,
                ),
            ],
            short_term_messages=[
                Message(
                    role="user",
                    content="Why is TTFT critical for AI chat applications?",
                ),
                Message(
                    role="assistant",
                    content="TTFT determines perceived latency and interactive responsiveness for end users.",
                ),
            ],
        ),
        graph=GraphContext(
            entities=[
                EntityNode(id="e1", name="StreamingEngine", type="Component"),
                EntityNode(id="e2", name="TTFTMetric", type="Telemetry"),
            ],
            relationships=[
                RelationshipEdge(
                    source_id="StreamingEngine",
                    relation_type="records",
                    target_id="TTFTMetric",
                ),
            ],
        ),
        retrieval=RetrievalContext(
            chunks=[
                DocumentChunk(
                    chunk_id="chk_stream_1",
                    file_id="streaming_spec.pdf",
                    content="Streaming engines must emit the first token with sub-50ms latency and balance subsequent chunk sizes.",
                    score=0.96,
                )
            ]
        ),
    )

    ctx = PipelineContext(
        request_id="req_stream_test_001",
        message_id="msg_stream_test_001",
        conversation_id="conv_stream_test_001",
        user_id="Elena",
        user_message="Explain in 3 numbered points why low TTFT and token chunk streaming matter in interactive LLMs.",
        context_bundle=bundle,
        selected_provider="nvidia",
        engine_type="langgraph",
        plan=ExecutionPlan(
            mode=UserMode.SMART,
            engine_type="langgraph",
            intent=IntentCategory.QUESTION_ANSWERING,
            skill=Skill.RESEARCH,
            reasoning_mode=ReasoningMode.REACT,
        ),
    )

    wf_res = WorkflowResult(
        mode="smart",
        engine_type="langgraph",
        user_message="Explain in 3 numbered points why low TTFT and token chunk streaming matter in interactive LLMs.",
        tool_outputs=[
            ToolResult(
                tool_name="web_search",
                success=True,
                data={"query": "LLM streaming TTFT user experience benchmarks"},
            )
        ],
        draft_content="Low TTFT provides immediate feedback while token chunking prevents UI stutter.",
    )

    composed_prompt = builder.build(wf_res, ctx)
    trimmed_prompt = cwm.manage(composed_prompt, target_model=config.nvidia_model)
    print(
        f"   Composed Tokens: {composed_prompt.total_tokens} -> Managed Tokens: {trimmed_prompt.total_tokens}"
    )

    # 3. Setup StreamingEngine with Captured Kafka Publisher
    published_chunks: list[ChatResponseChunkEvent] = []
    published_responses: list[tuple[PipelineContext, str]] = []
    published_memory_updates: list[tuple[PipelineContext, str]] = []

    mock_publisher = MagicMock(spec=KafkaPublisher)

    async def capture_chunk(event: ChatResponseChunkEvent):
        published_chunks.append(event)
        # Print chunk in real-time as it would appear to end user
        print(event.content, end="", flush=True)

    async def capture_response(context: PipelineContext, full_content: str):
        published_responses.append((context, full_content))

    async def capture_memory(context: PipelineContext, full_content: str):
        published_memory_updates.append((context, full_content))

    mock_publisher.publish_chunk = AsyncMock(side_effect=capture_chunk)
    mock_publisher.publish_response = AsyncMock(side_effect=capture_response)
    mock_publisher.publish_memory_update = AsyncMock(side_effect=capture_memory)

    engine = StreamingEngine(
        publisher=mock_publisher,
        first_chunk_size=1,
        default_chunk_size=6,
        code_block_chunk_size=20,
        flush_interval_ms=50,
    )

    # 4. Execute Live Stream through StreamingEngine
    print("\nStage 3: Streaming Live LLM Response via StreamingEngine...")
    print("-" * 70)
    t0 = time.perf_counter()
    live_token_stream = router.generate_stream(trimmed_prompt)
    full_response = await engine.stream(live_token_stream, ctx)
    stream_duration = time.perf_counter() - t0
    print("\n" + "-" * 70)

    # 5. Analyze Results & Telemetry
    print("\nStage 4: Telemetry & Streaming Validation:")
    print(f"   Total Streaming Latency: {stream_duration:.3f}s")
    print(f"   Total Chunks Published:  {len(published_chunks)}")
    print(
        f"   First Chunk Token:       '{published_chunks[0].content}' (Index={published_chunks[0].chunk_index}, Seq={published_chunks[0].sequence_number})"
    )
    print(f"   Last Chunk Flag (is_last): {published_chunks[-1].is_last}")
    if ctx.first_chunk_at and ctx.inference_started_at:
        ttft_ms = (ctx.first_chunk_at - ctx.inference_started_at).total_seconds() * 1000.0
        print(f"   Time-To-First-Token (TTFT): {ttft_ms:.2f}ms")

    assert len(published_chunks) > 0, "No chunks were published!"
    assert len(published_responses) == 1, "Final response event not published!"
    assert len(published_memory_updates) == 1, "Memory update event not published!"
    assert full_response == "".join(c.content for c in published_chunks), "Chunk assembly mismatch!"

    # 6. Test Cancellation Handling
    print("\nStage 5: Testing Cooperative Stream Cancellation...")
    cancel_token = CancellationToken()
    published_cancels = []

    async def capture_cancel(context: PipelineContext, reason: str = ""):
        published_cancels.append((context, reason))

    mock_publisher.publish_cancellation = AsyncMock(side_effect=capture_cancel)

    async def infinite_tokens():
        yield "Starting stream... "
        yield "Chunk 1 "
        cancel_token.cancel(reason="user_stop_button")
        yield "Should not be processed"

    partial = await engine.stream(infinite_tokens(), ctx, cancellation_token=cancel_token)
    print(f"   Cancellation Triggered: {cancel_token.is_cancelled} (Reason: {cancel_token.reason})")
    print(f"   Cancelled Event Emitted: {len(published_cancels) == 1}")
    print(f"   Partial Content Retained: '{partial}'")

    print("\n" + "=" * 70)
    print("STREAMING ENGINE LIVE VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
