"""
Live Verification Script for Phase 17: Reliability Layer.
Verifies RetryManager backoff, CircuitBreaker state transitions, TTLCache hit/miss rates,
and ErrorHandler error event generation.
"""

import asyncio
import time

from app.exceptions.provider import (
    AllProvidersFailedError,
    ProviderRateLimitError,
)
from app.models.execution_plan import ExecutionPlan, IntentCategory, ReasoningMode, Skill, UserMode
from app.models.pipeline_context import PipelineContext
from app.utils.cache import create_web_search_cache
from app.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
)
from app.utils.error_handler import ErrorHandler
from app.utils.retry import RetryManager, RetryPolicy


async def main():
    print("\n" + "=" * 70)
    print("LIVE VERIFICATION: PHASE 17 RELIABILITY LAYER")
    print("=" * 70 + "\n")

    # 1. Test Retry Manager with Exponential Backoff & Jitter
    print("Stage 1: Testing RetryManager Transient Fault Recovery...")
    policy = RetryPolicy(max_attempts=3, initial_delay_ms=50, max_delay_ms=200, jitter=True)
    retry_manager = RetryManager(policy=policy)

    transient_attempts = 0

    async def flaky_upstream():
        nonlocal transient_attempts
        transient_attempts += 1
        if transient_attempts < 3:
            raise ProviderRateLimitError(
                f"HTTP 429 Rate limit spike (attempt {transient_attempts})", provider="nvidia"
            )
        return {"status": "success", "content": "Upstream service recovered successfully."}

    t0 = time.perf_counter()
    result = await retry_manager.execute_with_retry(
        flaky_upstream, operation_name="nvidia_pre_stream"
    )
    elapsed = time.perf_counter() - t0
    print(f"   Attempts required: {transient_attempts}/3")
    print(f"   Backoff + Jitter Duration: {elapsed:.3f}s")
    print(f"   Operation Result: {result['content']}")
    assert transient_attempts == 3

    # 2. Test Circuit Breaker State Transitions and Half-Open Recovery
    print(
        "\nStage 2: Testing CircuitBreaker State Transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)..."
    )
    config = CircuitBreakerConfig(
        failure_threshold=2, success_threshold=2, recovery_timeout_seconds=1
    )
    cb = CircuitBreaker("nvidia_nim", config=config)

    print(f"   Initial State: {cb.state.value} (is_open={cb.is_open()})")
    await cb.on_failure()
    await cb.on_failure()
    print(f"   After 2 Failures: {cb.state.value} (is_open={cb.is_open()})")
    assert cb.state == CircuitState.OPEN

    print("   Waiting 1.1s for recovery timeout...")
    await asyncio.sleep(1.1)
    print(f"   Ready for Half-Open Probe: is_open={cb.is_open()}")

    async with cb.call():
        print("   -> Probe 1 Successful")
    print(f"   State after probe 1: {cb.state.value}")
    assert cb.state == CircuitState.HALF_OPEN

    async with cb.call():
        print("   -> Probe 2 Successful")
    print(f"   State after probe 2: {cb.state.value} (Fully Recovered)")
    assert cb.state == CircuitState.CLOSED

    # 3. Test Per-Dependency Isolation
    print("\nStage 3: Testing Per-Dependency Circuit Breaker Isolation...")
    registry = CircuitBreakerRegistry()
    cb_groq = registry.get_or_create("groq")
    cb_nvidia = registry.get_or_create("nvidia")
    cb_graph = registry.get_or_create("graph_service")

    # Trip groq only
    cb_groq.trip()
    print(f"   Groq Circuit: {cb_groq.state.value} (is_open={cb_groq.is_open()})")
    print(f"   NVIDIA Circuit: {cb_nvidia.state.value} (is_open={cb_nvidia.is_open()})")
    print(f"   Graph Service Circuit: {cb_graph.state.value} (is_open={cb_graph.is_open()})")
    assert cb_groq.is_open() is True
    assert cb_nvidia.is_open() is False
    assert cb_graph.is_open() is False

    # 4. Test TTLCache Acceleration & Eviction
    print("\nStage 4: Testing TTLCache Performance & Eviction...")
    cache = create_web_search_cache()
    query_hash = "sha256_mock_hash_query_llm_reliability"
    cache.set(
        query_hash,
        {
            "results": [
                "Reliability Layer Design Pattern",
                "Circuit Breakers in Distributed Systems",
            ]
        },
    )

    # Cache hit
    hit_val = cache.get(query_hash)
    miss_val = cache.get("nonexistent_key")
    stats = cache.get_stats()
    print(
        f"   Cache Hits: {stats['hits']}, Misses: {stats['misses']}, Current Size: {stats['size']}"
    )
    assert hit_val is not None
    assert miss_val is None

    # 5. Test Error Hierarchy & Kafka Error Event Generation
    print("\nStage 5: Testing Error Hierarchy Classification & Event Formatting...")
    handler = ErrorHandler()

    ctx = PipelineContext(
        conversation_id="conv_rel_live_001",
        user_id="Elena",
        message_id="msg_rel_live_001",
        request_id="req_rel_live_001",
        user_message="Summarize fault tolerance strategies",
        selected_provider="nvidia",
        engine_type="mode_handler",
        plan=ExecutionPlan(
            mode=UserMode.DEFAULT,
            engine_type="mode_handler",
            intent=IntentCategory.QUESTION_ANSWERING,
            skill=Skill.GENERAL_CHAT,
            reasoning_mode=ReasoningMode.DIRECT,
        ),
    )

    err = AllProvidersFailedError("All LLM providers unavailable", provider="nvidia")
    classification = handler.classify(err)
    print(f"   Error: {type(err).__name__}")
    print(
        f"   Taxonomy Classification: type={classification.error_type.value}, code={classification.error_code}, retriable={classification.is_retriable}"
    )

    event = handler.build_error_response_event(ctx, err)
    print(
        f"   Published Error Event: status='{event.status}', code='{event.error_code}', message='{event.error_message}'"
    )
    assert event.status == "error"
    assert event.error_code == "PROVIDER_ALL_FAILED"

    print("\n" + "=" * 70)
    print("RELIABILITY LAYER LIVE VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
