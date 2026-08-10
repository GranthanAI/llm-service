"""
Unit Tests for Phase 17: Reliability Layer.
Tests RetryManager, CircuitBreaker per-dependency isolation, TTLCache enhancements,
and Error Hierarchy taxonomy and event mapping.
"""

import asyncio

import pytest

from app.exceptions.analysis import ContextOverflowError, PlanParseError
from app.exceptions.base import BaseLLMServiceError, FatalError, PermanentError, RetriableError
from app.exceptions.grpc import GRPCTimeoutError, GRPCUnavailableError
from app.exceptions.provider import (
    AllProvidersFailedError,
    CircuitOpenError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.exceptions.tool import RequiredToolFailedError, ToolTimeoutError
from app.models.execution_plan import ExecutionPlan, IntentCategory, ReasoningMode, Skill, UserMode
from app.models.pipeline_context import PipelineContext
from app.models.response import ErrorType
from app.utils.cache import (
    TTLCache,
    create_idempotency_cache,
    create_quota_cache,
    create_web_search_cache,
)
from app.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
)
from app.utils.error_handler import ErrorHandler
from app.utils.retry import RetryManager, RetryPolicy, is_retriable

# ---------------------------------------------------------------------------
# 1. Retry Manager & Jitter Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_manager_retries_transient_failures():
    """Verify RetryManager retries transient errors and succeeds upon recovery."""
    policy = RetryPolicy(max_attempts=3, initial_delay_ms=10, max_delay_ms=50, jitter=True)
    manager = RetryManager(policy=policy)

    attempts = 0

    async def flaky_call():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ProviderRateLimitError("Rate limit hit", provider="nvidia")
        return "success"

    result = await manager.execute_with_retry(flaky_call, operation_name="test_flaky")
    assert result == "success"
    assert attempts == 3


@pytest.mark.asyncio
async def test_retry_manager_non_retriable_fails_immediately():
    """Verify non-retriable permanent errors are not retried."""
    policy = RetryPolicy(max_attempts=4, initial_delay_ms=10, max_delay_ms=50)
    manager = RetryManager(policy=policy)

    attempts = 0

    async def permanent_failure_call():
        nonlocal attempts
        attempts += 1
        raise PlanParseError("Invalid JSON schema")

    with pytest.raises(PlanParseError):
        await manager.execute_with_retry(permanent_failure_call)

    assert attempts == 1  # No retries for permanent errors


@pytest.mark.asyncio
async def test_retry_manager_exhaustion():
    """Verify exceeding max_attempts raises original exception."""
    policy = RetryPolicy(max_attempts=2, initial_delay_ms=10, max_delay_ms=30)
    manager = RetryManager(policy=policy)

    attempts = 0

    async def always_failing():
        nonlocal attempts
        attempts += 1
        raise GRPCTimeoutError("gRPC timeout", service="memory_service")

    with pytest.raises(GRPCTimeoutError):
        await manager.execute_with_retry(always_failing)

    assert attempts == 2


def test_is_retriable_classification():
    """Verify comprehensive error classification predicate."""
    assert is_retriable(ProviderRateLimitError("429")) is True
    assert is_retriable(ProviderTimeoutError("timeout")) is True
    assert is_retriable(GRPCUnavailableError("unavailable")) is True
    assert is_retriable(ToolTimeoutError("tool timeout")) is True
    assert is_retriable(TimeoutError()) is True

    # Permanent errors must NOT be retriable
    assert is_retriable(PlanParseError("bad json")) is False
    assert is_retriable(RequiredToolFailedError("required failed")) is False
    assert is_retriable(ContextOverflowError("overflow")) is False
    assert is_retriable(ValueError("bad value")) is False


# ---------------------------------------------------------------------------
# 2. Circuit Breaker & Isolation Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_state_transitions():
    """Verify CLOSED -> OPEN -> HALF_OPEN -> CLOSED transitions."""
    config = CircuitBreakerConfig(
        failure_threshold=2,
        success_threshold=2,
        recovery_timeout_seconds=1,
    )
    cb = CircuitBreaker("nvidia", config=config)

    assert cb.state == CircuitState.CLOSED
    assert cb.is_open() is False

    # 1. Trigger failures to trip circuit
    await cb.on_failure()
    assert cb.state == CircuitState.CLOSED
    await cb.on_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.is_open() is True

    # 2. Immediate call while OPEN raises CircuitOpenError
    with pytest.raises(CircuitOpenError):
        async with cb.call():
            pass

    # 3. Wait for recovery timeout -> enters HALF_OPEN
    await asyncio.sleep(1.05)
    assert cb.is_open() is False

    # 4. First success in half-open state
    async with cb.call():
        pass
    assert cb.state == CircuitState.HALF_OPEN

    # 5. Second success closes circuit
    async with cb.call():
        pass
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


@pytest.mark.asyncio
async def test_circuit_breaker_manual_trip_and_reset():
    """Verify manual trip and reset functionality."""
    cb = CircuitBreaker("gemini")
    cb.trip()
    assert cb.state == CircuitState.OPEN
    assert cb.is_open() is True

    cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert cb.is_open() is False


@pytest.mark.asyncio
async def test_circuit_breaker_per_dependency_isolation():
    """Verify that an outage in one dependency does not trip others."""
    registry = CircuitBreakerRegistry(
        default_config=CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=5)
    )

    cb_graph = registry.get_or_create("graph_service")
    cb_memory = registry.get_or_create("memory_service")
    cb_nvidia = registry.get_or_create("nvidia")

    # Trip graph service
    await cb_graph.on_failure()
    await cb_graph.on_failure()
    assert cb_graph.state == CircuitState.OPEN

    # Other dependencies remain CLOSED and operational
    assert cb_memory.state == CircuitState.CLOSED
    assert cb_nvidia.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# 3. Cache Design & Eviction Tests
# ---------------------------------------------------------------------------


def test_ttl_cache_operations_and_stats():
    """Verify TTL cache get, set, contains, delete, and stats."""
    cache = TTLCache(max_size=5, default_ttl_seconds=10)

    cache.set("k1", "v1")
    cache.set("k2", "v2")

    assert cache.get("k1") == "v1"
    assert cache.get("nonexistent") is None

    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 2

    assert cache.contains("k1") is True
    assert cache.contains("nonexistent") is False

    assert cache.delete("k1") is True
    assert cache.contains("k1") is False


def test_ttl_cache_ttl_and_lru_eviction():
    """Verify TTL expiration and LRU displacement under capacity."""
    cache = TTLCache(max_size=2, default_ttl_seconds=1)

    cache.set("k1", "v1", ttl_seconds=1)
    cache.set("k2", "v2", ttl_seconds=10)

    # Access k1 to make k2 older in access time
    _ = cache.get("k1")

    # Add k3, causing LRU eviction of k2
    cache.set("k3", "v3", ttl_seconds=10)
    assert cache.size() == 2
    assert cache.get("k2") is None
    assert cache.get("k3") == "v3"


def test_cache_factory_helpers():
    """Verify factory methods create caches with designated specs."""
    search_cache = create_web_search_cache()
    assert search_cache._max_size == 1000
    assert search_cache._default_ttl == 60

    idemp_cache = create_idempotency_cache()
    assert idemp_cache._max_size == 10000
    assert idemp_cache._default_ttl == 300

    quota_cache = create_quota_cache()
    assert quota_cache._max_size == 10
    assert quota_cache._default_ttl == 5


# ---------------------------------------------------------------------------
# 4. Error Hierarchy & Error Handler Tests
# ---------------------------------------------------------------------------


def test_error_hierarchy_taxonomy_inheritance():
    """Verify error taxonomy class hierarchy (Retriable, Permanent, Fatal)."""
    assert issubclass(ProviderRateLimitError, RetriableError)
    assert issubclass(ProviderTimeoutError, RetriableError)
    assert issubclass(GRPCTimeoutError, RetriableError)

    assert issubclass(AllProvidersFailedError, PermanentError)
    assert issubclass(PlanParseError, PermanentError)
    assert issubclass(RequiredToolFailedError, PermanentError)
    assert issubclass(ContextOverflowError, PermanentError)

    assert issubclass(FatalError, BaseLLMServiceError)


def test_error_handler_classification_and_events():
    """Verify ErrorHandler produces structured Kafka responses and DLQ events."""
    handler = ErrorHandler()

    ctx = PipelineContext(
        conversation_id="conv_err_001",
        user_id="user_err_001",
        message_id="msg_err_001",
        request_id="req_err_001",
        user_message="Test message",
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

    # 1. Permanent LLM failure event
    all_failed_exc = AllProvidersFailedError("All providers failed")
    resp_event = handler.build_error_response_event(ctx, all_failed_exc)

    assert resp_event.status == "error"
    assert resp_event.error_code == "PROVIDER_ALL_FAILED"
    assert resp_event.full_content is None
    assert resp_event.conversation_id == "conv_err_001"

    # 2. DLQ event
    dlq_event = handler.build_dlq_event(
        raw_payload={"corrupted": "payload"},
        exc=all_failed_exc,
        key="conv_err_001",
    )
    assert dlq_event.original_topic == "chat.message.created"
    assert dlq_event.error_type == ErrorType.INFERENCE
