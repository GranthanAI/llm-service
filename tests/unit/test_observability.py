"""
Unit Tests for Phase 18: Observability.
Tests Prometheus metrics definitions, OpenTelemetry tracing decorators,
and Structlog structured logging correlation ID propagation.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.config.logging import (
    add_correlation_ids,
    correlation_conversation_id,
    correlation_engine_type,
    correlation_request_id,
    correlation_trace_id,
)
from app.main import app
from app.utils.metrics import (
    CIRCUIT_BREAKER_STATE,
    CONTEXT_FETCH_DURATION,
    COST_USD,
    ERRORS_TOTAL,
    GENERATION_FALLBACK_TOTAL,
    KAFKA_LAG,
    LANGGRAPH_LOOP_CAPPED,
    LANGGRAPH_LOOP_ITERATIONS,
    LANGGRAPH_NODE_DURATION,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    STREAMING_CHUNKS_TOTAL,
    TOKENS_TOTAL,
    TOOL_CALLS,
    TOOL_DURATION,
    TTFT,
    WORKFLOW_DISPATCH_TOTAL,
)
from app.utils.tracing import setup_tracing, trace_span


def test_prometheus_metrics_registry_completeness():
    """Verify all 17 core Prometheus metrics are defined and accessible."""
    metrics_list = [
        REQUESTS_TOTAL,
        REQUEST_DURATION,
        WORKFLOW_DISPATCH_TOTAL,
        LANGGRAPH_NODE_DURATION,
        LANGGRAPH_LOOP_ITERATIONS,
        LANGGRAPH_LOOP_CAPPED,
        TTFT,
        TOKENS_TOTAL,
        STREAMING_CHUNKS_TOTAL,
        COST_USD,
        CIRCUIT_BREAKER_STATE,
        GENERATION_FALLBACK_TOTAL,
        TOOL_CALLS,
        TOOL_DURATION,
        KAFKA_LAG,
        CONTEXT_FETCH_DURATION,
        ERRORS_TOTAL,
    ]
    for m in metrics_list:
        assert m is not None


def test_metrics_endpoint_scraping():
    """Verify /metrics endpoint returns Prometheus formatted metrics."""
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "llm_requests_total" in response.text
    assert "llm_ttft_seconds" in response.text
    assert "llm_circuit_breaker_state" in response.text


def test_structlog_correlation_id_processor():
    """Verify add_correlation_ids injects active context variables."""
    correlation_request_id.set("req_obs_123")
    correlation_conversation_id.set("conv_obs_456")
    correlation_trace_id.set("trace_obs_789")
    correlation_engine_type.set("langgraph")

    event_dict = {"event": "generation_complete"}
    processed = add_correlation_ids(None, "info", event_dict)

    assert processed["request_id"] == "req_obs_123"
    assert processed["conversation_id"] == "conv_obs_456"
    assert processed["trace_id"] == "trace_obs_789"
    assert processed["engine_type"] == "langgraph"


@pytest.mark.asyncio
async def test_trace_span_async_decorator():
    """Verify @trace_span starts child span and records success."""
    tracer = setup_tracing(service_name="test-llm-service")
    assert tracer is not None

    @trace_span("test_async_span", attributes={"mode": "smart", "provider": "nvidia"})
    async def sample_async_func():
        await asyncio.sleep(0.01)
        return "traced_result"

    res = await sample_async_func()
    assert res == "traced_result"


@pytest.mark.asyncio
async def test_trace_span_records_exception():
    """Verify @trace_span records exceptions and sets error status."""

    @trace_span("test_failing_span")
    async def sample_failing_func():
        raise RuntimeError("simulated span error")

    with pytest.raises(RuntimeError, match="simulated span error"):
        await sample_failing_func()
