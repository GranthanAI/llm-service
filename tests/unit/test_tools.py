"""
Unit tests for Generic Tool Framework (BaseTool, Registry, Dispatcher, Executor, Validator, Normalizer, WebSearchTool).
Implements Phase 9 deliverables verification.
"""

import asyncio

import pytest

from app.exceptions.tool import RequiredToolFailedError, UnknownToolError
from app.models.execution_plan import ToolCall
from app.models.tool import (
    ToolParams,
    ToolResult,
    ToolSchema,
    ValidationResult,
)
from app.tools.base import BaseTool
from app.tools.dispatcher import ToolDispatcher
from app.tools.executor import ToolExecutor
from app.tools.normalizer import ToolNormalizer
from app.tools.registry import ToolRegistry
from app.tools.validator import ToolValidator
from app.tools.web_search import WebSearchTool


class MockSuccessTool(BaseTool):
    """Mock tool returning success."""

    name: str = "mock_success"
    description: str = "A mock tool that succeeds"
    timeout_ms: int = 2000

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        )

    def validate_params(self, params: ToolParams) -> ValidationResult:
        return ToolValidator.validate_required_fields(params, ["value"])

    async def execute(self, params: ToolParams) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"result": f"processed_{params.params.get('value')}"},
        )


class MockFailingTool(BaseTool):
    """Mock tool returning failure or raising exception."""

    name: str = "mock_fail"
    description: str = "A mock tool that fails"
    timeout_ms: int = 2000

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description)

    async def execute(self, params: ToolParams) -> ToolResult:
        raise ValueError("Simulated tool crash")


class MockSlowTool(BaseTool):
    """Mock tool exceeding timeout."""

    name: str = "mock_slow"
    description: str = "A mock tool that hangs"
    timeout_ms: int = 50  # 50ms

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description)

    async def execute(self, params: ToolParams) -> ToolResult:
        await asyncio.sleep(0.2)  # Sleep 200ms
        return ToolResult(tool_name=self.name, success=True)


# --- 1. Validator Tests ---


def test_tool_validator():
    """Verify ToolValidator checks required fields and types."""
    # Required fields
    params_valid = ToolParams(tool_name="test", params={"q": "query", "limit": 10})
    res = ToolValidator.validate_required_fields(params_valid, ["q", "limit"])
    assert res.valid is True

    params_missing = ToolParams(tool_name="test", params={"limit": 10})
    res_missing = ToolValidator.validate_required_fields(params_missing, ["q", "limit"])
    assert res_missing.valid is False
    assert "Missing required parameter(s): q" in res_missing.error

    # Types
    res_types_valid = ToolValidator.validate_field_types(params_valid, {"q": str, "limit": int})
    assert res_types_valid.valid is True

    res_types_invalid = ToolValidator.validate_field_types(params_valid, {"limit": str})
    assert res_types_invalid.valid is False


# --- 2. Registry Tests ---


def test_tool_registry():
    """Verify ToolRegistry registration, retrieval, and enablement."""
    registry = ToolRegistry()
    tool = MockSuccessTool()

    registry.register(tool, enabled=True)
    assert registry.is_enabled("mock_success") is True
    assert registry.get("mock_success") == tool

    schemas = registry.list_tools()
    assert len(schemas) == 1
    assert schemas[0].name == "mock_success"

    # Disable
    registry.disable("mock_success")
    assert registry.is_enabled("mock_success") is False
    with pytest.raises(UnknownToolError, match="disabled"):
        registry.get("mock_success")

    # Unregistered tool
    with pytest.raises(UnknownToolError, match="not registered"):
        registry.get("nonexistent")


# --- 3. Executor Tests ---


@pytest.mark.asyncio
async def test_tool_executor_success_and_timeout():
    """Verify ToolExecutor handles success, validation failure, and timeout limits."""
    executor = ToolExecutor()
    tool = MockSuccessTool()

    # Success
    res = await executor.execute(
        tool, ToolParams(tool_name="mock_success", params={"value": "foo"})
    )
    assert res.success is True
    assert res.data == {"result": "processed_foo"}
    assert res.latency_ms >= 0.0

    # Validation failure
    res_invalid = await executor.execute(tool, ToolParams(tool_name="mock_success", params={}))
    assert res_invalid.success is False
    assert "Validation failed" in res_invalid.error

    # Timeout
    slow_tool = MockSlowTool()
    res_timeout = await executor.execute(slow_tool, ToolParams(tool_name="mock_slow", params={}))
    assert res_timeout.success is False
    assert "timed out" in res_timeout.error
    assert res_timeout.metadata.get("timed_out") is True


# --- 4. Dispatcher Tests ---


@pytest.mark.asyncio
async def test_tool_dispatcher_parallel_and_sequential():
    """Verify ToolDispatcher parallel gathering, sequential ordering, and required error escalation."""
    registry = ToolRegistry()
    registry.register(MockSuccessTool())
    registry.register(MockFailingTool())
    dispatcher = ToolDispatcher(registry=registry)

    # 1. Parallel execution of multiple success tools
    calls = [
        ToolCall(tool_name="mock_success", params={"value": "1"}, parallel=True),
        ToolCall(tool_name="mock_success", params={"value": "2"}, parallel=True),
        ToolCall(tool_name="mock_success", params={"value": "3"}, parallel=False),
    ]
    results = await dispatcher.dispatch(calls)
    assert len(results) == 3
    assert all(r.success for r in results)

    # 2. Non-required failing tool does not raise
    failing_call = [ToolCall(tool_name="mock_fail", params={}, parallel=True, required=False)]
    fail_results = await dispatcher.dispatch(failing_call)
    assert len(fail_results) == 1
    assert fail_results[0].success is False

    # 3. Required failing tool raises RequiredToolFailedError
    required_fail_call = [ToolCall(tool_name="mock_fail", params={}, parallel=False, required=True)]
    with pytest.raises(RequiredToolFailedError, match="mock_fail"):
        await dispatcher.dispatch(required_fail_call)


# --- 5. WebSearchTool Tests ---


@pytest.mark.asyncio
async def test_web_search_tool_deduplication_and_caching():
    """Verify WebSearchTool deduplicates by domain and uses TTL cache."""
    search_tool = WebSearchTool(api_key="", cache_ttl_seconds=60)

    # Test domain deduplication method directly
    raw_results = [
        {"title": "Doc 1", "url": "https://fastapi.tiangolo.com/tutorial/", "snippet": "Snippet 1"},
        {"title": "Doc 2", "url": "https://fastapi.tiangolo.com/advanced/", "snippet": "Snippet 2"},
        {"title": "Python", "url": "https://python.org", "snippet": "Python home"},
    ]
    deduped = search_tool._deduplicate_by_domain(raw_results)
    assert len(deduped) == 2  # Only one fastapi.tiangolo.com entry preserved
    assert deduped[0]["title"] == "Doc 1"
    assert deduped[1]["title"] == "Python"

    # Execution with caching
    params = ToolParams(tool_name="web_search", params={"query": "FastAPI async"})
    res1 = await search_tool.execute(params)
    assert res1.success is True
    assert res1.data["type"] == "web_search"
    assert res1.metadata["cache_hit"] is False

    # Second call should hit cache
    res2 = await search_tool.execute(params)
    assert res2.success is True
    assert res2.metadata["cache_hit"] is True


# --- 6. Tool Normalizer Tests ---


def test_tool_normalizer():
    """Verify ToolNormalizer outputs uniform schema."""
    res = ToolNormalizer.normalize_web_search(
        query="test query",
        results=[{"title": "T", "url": "U", "snippet": "S"}],
        latency_ms=12.5,
    )
    assert res.tool_name == "web_search"
    assert res.success is True
    assert res.data["type"] == "web_search"
    assert res.data["query"] == "test query"
    assert len(res.data["results"]) == 1
    assert res.latency_ms == 12.5
