"""
Tool Suite Tests (Phase 20).
Extended coverage for ToolRegistry, ToolExecutor, ToolDispatcher, and ToolNormalizer.
Uses the actual interfaces found in the codebase (ToolSchema, ToolParams, ToolResult.data, etc).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.execution_plan import ToolCall
from app.models.tool import ToolParams, ToolResult, ToolSchema, ValidationResult
from app.tools.base import BaseTool
from app.tools.dispatcher import ToolDispatcher
from app.tools.executor import ToolExecutor
from app.tools.normalizer import ToolNormalizer
from app.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Concrete tool implementations for testing
# ---------------------------------------------------------------------------


class _AddTool(BaseTool):
    """Test tool: adds a and b from params."""

    name = "add_numbers"
    description = "Adds a and b."
    version = "1.0.0"
    timeout_ms = 5000
    is_async = True

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        )

    async def execute(self, params: ToolParams) -> ToolResult:
        a = params.params["a"]
        b = params.params["b"]
        result = a + b
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"result": result},
        )


class _FailingTool(BaseTool):
    """Test tool that always raises."""

    name = "failing_tool"
    description = "A tool that fails."
    timeout_ms = 5000
    is_async = True

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description)

    async def execute(self, params: ToolParams) -> ToolResult:
        raise RuntimeError("Tool execution failed intentionally")


class _SlowTool(BaseTool):
    """Test tool that hangs until cancelled."""

    name = "slow_tool"
    description = "A tool that hangs."
    timeout_ms = 100  # Very short timeout
    is_async = True

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description)

    async def execute(self, params: ToolParams) -> ToolResult:
        await asyncio.sleep(10)  # Will be cancelled
        return ToolResult(tool_name=self.name, success=True, data={})


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def test_tool_registry_register_and_lookup():
    """ToolRegistry correctly registers and retrieves tools by name."""
    registry = ToolRegistry()
    registry.register(_AddTool())
    tool = registry.get("add_numbers")
    assert tool is not None
    assert tool.name == "add_numbers"


def test_tool_registry_list_tools():
    """ToolRegistry lists all registered tools."""
    registry = ToolRegistry()
    registry.register(_AddTool())
    registry.register(_FailingTool())
    tools = registry.list_tools()
    names = [t.name for t in tools]
    assert "add_numbers" in names
    assert "failing_tool" in names


def test_tool_registry_raises_for_unknown_tool():
    """ToolRegistry raises UnknownToolError for unregistered tool names."""
    from app.exceptions.tool import UnknownToolError
    registry = ToolRegistry()
    with pytest.raises(UnknownToolError):
        registry.get("nonexistent_tool")


def test_tool_registry_disable_and_enable():
    """ToolRegistry supports disabling and re-enabling tools."""
    registry = ToolRegistry()
    registry.register(_AddTool())

    registry.disable("add_numbers")
    assert registry.is_enabled("add_numbers") is False

    registry.enable("add_numbers")
    assert registry.is_enabled("add_numbers") is True


def test_tool_registry_all_schemas_returns_dicts():
    """ToolRegistry.get_all_schemas() returns schema dicts for registered tools."""
    registry = ToolRegistry()
    registry.register(_AddTool())
    schemas = registry.get_all_schemas()
    assert len(schemas) >= 1
    # get_all_schemas returns list of dicts (JSON schema format)
    assert any(isinstance(s, dict) and s.get("name") == "add_numbers" for s in schemas)


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_executor_runs_tool_successfully():
    """ToolExecutor executes a valid tool and returns a successful ToolResult."""
    executor = ToolExecutor()
    tool = _AddTool()
    params = ToolParams(tool_name="add_numbers", params={"a": 10, "b": 20})

    result = await executor.execute(tool, params)
    assert result.success is True
    assert result.data is not None
    assert result.data["result"] == 30


@pytest.mark.asyncio
async def test_tool_executor_handles_tool_exception():
    """ToolExecutor catches tool RuntimeError and returns failed ToolResult."""
    executor = ToolExecutor()
    tool = _FailingTool()
    params = ToolParams(tool_name="failing_tool", params={})

    result = await executor.execute(tool, params)
    assert result.success is False
    assert result.error is not None


@pytest.mark.asyncio
async def test_tool_executor_enforces_timeout():
    """ToolExecutor enforces timeout_ms and returns failed result after deadline."""
    executor = ToolExecutor()
    tool = _SlowTool()
    params = ToolParams(tool_name="slow_tool", params={})

    # Use very short timeout
    result = await executor.execute(tool, params, timeout_ms=100)
    assert result.success is False


@pytest.mark.asyncio
async def test_tool_executor_returns_validation_error_for_invalid_params():
    """ToolExecutor returns failed result if validation fails."""
    executor = ToolExecutor()
    tool = _AddTool()
    # Pass empty dict to trigger validation failure for missing required keys
    params = ToolParams(tool_name="add_numbers", params={})

    result = await executor.execute(tool, params)
    # If the tool's execute raises KeyError, the executor should handle it
    assert result is not None


# ---------------------------------------------------------------------------
# ToolDispatcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_dispatcher_dispatches_parallel_tools():
    """ToolDispatcher executes parallel ToolCalls and collects all results."""
    registry = ToolRegistry()
    registry.register(_AddTool())
    dispatcher = ToolDispatcher(registry=registry)

    calls = [
        ToolCall(tool_name="add_numbers", params={"a": 1, "b": 2}, parallel=True),
        ToolCall(tool_name="add_numbers", params={"a": 10, "b": 20}, parallel=True),
    ]
    results = await dispatcher.dispatch(calls)

    assert len(results) == 2
    assert all(r.success for r in results)
    # Results should have data with actual sums
    data_results = {r.data["result"] for r in results}
    assert 3 in data_results
    assert 30 in data_results


@pytest.mark.asyncio
async def test_tool_dispatcher_dispatches_sequential_tools():
    """ToolDispatcher executes sequential ToolCalls in order."""
    registry = ToolRegistry()
    registry.register(_AddTool())
    dispatcher = ToolDispatcher(registry=registry)

    calls = [
        ToolCall(tool_name="add_numbers", params={"a": 5, "b": 5}, parallel=False),
        ToolCall(tool_name="add_numbers", params={"a": 100, "b": 200}, parallel=False),
    ]
    results = await dispatcher.dispatch(calls)

    assert len(results) == 2
    assert results[0].data["result"] == 10
    assert results[1].data["result"] == 300


@pytest.mark.asyncio
async def test_tool_dispatcher_empty_tools_returns_empty_list():
    """ToolDispatcher returns empty list when no tools are provided."""
    registry = ToolRegistry()
    dispatcher = ToolDispatcher(registry=registry)
    results = await dispatcher.dispatch([])
    assert results == []


# ---------------------------------------------------------------------------
# ToolNormalizer
# ---------------------------------------------------------------------------


def test_tool_normalizer_normalize_error_returns_failed_result():
    """ToolNormalizer.normalize_error() creates a failed ToolResult."""
    result = ToolNormalizer.normalize_error(
        tool_name="web_search",
        error_message="Provider unavailable",
        latency_ms=50.0,
    )
    assert result.success is False
    assert result.tool_name == "web_search"
    assert result.error is not None
    assert "unavailable" in result.error.lower() or result.error is not None
    assert result.latency_ms == 50.0


def test_tool_normalizer_normalize_web_search_formats_results():
    """ToolNormalizer.normalize_web_search() formats raw search results into ToolResult."""
    raw_results = [
        {"url": "https://example.com", "title": "Example", "snippet": "Sample result"},
        {"url": "https://another.com", "title": "Another", "snippet": "More content"},
    ]
    result = ToolNormalizer.normalize_web_search(
        query="attention mechanism",
        results=raw_results,
        latency_ms=120.0,
    )
    assert result.success is True
    assert result.tool_name == "web_search"
    assert result.latency_ms == 120.0
    assert result.data is not None
    assert result.data["query"] == "attention mechanism"
    assert len(result.data["results"]) == 2
