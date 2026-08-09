"""
Tool Execution Engine.
Implements LLD v2.0 Section 15.5.
"""

import asyncio
import time

import structlog

from app.config.logging import get_logger
from app.models.tool import ToolParams, ToolResult
from app.tools.base import BaseTool
from app.tools.normalizer import ToolNormalizer
from app.utils.tracing import trace_span


class ToolExecutor:
    """
    Executes individual tool calls with parameter validation,
    strict timeout enforcement via asyncio.wait_for, and latency telemetry.
    """

    def __init__(self, logger: structlog.stdlib.BoundLogger | None = None):
        self.logger = logger or get_logger("tool_executor")

    @trace_span("tool_executor_execute")
    async def execute(
        self,
        tool: BaseTool,
        params: ToolParams,
        timeout_ms: int | None = None,
    ) -> ToolResult:
        """Execute tool with parameter validation and timeout."""
        effective_timeout = timeout_ms or tool.timeout_ms
        start_time = time.perf_counter()

        self.logger.debug(
            "Executing tool",
            tool_name=tool.name,
            timeout_ms=effective_timeout,
            trace_id=params.trace_id,
            conversation_id=params.conversation_id,
        )

        # 1. Parameter Validation
        validation = tool.validate_params(params)
        if not validation.valid:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.logger.warning(
                "Tool parameter validation failed",
                tool_name=tool.name,
                error=validation.error,
            )
            return ToolNormalizer.normalize_error(
                tool_name=tool.name,
                error_message=f"Validation failed: {validation.error}",
                latency_ms=latency_ms,
            )

        # 2. Asynchronous Execution with Timeout Enforcement
        try:
            result: ToolResult = await asyncio.wait_for(
                tool.execute(params),
                timeout=effective_timeout / 1000.0,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            result.latency_ms = latency_ms

            self.logger.info(
                "Tool execution completed",
                tool_name=tool.name,
                success=result.success,
                latency_ms=round(latency_ms, 2),
            )
            return result

        except TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            error_msg = f"Tool timed out after {effective_timeout}ms"
            self.logger.warning(
                "Tool execution timed out",
                tool_name=tool.name,
                timeout_ms=effective_timeout,
            )
            return ToolNormalizer.normalize_error(
                tool_name=tool.name,
                error_message=error_msg,
                latency_ms=latency_ms,
                metadata={"timed_out": True},
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.logger.error(
                "Tool execution raised an unhandled exception",
                tool_name=tool.name,
                error=str(exc),
                latency_ms=round(latency_ms, 2),
                exc_info=True,
            )
            return ToolNormalizer.normalize_error(
                tool_name=tool.name,
                error_message=f"Tool error: {exc!s}",
                latency_ms=latency_ms,
            )
