"""
Tool Dispatcher.
Implements LLD v2.0 Section 15.3 and Section 15.4.
"""

import asyncio

import structlog

from app.config.logging import get_logger
from app.exceptions.tool import RequiredToolFailedError, UnknownToolError
from app.models.execution_plan import ToolCall
from app.models.tool import ToolParams, ToolResult
from app.tools.executor import ToolExecutor
from app.tools.normalizer import ToolNormalizer
from app.tools.registry import ToolRegistry
from app.utils.tracing import trace_span


class ToolDispatcher:
    """
    Orchestrates execution of ToolCalls requested by Mode Handlers or LangGraph nodes.
    Supports parallel gathering and sequential execution with failure enforcement.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        executor: ToolExecutor | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.registry: ToolRegistry = registry
        self.executor: ToolExecutor = executor or ToolExecutor()
        self.logger = logger or get_logger("tool_dispatcher")

    @trace_span("tool_dispatcher_dispatch")
    async def dispatch(
        self,
        tools: list[ToolCall],
        trace_id: str = "",
        user_id: str = "",
        conversation_id: str = "",
    ) -> list[ToolResult]:
        """
        Dispatch a collection of tool calls following the LLD Section 15.4 algorithm.
        """
        if not tools:
            return []

        self.logger.info(
            "Dispatching tool calls",
            tools_count=len(tools),
            tool_names=[t.tool_name for t in tools],
            conversation_id=conversation_id,
        )

        # 1. Partition into parallel and sequential execution groups
        parallel_group = [t for t in tools if t.parallel]
        sequential_group = [t for t in tools if not t.parallel]

        results: list[ToolResult] = []

        # 2. Execute parallel group concurrently
        if parallel_group:
            parallel_results = await self._dispatch_parallel(
                parallel_group, trace_id, user_id, conversation_id
            )
            results.extend(parallel_results)

        # 3. Execute sequential group in strict order
        if sequential_group:
            sequential_results = await self._dispatch_sequential(
                sequential_group, trace_id, user_id, conversation_id
            )
            results.extend(sequential_results)

        return results

    async def _dispatch_parallel(
        self,
        tools: list[ToolCall],
        trace_id: str,
        user_id: str,
        conversation_id: str,
    ) -> list[ToolResult]:
        """Execute parallel tools via asyncio.gather."""
        tasks = [self._execute_single_tool(t, trace_id, user_id, conversation_id) for t in tools]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[ToolResult] = []
        for tool_call, raw_res in zip(tools, raw_results, strict=False):
            if isinstance(raw_res, Exception):
                self.logger.error(
                    "Parallel tool invocation failed with exception",
                    tool_name=tool_call.tool_name,
                    error=str(raw_res),
                )
                if tool_call.required:
                    raise RequiredToolFailedError(
                        f"Required tool '{tool_call.tool_name}' failed: {raw_res!s}",
                        tool_name=tool_call.tool_name,
                    )
                results.append(
                    ToolNormalizer.normalize_error(
                        tool_name=tool_call.tool_name,
                        error_message=str(raw_res),
                    )
                )
            else:
                if not raw_res.success and tool_call.required:
                    self.logger.error(
                        "Required parallel tool execution reported failure",
                        tool_name=tool_call.tool_name,
                        error=raw_res.error,
                    )
                    raise RequiredToolFailedError(
                        f"Required tool '{tool_call.tool_name}' execution failed: {raw_res.error}",
                        tool_name=tool_call.tool_name,
                    )
                results.append(raw_res)

        return results

    async def _dispatch_sequential(
        self,
        tools: list[ToolCall],
        trace_id: str,
        user_id: str,
        conversation_id: str,
    ) -> list[ToolResult]:
        """Execute sequential tools in deterministic order."""
        results: list[ToolResult] = []
        for tool_call in tools:
            try:
                res = await self._execute_single_tool(tool_call, trace_id, user_id, conversation_id)
                if not res.success and tool_call.required:
                    self.logger.error(
                        "Required sequential tool execution failed",
                        tool_name=tool_call.tool_name,
                        error=res.error,
                    )
                    raise RequiredToolFailedError(
                        f"Required tool '{tool_call.tool_name}' execution failed: {res.error}",
                        tool_name=tool_call.tool_name,
                    )
                results.append(res)
            except Exception as exc:
                if tool_call.required:
                    raise RequiredToolFailedError(
                        f"Required tool '{tool_call.tool_name}' raised: {exc!s}",
                        tool_name=tool_call.tool_name,
                    ) from exc
                results.append(
                    ToolNormalizer.normalize_error(
                        tool_name=tool_call.tool_name,
                        error_message=str(exc),
                    )
                )
        return results

    async def _execute_single_tool(
        self,
        tool_call: ToolCall,
        trace_id: str,
        user_id: str,
        conversation_id: str,
    ) -> ToolResult:
        """Fetch tool from registry and delegate to executor."""
        try:
            tool = self.registry.get(tool_call.tool_name)
        except UnknownToolError as exc:
            self.logger.warning(
                "Unknown or disabled tool requested",
                tool_name=tool_call.tool_name,
                error=str(exc),
            )
            return ToolNormalizer.normalize_error(
                tool_name=tool_call.tool_name,
                error_message=str(exc),
            )

        params = ToolParams(
            tool_name=tool_call.tool_name,
            params=tool_call.params,
            trace_id=trace_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return await self.executor.execute(tool=tool, params=params)
