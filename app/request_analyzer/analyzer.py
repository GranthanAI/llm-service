"""
Request Analyzer Engine.
Implements LLD v2.0 Section 10 and HLD v2.0 Section 9.
"""

import json
import time
from typing import Any

import structlog

from app.config.logging import get_logger
from app.config.settings import LLMServiceConfig, get_settings
from app.models.execution_plan import ExecutionPlan, UserMode
from app.models.pipeline_context import PipelineContext
from app.request_analyzer.groq_client import GroqAnalysisClient
from app.request_analyzer.prompt_template import AnalysisPromptBuilder
from app.request_analyzer.safe_default import SafeDefaultFactory
from app.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from app.utils.retry import RetryManager, RetryPolicy
from app.utils.tracing import trace_span


class RequestAnalyzer:
    """
    Unified Request Analysis and Planning Engine.
    Executes Call 1 (Groq JSON mode) to produce authoritative ExecutionPlan.
    Applies safety overrides and falls back to SafeDefaultPlan if Groq fails.
    """

    def __init__(
        self,
        groq_client: GroqAnalysisClient,
        prompt_builder: AnalysisPromptBuilder | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        retry_manager: RetryManager | None = None,
        config: LLMServiceConfig | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.groq_client: GroqAnalysisClient = groq_client
        self.prompt_builder: AnalysisPromptBuilder = prompt_builder or AnalysisPromptBuilder()
        self.circuit_breaker: CircuitBreaker = circuit_breaker or CircuitBreaker(
            name="groq",
            config=CircuitBreakerConfig(failure_threshold=5, recovery_timeout_seconds=30),
        )
        self.retry_manager: RetryManager = retry_manager or RetryManager(
            policy=RetryPolicy(max_attempts=2, initial_delay_ms=50, max_delay_ms=200)
        )
        self.config: LLMServiceConfig = config or get_settings()
        self.logger = logger or get_logger("request_analyzer")

    @trace_span("request_analyzer_analyze")
    async def analyze(
        self,
        ctx: PipelineContext,
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> ExecutionPlan:
        """
        Analyze request context and produce ExecutionPlan.
        Guaranteed to return a valid ExecutionPlan (never raises).
        """
        start_time = time.perf_counter()
        messages = self.prompt_builder.build_prompt(ctx, tools_schema)

        try:
            # Execute Groq call protected by Circuit Breaker and Retries
            async with self.circuit_breaker.call():

                async def _call_groq():
                    return await self.groq_client.complete(messages=messages)

                raw_json = await self.retry_manager.execute_with_retry(
                    _call_groq,
                    operation_name="groq_request_analysis",
                    provider="groq",
                )

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Parse JSON and validate schema
            plan_dict = json.loads(raw_json)
            plan = ExecutionPlan.model_validate(plan_dict)
            plan.groq_model_used = self.groq_client.model
            plan.analysis_latency_ms = latency_ms

            # Apply strict safety overrides (LLD v2.0 Section 10.3)
            plan = self._apply_safety_overrides(plan, ctx)

            self.logger.info(
                "Request analysis completed",
                mode=plan.mode.value,
                intent=plan.intent.value,
                skill=plan.skill.value,
                tools_count=len(plan.tools),
                latency_ms=round(latency_ms, 2),
            )
            return plan

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.logger.warning(
                "Request analysis failed — falling back to SafeDefaultPlan",
                error=str(exc),
                latency_ms=round(latency_ms, 2),
                conversation_id=ctx.conversation_id,
            )
            fallback_plan = SafeDefaultFactory.create(
                latency_ms=latency_ms, reason=type(exc).__name__
            )
            return fallback_plan

    def _apply_safety_overrides(self, plan: ExecutionPlan, ctx: PipelineContext) -> ExecutionPlan:
        """
        Apply deterministic safety rules per LLD v2.0 Section 10.3:
        1. Mode ask_files without file_ids -> default
        2. Web search disabled in config -> strip web_search tool
        3. Smart mode iteration cap
        4. Deep Research mode iteration cap
        5. Non-iterative modes iteration cap = 1
        """
        # Override Rule 1: Cannot ask files without attached file_ids
        if plan.mode == UserMode.ASK_FILES and len(ctx.file_ids) == 0:
            self.logger.info(
                "Safety override applied: ask_files with 0 files reverted to default mode"
            )
            plan.mode = UserMode.DEFAULT

        # Override Rule 2: Web search feature flag enforcement
        if not self.config.enable_web_search and plan.tools:
            plan.tools = [t for t in plan.tools if t.tool_name != "web_search"]

        # Override Rule 3: Smart graph iteration cap
        if plan.mode == UserMode.SMART:
            if plan.max_iterations > self.config.langgraph_max_loop_iterations_smart:
                plan.max_iterations = self.config.langgraph_max_loop_iterations_smart

        # Override Rule 4: Deep Research graph iteration cap
        elif plan.mode == UserMode.DEEP_RESEARCH:
            if plan.max_iterations > self.config.langgraph_max_loop_iterations_deep_research:
                plan.max_iterations = self.config.langgraph_max_loop_iterations_deep_research

        # Override Rule 5: Mode Handlers are strictly single-pass by construction
        else:
            plan.max_iterations = 1

        return plan
