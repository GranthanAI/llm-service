"""
Request Analyzer Engine.
Implements LLD v2.0 Section 10 and HLD v2.0 Section 9.
"""

import asyncio
import json
import time
from typing import Any

import structlog

from app.config.logging import get_logger
from app.config.settings import LLMServiceConfig, get_settings
from app.models.execution_plan import ExecutionPlan, UserMode
from app.models.pipeline_context import PipelineContext
from app.providers.nvidia import NVIDIAAdapter
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
    If Groq fails, attempts the same analysis via NVIDIA (when configured) as
    an intermediate fallback before giving up to the heuristic SafeDefaultPlan.
    Groq remains the sole provider used for generation-router chat replies is
    untouched by this — this fallback applies only to the planning/analysis
    call, never to response generation.
    """

    def __init__(
        self,
        groq_client: GroqAnalysisClient,
        prompt_builder: AnalysisPromptBuilder | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        retry_manager: RetryManager | None = None,
        config: LLMServiceConfig | None = None,
        nvidia_adapter: NVIDIAAdapter | None = None,
        fallback_clients: list[GroqAnalysisClient] | None = None,
        fallback_adapters: list | None = None,
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
        self.nvidia_adapter: NVIDIAAdapter | None = nvidia_adapter
        # Other Groq models (same key) tried in order when the primary analysis model fails.
        self.fallback_clients: list[GroqAnalysisClient] = list(fallback_clients or [])
        # Non-Groq providers (e.g. Gemini) with an async execute(messages) -> response.content; tried after
        # the Groq models and before the slow NVIDIA last resort.
        self.fallback_adapters: list = list(fallback_adapters or [])
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
        Tries Groq first; if that fails and an NVIDIA adapter is configured,
        retries the same analysis via NVIDIA before giving up to SafeDefaultPlan.
        """
        start_time = time.perf_counter()
        messages = self.prompt_builder.build_prompt(ctx, tools_schema)
        groq_exc: Exception | None = None

        try:
            raw_json = await self._call_groq(messages)
            plan = self._finalize_plan(raw_json, start_time, ctx, model_used=self.groq_client.model)
            self.logger.info(
                "Request analysis completed",
                provider="groq",
                mode=plan.mode.value,
                intent=plan.intent.value,
                skill=plan.skill.value,
                tools_count=len(plan.tools),
                latency_ms=round(plan.analysis_latency_ms, 2),
            )
            return plan
        except Exception as exc:
            groq_exc = exc
            self.logger.warning(
                "Groq request analysis failed",
                error=str(exc),
                conversation_id=ctx.conversation_id,
            )

        for client in self.fallback_clients:
            try:
                raw_json = await asyncio.wait_for(
                    client.complete(messages=messages), timeout=client.timeout_ms / 1000.0 + 1.0
                )
                plan = self._finalize_plan(raw_json, start_time, ctx, model_used=client.model)
                self.logger.info(
                    "Request analysis completed via fallback model",
                    model=client.model,
                    mode=plan.mode.value,
                    intent=plan.intent.value,
                    latency_ms=round(plan.analysis_latency_ms, 2),
                )
                return plan
            except Exception as fb_exc:
                self.logger.warning(
                    "Fallback analysis model failed",
                    model=client.model,
                    error=str(fb_exc)[:200],
                    conversation_id=ctx.conversation_id,
                )

        for adapter in self.fallback_adapters:
            try:
                response = await asyncio.wait_for(
                    adapter.execute(messages),
                    timeout=self.config.analysis_fallback_timeout_ms / 1000.0,
                )
                plan = self._finalize_plan(response.content, start_time, ctx, model_used=response.model)
                self.logger.info(
                    "Request analysis completed via fallback provider",
                    provider=getattr(adapter, "provider_name", "?"),
                    model=response.model,
                    mode=plan.mode.value,
                    latency_ms=round(plan.analysis_latency_ms, 2),
                )
                return plan
            except Exception as fb_exc:
                self.logger.warning(
                    "Fallback analysis provider failed",
                    provider=getattr(adapter, "provider_name", "?"),
                    error=str(fb_exc)[:200],
                    conversation_id=ctx.conversation_id,
                )

        if self.nvidia_adapter is not None:
            try:
                response = await asyncio.wait_for(
                    self.nvidia_adapter.execute(messages),
                    timeout=self.config.analysis_fallback_timeout_ms / 1000.0,
                )
                plan = self._finalize_plan(
                    response.content, start_time, ctx, model_used=response.model
                )
                self.logger.info(
                    "Request analysis completed via NVIDIA fallback",
                    provider="nvidia",
                    mode=plan.mode.value,
                    intent=plan.intent.value,
                    skill=plan.skill.value,
                    tools_count=len(plan.tools),
                    latency_ms=round(plan.analysis_latency_ms, 2),
                )
                return plan
            except Exception as nvidia_exc:
                self.logger.warning(
                    "NVIDIA request analysis fallback also failed — using SafeDefaultPlan",
                    error=str(nvidia_exc),
                    conversation_id=ctx.conversation_id,
                )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return SafeDefaultFactory.create(latency_ms=latency_ms, reason=type(groq_exc).__name__)

    async def _call_groq(self, messages: list[dict[str, str]]) -> str:
        """Executes the Groq analysis call protected by Circuit Breaker and Retries."""
        async with self.circuit_breaker.call():

            async def _do_call():
                return await self.groq_client.complete(messages=messages)

            return await self.retry_manager.execute_with_retry(
                _do_call,
                operation_name="groq_request_analysis",
                provider="groq",
            )

    def _finalize_plan(
        self,
        raw_json: str,
        start_time: float,
        ctx: PipelineContext,
        model_used: str,
    ) -> ExecutionPlan:
        """Parses the raw JSON plan, validates it, and applies safety overrides."""
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        plan_dict = json.loads(self._extract_json_object(raw_json))
        plan = ExecutionPlan.model_validate(plan_dict)
        plan.groq_model_used = model_used
        plan.analysis_latency_ms = latency_ms
        return self._apply_safety_overrides(plan, ctx)

    @staticmethod
    def _extract_json_object(raw: str) -> str:
        """
        Isolates the JSON object from a raw completion. Groq's JSON mode always
        returns pure JSON, but a reasoning-capable NVIDIA fallback model may
        prepend chain-of-thought text (or wrap the object in a markdown code
        fence) before the actual object — take the substring from the first
        '{' to the matching last '}' rather than assuming the whole string
        is valid JSON.
        """
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            return raw
        return raw[start : end + 1]

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

        # Override Rule 1b: a disabled mode can't be chosen by the planner either (Auto never lands in it)
        if plan.mode == UserMode.DEEP_RESEARCH and not self.config.enable_deep_research_mode:
            self.logger.info("Safety override applied: deep_research is disabled, reverted to default mode")
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
