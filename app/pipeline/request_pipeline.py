"""
Request Pipeline Orchestrator.
Wires together Context Collection -> Request Analysis -> Workflow Dispatch ->
Prompt Build -> Context Window Management -> Generation Router -> Streaming
Engine into the single callable invoked by the Kafka consumer for every
chat.message.created event, per HLD v2.0 Section 5.1 (Canonical Pipeline).
"""

import time
from datetime import UTC, datetime

import structlog

from app.config.logging import get_logger
from app.config.settings import LLMServiceConfig, get_settings
from app.context.collector import ContextCollector
from app.context.merger import ContextMerger
from app.context_window.manager import ContextWindowManager
from app.models.pipeline_context import PipelineContext
from app.models.request import ChatMessageCreatedEvent
from app.producers.kafka_producer import KafkaPublisher
from app.prompts.builder import PromptBuilder
from app.providers.router import GenerationRouter
from app.request_analyzer.analyzer import RequestAnalyzer
from app.request_analyzer.fast_path import is_simple_smalltalk
from app.request_analyzer.safe_default import SafeDefaultFactory
from app.security.pii import PIIDetector
from app.security.sanitizer import InputSanitizer
from app.services.streaming_service import StreamingEngine
from app.utils.error_handler import ErrorHandler
from app.utils.helpers import generate_request_id
from app.workflow_engine.engine import WorkflowEngine


class RequestPipeline:
    """
    Single orchestration point that turns an incoming ChatMessageCreatedEvent
    into a fully streamed and published response. Reused as-is by the Kafka
    consumer path; every downstream component is unchanged and shared.
    """

    def __init__(
        self,
        context_collector: ContextCollector,
        request_analyzer: RequestAnalyzer,
        workflow_engine: WorkflowEngine,
        prompt_builder: PromptBuilder,
        context_window_manager: ContextWindowManager,
        generation_router: GenerationRouter,
        streaming_engine: StreamingEngine,
        publisher: KafkaPublisher,
        config: LLMServiceConfig | None = None,
        sanitizer: InputSanitizer | None = None,
        pii_detector: PIIDetector | None = None,
        error_handler: ErrorHandler | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.context_collector = context_collector
        self.request_analyzer = request_analyzer
        self.workflow_engine = workflow_engine
        self.prompt_builder = prompt_builder
        self.context_window_manager = context_window_manager
        self.generation_router = generation_router
        self.streaming_engine = streaming_engine
        self.publisher = publisher
        self.config = config or get_settings()
        self.sanitizer = sanitizer or InputSanitizer()
        self.pii_detector = pii_detector or PIIDetector()
        self.error_handler = error_handler or ErrorHandler()
        self.logger = logger or get_logger("request_pipeline")

    async def run(self, event: ChatMessageCreatedEvent) -> None:
        """
        Executes the full canonical pipeline for one chat.message.created event.
        Never raises — on failure, publishes an error-variant chat.response.generated
        event so the Conversation Service is never left waiting indefinitely.
        """
        request_id = generate_request_id()
        ctx = PipelineContext.from_event(event, request_id=request_id)
        t_start = time.perf_counter()
        # Timing fields (ms). Never contains message content.
        timing: dict[str, float | bool | str | None] = {
            # Kafka produce -> consume (event.timestamp is set by Conversation Service's outbox).
            "queue_delay_ms": round(
                max(0.0, (datetime.now(UTC) - event.timestamp).total_seconds() * 1000.0), 1
            ),
        }
        ids = {
            "conversation_id": ctx.conversation_id,
            "message_id": ctx.message_id,
            "response_message_id": ctx.response_message_id,
            "request_id": ctx.request_id,
            "trace_id": ctx.trace_id,
        }

        def _lap(name: str, since: float) -> float:
            now = time.perf_counter()
            timing[name] = round((now - since) * 1000.0, 1)
            return now

        try:
            # 1. Input sanitization + PII redaction (mutates ctx.safety_check_failed / pii_detected)
            sanitized = self.sanitizer.sanitize(ctx.user_message, ctx=ctx)
            redacted, _ = self.pii_detector.detect_and_redact(sanitized, ctx=ctx)
            ctx.user_message = redacted

            # Fast path: an obvious greeting / thanks / goodbye needs no Memory/Graph context,
            # no analysis call and no tools. Safety checks above still ran, and generation,
            # streaming and persistence below are the normal path.
            fast = (
                self.config.enable_fast_path
                and not ctx.safety_check_failed
                and is_simple_smalltalk(
                    ctx.user_message, file_ids=ctx.file_ids, mode_hint=ctx.mode_hint
                )
            )
            timing["fast_path"] = fast
            t = time.perf_counter()

            # 2. Context Collection (Memory, Graph, Retrieval) — never raises
            if fast:
                ctx.context_bundle = ContextMerger().merge()
            else:
                ctx.context_bundle = await self.context_collector.collect(ctx)
            ctx.context_collected_at = datetime.now(UTC)
            t = _lap("context_ms", t)

            # 3. Request Analysis -> ExecutionPlan — never raises, falls back internally
            if fast:
                plan = SafeDefaultFactory.create(latency_ms=0.0, reason="fast_path")
                plan.groq_model_used = "fast_path"
                plan.analysis_confidence = 1.0
            else:
                plan = await self.request_analyzer.analyze(ctx)
            ctx.plan = plan
            t = _lap("analysis_ms", t)

            # 4. Workflow Dispatch (Mode Handler or LangGraph)
            result = await self.workflow_engine.execute(plan, ctx)
            t = _lap("workflow_ms", t)

            # 5. Prompt Build
            composed = self.prompt_builder.build(result, ctx)

            # 6. Context Window Management (trim to model budget)
            if self.config.enable_nvidia:
                target_model = self.config.nvidia_model
                max_tokens = self.config.nvidia_max_tokens
                provider_name = "nvidia"
            else:
                target_model = self.config.groq_model
                max_tokens = self.config.groq_max_tokens
                provider_name = "groq"
            trimmed = self.context_window_manager.manage(composed, target_model=target_model)
            t = _lap("prompt_ms", t)

            # 7. Generation (streaming) with provider fallback
            ctx.inference_started_at = datetime.now(UTC)
            ctx.selected_provider = provider_name
            params: dict = {
                "temperature": plan.suggested_temperature,
                "max_tokens": max_tokens,
            }
            token_iter = self._track_route(
                self.generation_router.generate_stream(trimmed, params), params, ctx
            )

            # 8. Streaming Engine: publishes chat.response.chunk incrementally, then
            # chat.response.generated + memory.update.requested on completion.
            await self.streaming_engine.stream(token_iter, ctx)
            _lap("generation_ms", t)
            if ctx.first_chunk_at and ctx.inference_started_at:
                timing["ttft_ms"] = round(
                    (ctx.first_chunk_at - ctx.inference_started_at).total_seconds() * 1000.0, 1
                )
            timing["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 1)
            timing["provider"] = ctx.selected_provider
            timing["fallback_used"] = ctx.generation_fallback_used
            timing["outcome"] = "ok"
            self.logger.info("pipeline_timing", **ids, **timing)

        except Exception as exc:
            timing["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 1)
            timing["outcome"] = "error"
            timing["error_type"] = type(exc).__name__
            self.logger.info("pipeline_timing", **ids, **timing)
            self.logger.error(
                "Request pipeline failed — publishing error response",
                conversation_id=ctx.conversation_id,
                message_id=ctx.message_id,
                error=str(exc),
                exc_info=True,
            )
            self.error_handler.record_metrics(exc, stage=ctx.engine_type or "pipeline")
            error_event = self.error_handler.build_error_response_event(ctx, exc)
            try:
                await self.publisher.publish_error_response(ctx, error_event)
            except Exception:
                self.logger.error(
                    "Failed to publish error response event — request will time out downstream",
                    conversation_id=ctx.conversation_id,
                    message_id=ctx.message_id,
                )

    @staticmethod
    async def _track_route(token_iter, params: dict, ctx: PipelineContext):
        """Copy the router's actual provider choice onto the context once streaming starts."""
        first = True
        async for token in token_iter:
            if first:
                first = False
                route = params.get("_route") or {}
                if route.get("fallback_used"):
                    ctx.generation_fallback_used = True
                    ctx.selected_provider = route.get("provider") or ctx.selected_provider
            yield token
