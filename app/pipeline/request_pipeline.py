"""
Request Pipeline Orchestrator.
Wires together Context Collection -> Request Analysis -> Workflow Dispatch ->
Prompt Build -> Context Window Management -> Generation Router -> Streaming
Engine into the single callable invoked by the Kafka consumer for every
chat.message.created event, per HLD v2.0 Section 5.1 (Canonical Pipeline).
"""

import time
from datetime import UTC, datetime
from typing import Any

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
from app.context.schemas import ContextBundle, MemoryContext, Message, Role
from app.exceptions.user_facing import DocumentContextUnavailableError, InvalidModeError
from app.models.execution_plan import UserMode
from app.tools.query_rewrite import SearchQueryRewriter
from app.tools.selection import build_selected_calls, drop_calls_covered_by_mode, filter_default_tools
from app.workflow_engine.modes import apply_requested_mode, resolve_requested_mode
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
        tool_registry: Any | None = None,
        tool_dispatcher: Any | None = None,
        query_rewriter: SearchQueryRewriter | None = None,
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
        self.tool_registry = tool_registry
        self.tool_dispatcher = tool_dispatcher
        # Default: rewrite follow-up search queries with the analyzer's Groq client when one exists.
        self.query_rewriter = query_rewriter or SearchQueryRewriter(
            getattr(request_analyzer, "groq_client", None),
            fallback_clients=getattr(request_analyzer, "fallback_clients", None),
            fallback_adapters=getattr(request_analyzer, "fallback_adapters", None),
        )

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

            # Explicit tool selection: validate against the live registry BEFORE doing any work.
            # Unknown / disabled / un-runnable tools are rejected with a clear message; nothing runs.
            # Saved DEFAULT tools (not chosen on this message) are filtered to what still
            # works instead of failing the message; explicit choices stay strict.
            if event.defaults_applied.tools:
                ctx.selected_tools, dropped_tools = filter_default_tools(ctx.selected_tools, self.tool_registry)
                if dropped_tools:
                    self.logger.info("default_tools_skipped", dropped=dropped_tools, **ids)
            selected_calls = build_selected_calls(ctx.selected_tools, self.tool_registry, ctx.user_message)
            timing["tools_selected"] = len(selected_calls)
            # Explicit response mode (None = Auto). Unknown / disabled modes are rejected, never ignored.
            # A saved DEFAULT mode that is no longer available falls back to Auto.
            try:
                requested_mode = resolve_requested_mode(event.mode_hint, self.config)
            except InvalidModeError:
                if not event.defaults_applied.mode:
                    raise
                self.logger.info("default_mode_skipped", mode=event.mode_hint, **ids)
                requested_mode = None
            timing["mode"] = requested_mode.value if requested_mode else "auto"

            # Fast path: an obvious greeting / thanks / goodbye needs no Memory/Graph context,
            # no analysis call and no tools. Safety checks above still ran, and generation,
            # streaming and persistence below are the normal path.
            fast = (
                self.config.enable_fast_path
                and not ctx.safety_check_failed
                and not selected_calls
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
            self._enforce_document_context(ctx)
            self._ensure_conversation_history(ctx)
            t = _lap("context_ms", t)

            # 3. Request Analysis -> ExecutionPlan — never raises, falls back internally
            if fast:
                plan = SafeDefaultFactory.create(latency_ms=0.0, reason="fast_path")
                plan.groq_model_used = "fast_path"
                plan.analysis_confidence = 1.0
            else:
                plan = await self.request_analyzer.analyze(ctx)
            plan = apply_requested_mode(plan, requested_mode, self.config)
            # Web search mode already runs the search: do not run it again for an explicit tool pick.
            selected_calls = drop_calls_covered_by_mode(selected_calls, plan.mode)
            ctx.plan = plan
            t = _lap("analysis_ms", t)

            # 4. Workflow Dispatch (Mode Handler or LangGraph)
            # A follow-up like "check again and tell" is useless to a search engine on its own: build a
            # standalone query from the conversation before any search tool runs.
            search_plan_tools = [t_ for t_ in plan.tools if t_.tool_name == "web_search"]
            if (
                plan.mode == UserMode.WEB_SEARCH
                or search_plan_tools
                or any(c.tool_name == "web_search" for c in selected_calls)
            ):
                history = ctx.context_bundle.memory.short_term_messages if ctx.context_bundle and ctx.context_bundle.memory else []
                ctx.search_query = await self.query_rewriter.rewrite(ctx.user_message, list(history))
                for c in selected_calls:
                    if c.tool_name == "web_search" and "query" in c.params:
                        c.params["query"] = ctx.search_query
                for t_ in search_plan_tools:
                    if str(t_.params.get("query", "")).strip() in ("", ctx.user_message.strip()):
                        t_.params["query"] = ctx.search_query
                timing["search_query_rewritten"] = ctx.search_query != ctx.user_message
            if selected_calls:
                # The user chose these tools explicitly, so they run exactly once and are required
                # (a failure is reported to the user, not silently dropped). Remove any duplicate the
                # planner also asked for so the workflow does not run it a second time.
                chosen = {c.tool_name for c in selected_calls}
                plan.tools = [t_ for t_ in plan.tools if t_.tool_name not in chosen]
            result = await self.workflow_engine.execute(plan, ctx)
            if selected_calls:
                forced = await self.tool_dispatcher.dispatch(
                    selected_calls,
                    trace_id=ctx.trace_id,
                    user_id=ctx.user_id,
                    conversation_id=ctx.conversation_id,
                )
                result.tool_outputs.extend(forced)
                self.logger.info(
                    "selected_tools_executed",
                    **ids,
                    tools=[c.tool_name for c in selected_calls],
                    succeeded=[r.tool_name for r in forced if r.success],
                )
            t = _lap("workflow_ms", t)

            # 5. Prompt Build
            composed = self.prompt_builder.build(result, ctx)
            if ctx.file_ids:
                section_names = {sec.name for sec in getattr(composed, "sections", [])}
                bundle = ctx.context_bundle
                self.logger.info(
                    "rag_prompt_context",
                    **ids,
                    file_ids=len(ctx.file_ids),
                    retrieval_status=bundle.retrieval.status if bundle and bundle.retrieval else "MISSING",
                    chunks=len(bundle.retrieval.chunks) if bundle and bundle.retrieval else 0,
                    retrieval_in_prompt="retrieval" in section_names,
                )

            # 6. Context Window Management (trim to model budget)
            if self.config.enable_nvidia:
                target_model = self.config.nvidia_model
                max_tokens = self.config.nvidia_max_tokens
                provider_name = "nvidia"
            else:
                target_model = self.config.groq_model
                max_tokens = self.config.groq_max_tokens
                provider_name = "groq"
            trimmed = self.context_window_manager.manage(
                composed,
                target_model=target_model,
                custom_input_budget=self.config.prompt_token_budget or None,
            )
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

    def _ensure_conversation_history(self, ctx: PipelineContext) -> None:
        """Use the history that came with the request when the memory service supplied none (it is
        asynchronous and was observed degraded), so the model and tools always see the conversation."""
        bundle = ctx.context_bundle
        memory = bundle.memory if bundle else None
        inherited = [
            Message(role=Role(role), content=content)
            for role, content in ctx.inherited_history
            if role in ("user", "assistant") and content.strip()
        ]
        if inherited and memory and memory.short_term_messages:
            # A branch: the memory service only knows this conversation's own turns. Ancestor turns
            # (up to the branch point) come first, then the branch's own.
            memory.short_term_messages = inherited + list(memory.short_term_messages)
            self.logger.info("inherited_history_added", conversation_id=ctx.conversation_id, count=len(inherited))
            return
        if (memory and memory.short_term_messages) or not (ctx.event_history or inherited):
            return
        messages = inherited + [
            Message(role=Role(role), content=content)
            for role, content in ctx.event_history
            if role in ("user", "assistant") and content.strip()
        ]
        if not messages:
            return
        bundle = bundle or ContextBundle()
        memory = memory or MemoryContext()
        memory.short_term_messages = messages
        bundle.memory = memory
        ctx.context_bundle = bundle
        self.logger.info(
            "history_from_event",
            conversation_id=ctx.conversation_id,
            request_id=ctx.request_id,
            messages=len(messages),
            memory_degraded=bool(bundle.degraded),
        )

    def _enforce_document_context(self, ctx: PipelineContext) -> None:
        """Attached documents must actually be searched.

        If the question carries file_ids and the Retrieval Service could not use them (no access, not
        indexed yet, indexing failed, service down, timeout), fail the turn with an actionable message
        instead of answering from general knowledge as though the document had been read. An indexed
        document with no relevant passage is allowed through, with a notice so the model says so.
        """
        if not ctx.file_ids:
            return
        bundle = ctx.context_bundle
        retrieval = bundle.retrieval if bundle else None
        if retrieval is None:
            # The gRPC call itself failed (collector degraded gracefully): treat as unavailable.
            raise DocumentContextUnavailableError("UNAVAILABLE", "retrieval service unreachable")
        if retrieval.status in ("ACCESS_DENIED", "NOT_INDEXED", "INDEXING", "INDEXING_FAILED", "UNAVAILABLE", "TIMEOUT"):
            raise DocumentContextUnavailableError(retrieval.status, retrieval.status_detail)
        notes = []
        if retrieval.status == "NO_MATCH":
            notes.append(
                "The user attached document(s), but no passage in them was relevant to this question. "
                "Say clearly that the attached document does not appear to contain the answer; do not invent its contents."
            )
        if 0 < retrieval.authorized_count < retrieval.requested_count:
            notes.append(
                f"{retrieval.requested_count - retrieval.authorized_count} of the attached documents could not be accessed and were skipped."
            )
        if notes:
            ctx.document_notice = " ".join(notes)

    @staticmethod
    async def _track_route(token_iter, params: dict, ctx: PipelineContext):
        """Copy the router's actual provider choice onto the context once streaming starts."""
        first = True
        async for token in token_iter:
            if first:
                first = False
                route = params.get("_route") or {}
                # The provider that actually produced the first token (the chain reports it on every turn).
                if route.get("provider") and route["provider"] != "primary":
                    ctx.selected_provider = route["provider"]
                if route.get("fallback_used"):
                    ctx.generation_fallback_used = True
            yield token
