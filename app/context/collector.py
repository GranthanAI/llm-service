"""
Baseline Context Collector with Parallel Scatter-Gather and Graceful Degradation.
Implements LLD v2.0 Section 9 and HLD v2.0 Section 8 & 14.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import TYPE_CHECKING

import structlog

from app.config.logging import get_logger
from app.config.settings import LLMServiceConfig, get_settings
from app.context.previous_chunks import PreviousChunkStore
from app.context.retrieval_query import RetrievalQuery, RetrievalQueryRewriter
from app.context.merger import ContextMerger
from app.context.schemas import (
    ContextBundle,
    GraphContext,
    MemoryContext,
    RetrievalContext,
)
from app.grpc.clients.graph_client import GraphServiceClient
from app.grpc.clients.memory_client import MemoryServiceClient
from app.grpc.clients.retrieval_client import RetrievalServiceClient
from app.utils.metrics import CONTEXT_DEGRADED_TOTAL, CONTEXT_FETCH_DURATION
from app.utils.tracing import trace_span

if TYPE_CHECKING:
    from app.models.pipeline_context import PipelineContext


# Retrieval outcomes worth one quick retry (a cold embedding model or a slow dependency).
_TRANSIENT = {"UNAVAILABLE", "TIMEOUT"}


class ContextCollector:
    """
    Parallel Context Collector for baseline providers (Memory, Graph, Retrieval).
    Executes unconditionally on every request before Request Analysis and Workflow Dispatch.
    Never raises an exception — always returns a valid ContextBundle (graceful degradation).
    """

    def __init__(
        self,
        memory_client: MemoryServiceClient,
        graph_client: GraphServiceClient,
        retrieval_client: RetrievalServiceClient,
        merger: ContextMerger | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
        query_rewriter: RetrievalQueryRewriter | None = None,
        previous_chunks: PreviousChunkStore | None = None,
        config: LLMServiceConfig | None = None,
    ):
        self.config = config or get_settings()
        # None = no rewrite (the raw message is the query). Wired in app.main once the fast models exist.
        self.query_rewriter = query_rewriter
        self.previous_chunks = previous_chunks or PreviousChunkStore(ttl_s=self.config.retrieval_previous_chunks_ttl_s)
        self.memory_client: MemoryServiceClient = memory_client
        self.graph_client: GraphServiceClient = graph_client
        self.retrieval_client: RetrievalServiceClient = retrieval_client
        self.merger: ContextMerger = merger or ContextMerger()
        self.logger = logger or get_logger("context_collector")

    @trace_span("context_collector_collect")
    async def collect(self, ctx: PipelineContext) -> ContextBundle:
        """
        Scatter-gather baseline context from Memory, Graph, and Retrieval services in parallel.
        Implements LLD v2.0 Section 9.2 Always-Fetch Algorithm.
        """
        self.logger.info(
            "Starting parallel context collection",
            conversation_id=ctx.conversation_id,
            user_id=ctx.user_id,
            request_id=ctx.request_id,
            file_ids_count=len(ctx.file_ids),
        )

        # Step 1 & 2: Build and execute all three async fetch tasks unconditionally in parallel
        tasks = [
            self._fetch_memory(ctx),
            self._fetch_graph(ctx),
            self._fetch_retrieval(ctx),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 3: Process results with graceful degradation handling
        memory_res: MemoryContext | None = None
        graph_res: GraphContext | None = None
        retrieval_res: RetrievalContext | None = None
        missing_sources: list[str] = []

        sources = ["memory", "graph", "retrieval"]
        for i, res in enumerate(results):
            source_name = sources[i]
            if isinstance(res, Exception):
                self.logger.warning(
                    "Baseline context provider failed — continuing with graceful degradation",
                    source=source_name,
                    error=str(res),
                    conversation_id=ctx.conversation_id,
                )
                CONTEXT_DEGRADED_TOTAL.labels(source=source_name, reason=type(res).__name__).inc()
                missing_sources.append(source_name)
            else:
                if source_name == "memory" and isinstance(res, MemoryContext):
                    memory_res = res
                elif source_name == "graph" and isinstance(res, GraphContext):
                    graph_res = res
                elif source_name == "retrieval" and isinstance(res, RetrievalContext):
                    retrieval_res = res

        # Step 4: Merge, deduplicate, and rank items
        bundle = self.merger.merge(
            memory=memory_res,
            graph=graph_res,
            retrieval=retrieval_res,
            missing_sources=missing_sources,
        )

        self.logger.info(
            "Context collection completed",
            degraded=bundle.degraded,
            missing_sources=bundle.missing_sources,
            memory_messages=len(bundle.memory.short_term_messages) if bundle.memory else 0,
            memory_facts=len(bundle.memory.long_term_facts) if bundle.memory else 0,
            graph_entities=len(bundle.graph.entities) if bundle.graph else 0,
            retrieval_chunks=len(bundle.retrieval.chunks) if bundle.retrieval else 0,
        )

        # Step 5: Always return bundle (never raises)
        return bundle

    async def _fetch_memory(self, ctx: PipelineContext) -> MemoryContext:
        """Fetch memory context via gRPC."""
        start_time = time.perf_counter()
        try:
            res = await self.memory_client.get_memory_context(
                user_id=ctx.user_id,
                conversation_id=ctx.conversation_id,
                query=ctx.user_message,
                trace_id=ctx.trace_id,
                ancestors=ctx.ancestors,
            )
            return res
        finally:
            CONTEXT_FETCH_DURATION.labels(source="memory").observe(time.perf_counter() - start_time)

    async def _fetch_graph(self, ctx: PipelineContext) -> GraphContext:
        """Fetch knowledge graph context via gRPC."""
        start_time = time.perf_counter()
        try:
            res = await self.graph_client.get_graph_context(
                user_id=ctx.user_id,
                conversation_id=ctx.conversation_id,
                query=ctx.user_message,
                trace_id=ctx.trace_id,
            )
            return res
        finally:
            CONTEXT_FETCH_DURATION.labels(source="graph").observe(time.perf_counter() - start_time)

    def _input_budget_tokens(self) -> int:
        cfg = self.config
        if cfg.retrieval_input_budget_tokens:
            return cfg.retrieval_input_budget_tokens
        return min(t for t in (cfg.prompt_token_budget, cfg.context_budget_default_tokens) if t > 0)

    async def _try_full_text(self, ctx: PipelineContext) -> RetrievalContext | None:
        """Whole text of small attached documents, or None (too big, not supported by the server, any error)."""
        budget = int(self._input_budget_tokens() * self.config.retrieval_full_text_share)
        if budget <= 0:
            return None
        try:
            res = await self.retrieval_client.get_document_chunks(
                user_id=ctx.user_id, file_ids=ctx.file_ids, max_tokens=budget,
                trace_id=ctx.trace_id, conversation_id=ctx.conversation_id,
            )
        except Exception as exc:  # UNIMPLEMENTED (older server), timeout, anything: normal retrieval instead
            self.logger.info("retrieval_full_text_unavailable", error=type(exc).__name__, request_id=ctx.request_id)
            return None
        if res.status != "OK" or res.truncated or not res.chunks:
            return None
        return res

    async def _fetch_retrieval(self, ctx: PipelineContext) -> RetrievalContext | None:
        """Fetch retrieval chunks via gRPC.

        Retrieval only searches explicitly attached files (scope FILES); ALL /
        KNOWLEDGE_BASE are unsupported server-side. With no attached files there
        is nothing to search, so skip the RPC (and its unsupported-scope metric).

        Small attachments go in full (flag); otherwise a standalone rewrite of the question is searched and the
        previous turn's chunk ids are offered again (flags). Every step falls back to the plain behaviour.
        """
        if not ctx.file_ids:
            return None
        cfg = self.config
        start_time = time.perf_counter()
        stages: dict[str, float] = {}
        try:
            res = None
            rq = RetrievalQuery(query=ctx.user_message, method="disabled")
            previous: list[str] = []

            if cfg.retrieval_full_text_enabled:
                t = time.perf_counter()
                res = await self._try_full_text(ctx)
                stages["full_text_ms"] = round((time.perf_counter() - t) * 1000, 1)

            if res is None:
                if cfg.retrieval_query_rewrite_enabled and self.query_rewriter is not None:
                    t = time.perf_counter()
                    try:
                        rq = await self.query_rewriter.build(ctx.user_message, list(ctx.event_history))
                    except Exception as exc:  # the rewriter should not raise; if it does, search the raw message
                        self.logger.warning("retrieval_rewrite_error", error=type(exc).__name__)
                    stages["rewrite_ms"] = round((time.perf_counter() - t) * 1000, 1)
                if cfg.retrieval_previous_chunks_enabled:
                    previous = self.previous_chunks.get(ctx.conversation_id)

                t = time.perf_counter()
                for attempt in (1, 2):
                    res = await self.retrieval_client.get_relevant_chunks(
                        user_id=ctx.user_id,
                        conversation_id=ctx.conversation_id,
                        query=rq.query,
                        file_ids=ctx.file_ids,
                        trace_id=ctx.trace_id,
                        top_k=cfg.retrieval_top_k,
                        max_tokens=cfg.retrieval_max_tokens,
                        dense_query=rq.dense_query,
                        previous_chunk_ids=previous,
                    )
                    if res.status not in _TRANSIENT or attempt == 2:
                        break
                    self.logger.warning(
                        "Retrieval returned a transient status; retrying once",
                        status=res.status,
                        conversation_id=ctx.conversation_id,
                        request_id=ctx.request_id,
                    )
                    await asyncio.sleep(0.2)
                stages["search_ms"] = round((time.perf_counter() - t) * 1000, 1)

            if cfg.retrieval_previous_chunks_enabled and res.chunks and not res.full_text:
                self.previous_chunks.record(ctx.conversation_id, [c.chunk_id for c in res.chunks])
            self.logger.info(
                "retrieval_result",
                conversation_id=ctx.conversation_id,
                request_id=ctx.request_id,
                response_message_id=ctx.response_message_id,
                status=res.status,
                detail=res.status_detail,
                requested=res.requested_count or len(ctx.file_ids),
                authorized=res.authorized_count,
                ready=res.ready_count,
                chunks=len(res.chunks),
                sources=sorted({c.file_id for c in res.chunks}),
            )
            self._log_feedback(ctx, rq, res, previous, stages)
            return res
        finally:
            CONTEXT_FETCH_DURATION.labels(source="retrieval").observe(
                time.perf_counter() - start_time
            )

    def _log_feedback(self, ctx, rq: RetrievalQuery, res: RetrievalContext, previous: list[str], stages: dict) -> None:
        """One structured line per retrieval (see docs/retrieval-feedback.md). Never document content; query text only
        behind LOG_QUERY_TEXT."""
        fields = dict(
            trace_id=ctx.trace_id,
            conversation_id=ctx.conversation_id,
            message_id=ctx.message_id,
            query_hash=hashlib.sha256(str(ctx.user_message).encode("utf-8")).hexdigest()[:16],
            rewrite_method=rq.method,
            rewrite_used=bool(rq.rewritten),
            dense_query_sent=bool(rq.dense_query),
            previous_chunks_sent=len(previous),
            full_text=res.full_text,
            status=res.status,
            coverage=res.coverage,
            top_score=res.top_score,
            chunk_ids=[c.chunk_id for c in res.chunks],
            scores=[c.score for c in res.chunks],
            latency_ms=stages,
        )
        if self.config.log_query_text:
            fields["query_text"] = str(ctx.user_message)
            fields["retrieval_query_text"] = rq.query
            fields["dense_query_text"] = rq.dense_query
        self.logger.info("retrieval_feedback", **fields)
