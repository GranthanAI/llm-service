"""
Baseline Context Collector with Parallel Scatter-Gather and Graceful Degradation.
Implements LLD v2.0 Section 9 and HLD v2.0 Section 8 & 14.
"""

import asyncio
import time

import structlog

from app.config.logging import get_logger
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
from app.models.pipeline_context import PipelineContext
from app.utils.metrics import CONTEXT_DEGRADED_TOTAL, CONTEXT_FETCH_DURATION
from app.utils.tracing import trace_span


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
    ):
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

    async def _fetch_retrieval(self, ctx: PipelineContext) -> RetrievalContext:
        """Fetch retrieval chunks via gRPC."""
        start_time = time.perf_counter()
        try:
            res = await self.retrieval_client.get_relevant_chunks(
                user_id=ctx.user_id,
                conversation_id=ctx.conversation_id,
                query=ctx.user_message,
                file_ids=ctx.file_ids if ctx.file_ids else None,
                trace_id=ctx.trace_id,
            )
            return res
        finally:
            CONTEXT_FETCH_DURATION.labels(source="retrieval").observe(
                time.perf_counter() - start_time
            )
