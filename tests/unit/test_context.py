"""
Unit tests for ContextCollector, ContextMerger, Deduplication, Ranking, and Graceful Degradation.
Implements Phase 5 deliverables verification.
"""

from unittest.mock import AsyncMock

import pytest

from app.context.collector import ContextCollector
from app.context.merger import ContextMerger
from app.context.schemas import (
    ContextBundle,
    DocumentChunk,
    EntityNode,
    Fact,
    GraphContext,
    MemoryContext,
    Message,
    RelationshipEdge,
    RetrievalContext,
    Role,
)
from app.exceptions.grpc import GRPCTimeoutError, GRPCUnavailableError
from app.models.pipeline_context import PipelineContext


@pytest.fixture
def mock_memory_client() -> AsyncMock:
    client = AsyncMock()
    client.get_memory_context.return_value = MemoryContext(
        short_term_messages=[
            Message(role=Role.USER, content="Explain relativity"),
            Message(role=Role.ASSISTANT, content="Relativity has two parts..."),
        ],
        long_term_facts=[
            Fact(fact_id="f1", statement="User is a physics student", confidence=0.95),
            Fact(fact_id="f2", statement="User is a physics student", confidence=0.95),  # duplicate
        ],
    )
    return client


@pytest.fixture
def mock_graph_client() -> AsyncMock:
    client = AsyncMock()
    client.get_graph_context.return_value = GraphContext(
        entities=[
            EntityNode(
                id="e1",
                name="Albert Einstein",
                type="Person",
                description="Famous theoretical physicist.",
            ),
            EntityNode(id="e1", name="Albert Einstein", type="Person"),  # duplicate ID
            EntityNode(id="e2", name="General Relativity", type="Theory"),
        ],
        relationships=[
            RelationshipEdge(source_id="e1", target_id="e2", relation_type="DEVELOPED"),
            RelationshipEdge(
                source_id="e1", target_id="e2", relation_type="DEVELOPED"
            ),  # duplicate edge
        ],
        subgraph_summary="Albert Einstein developed General Relativity.",
    )
    return client


@pytest.fixture
def mock_retrieval_client() -> AsyncMock:
    client = AsyncMock()
    client.get_relevant_chunks.return_value = RetrievalContext(
        chunks=[
            DocumentChunk(
                chunk_id="c1",
                file_id="f1",
                content="General Relativity describes gravitation.",
                score=0.82,
            ),
            DocumentChunk(
                chunk_id="c2", file_id="f2", content="Famous theoretical physicist.", score=0.90
            ),  # overlaps with graph entity description
            DocumentChunk(
                chunk_id="c3",
                file_id="f1",
                content="General Relativity describes gravitation.",
                score=0.82,
            ),  # duplicate chunk
            DocumentChunk(
                chunk_id="c4",
                file_id="f3",
                content="Special Relativity was proposed in 1905.",
                score=0.95,
            ),
        ],
        total_chunks=4,
        query="relativity",
    )
    return client


@pytest.mark.asyncio
async def test_context_collector_parallel_gather_and_merger(
    mock_memory_client: AsyncMock,
    mock_graph_client: AsyncMock,
    mock_retrieval_client: AsyncMock,
):
    """Verify ContextCollector gathers in parallel and ContextMerger sanitizes."""
    collector = ContextCollector(
        memory_client=mock_memory_client,
        graph_client=mock_graph_client,
        retrieval_client=mock_retrieval_client,
        merger=ContextMerger(),
    )

    ctx = PipelineContext(
        conversation_id="conv_100",
        user_id="user_200",
        message_id="msg_300",
        request_id="req_400",
        user_message="Explain relativity",
        file_ids=["f1"],
    )

    bundle: ContextBundle = await collector.collect(ctx)

    assert bundle.degraded is False
    assert len(bundle.missing_sources) == 0

    # 1. Verify Memory deduplication
    assert bundle.memory is not None
    assert len(bundle.memory.short_term_messages) == 2
    assert len(bundle.memory.long_term_facts) == 1  # 1 duplicate removed

    # 2. Verify Graph deduplication
    assert bundle.graph is not None
    assert len(bundle.graph.entities) == 2  # duplicate node ID removed
    assert len(bundle.graph.relationships) == 1  # duplicate edge removed
    assert bundle.graph.subgraph_summary == "Albert Einstein developed General Relativity."

    # 3. Verify Retrieval deduplication and ranking
    assert bundle.retrieval is not None
    # c2 removed because it matches entity description 'Famous theoretical physicist.'
    # c3 removed because it's duplicate of c1
    # Remaining: c4 (score 0.95) and c1 (score 0.82) in descending order
    assert len(bundle.retrieval.chunks) == 2
    assert bundle.retrieval.chunks[0].chunk_id == "c4"
    assert bundle.retrieval.chunks[0].score == 0.95
    assert bundle.retrieval.chunks[1].chunk_id == "c1"
    assert bundle.retrieval.chunks[1].score == 0.82


@pytest.mark.asyncio
async def test_context_collector_graceful_degradation_single_failure(
    mock_graph_client: AsyncMock,
    mock_retrieval_client: AsyncMock,
):
    """Verify graceful degradation when Memory Service fails with gRPC error."""
    failing_memory_client = AsyncMock()
    failing_memory_client.get_memory_context.side_effect = GRPCUnavailableError(
        "Connection refused", service="MemoryService"
    )

    collector = ContextCollector(
        memory_client=failing_memory_client,
        graph_client=mock_graph_client,
        retrieval_client=mock_retrieval_client,
        merger=ContextMerger(),
    )

    ctx = PipelineContext(
        conversation_id="conv_100",
        user_id="user_200",
        message_id="msg_300",
        request_id="req_400",
        user_message="Explain relativity",
    )

    bundle = await collector.collect(ctx)

    assert bundle.degraded is True
    assert bundle.missing_sources == ["memory"]
    assert bundle.memory is None
    assert bundle.graph is not None
    assert bundle.retrieval is not None


@pytest.mark.asyncio
async def test_context_collector_graceful_degradation_all_services_fail():
    """Verify graceful degradation when all 3 baseline context providers fail."""
    failing_memory = AsyncMock()
    failing_memory.get_memory_context.side_effect = GRPCUnavailableError(
        "memory down", service="MemoryService"
    )

    failing_graph = AsyncMock()
    failing_graph.get_graph_context.side_effect = GRPCTimeoutError(
        "graph timed out", service="GraphService"
    )

    failing_retrieval = AsyncMock()
    failing_retrieval.get_relevant_chunks.side_effect = Exception("retrieval generic failure")

    collector = ContextCollector(
        memory_client=failing_memory,
        graph_client=failing_graph,
        retrieval_client=failing_retrieval,
        merger=ContextMerger(),
    )

    ctx = PipelineContext(
        conversation_id="conv_100",
        user_id="user_200",
        message_id="msg_300",
        request_id="req_400",
        user_message="Explain relativity",
    )

    # Must never raise an exception
    bundle = await collector.collect(ctx)

    assert bundle.degraded is True
    assert set(bundle.missing_sources) == {"memory", "graph", "retrieval"}
    assert bundle.memory is None
    assert bundle.graph is None
    assert bundle.retrieval is None


def test_context_merger_empty_inputs():
    """Verify ContextMerger handles empty / None inputs gracefully."""
    merger = ContextMerger()
    bundle = merger.merge(memory=None, graph=None, retrieval=None, missing_sources=["memory"])
    assert bundle.degraded is True
    assert bundle.missing_sources == ["memory"]
    assert bundle.memory is None
    assert bundle.graph is None
    assert bundle.retrieval is None
