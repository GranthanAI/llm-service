"""
Unit tests for gRPC Base Client, Connection Pool, Memory Client, Graph Client, and Retrieval Client.
Implements Phase 4 deliverables verification.
"""

from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from app.context.schemas import GraphContext, MemoryContext, RetrievalContext, Role
from app.exceptions.grpc import GRPCTimeoutError, GRPCUnavailableError
from app.grpc.clients.graph_client import GraphServiceClient
from app.grpc.clients.memory_client import MemoryServiceClient
from app.grpc.clients.retrieval_client import RetrievalServiceClient
from app.grpc.proto import graph_pb2, memory_pb2, retrieval_pb2


@pytest.mark.asyncio
async def test_connection_pool_round_robin():
    """Verify ChannelPool round-robin stub rotation."""
    client = MemoryServiceClient(host="localhost", port=50051, pool_size=3)
    # Mock channels & stubs
    client._channels = [AsyncMock(), AsyncMock(), AsyncMock()]
    mock_stubs = [MagicMock(), MagicMock(), MagicMock()]
    client._stubs = mock_stubs
    client._initialized = True

    s1 = await client.get_stub()
    s2 = await client.get_stub()
    s3 = await client.get_stub()
    s4 = await client.get_stub()

    assert s1 == mock_stubs[0]
    assert s2 == mock_stubs[1]
    assert s3 == mock_stubs[2]
    assert s4 == mock_stubs[0]  # wraps back to 0


def test_base_client_metadata_builder():
    """Verify metadata propagation."""
    client = MemoryServiceClient(host="localhost", port=50051)
    meta = client.build_metadata(
        trace_id="00-trace123-span456-01",
        user_id="user_abc",
        conversation_id="conv_xyz",
    )
    meta_dict = dict(meta)
    assert meta_dict["x-service-name"] == "llm-service"
    assert meta_dict["x-trace-id"] == "00-trace123-span456-01"
    assert meta_dict["x-user-id"] == "user_abc"
    assert meta_dict["x-conversation-id"] == "conv_xyz"


@pytest.mark.asyncio
async def test_memory_client_get_memory_context_success():
    """Verify MemoryServiceClient fetch and proto-to-domain conversion."""
    client = MemoryServiceClient(host="localhost", port=50051, deadline_ms=2000)

    # Mock response
    mock_response = memory_pb2.GetMemoryContextResponse(
        short_term_messages=[
            memory_pb2.Message(
                role="user", content="What is AI?", timestamp=1700000000000, message_id="m1"
            ),
            memory_pb2.Message(
                role="assistant", content="AI is...", timestamp=1700000001000, message_id="m2"
            ),
        ],
        long_term_facts=[
            memory_pb2.Fact(
                content="User prefers Python",
                confidence=0.9,
                category="preference",
                last_updated=1700000000000,
            ),
        ],
        total_tokens=150,
        request_id="req_mem_1",
    )

    mock_stub = MagicMock()
    mock_stub.GetMemoryContext = AsyncMock(return_value=mock_response)
    client.get_stub = AsyncMock(return_value=mock_stub)
    client._initialized = True

    context: MemoryContext = await client.get_memory_context(
        user_id="user_123",
        conversation_id="conv_456",
        query="What is AI?",
        trace_id="00-trace-01",
    )

    assert len(context.short_term_messages) == 2
    assert context.short_term_messages[0].role == Role.USER
    assert context.short_term_messages[0].content == "What is AI?"
    assert len(context.long_term_facts) == 1
    assert context.long_term_facts[0].statement == "User prefers Python"
    assert context.long_term_facts[0].confidence == 0.9


@pytest.mark.asyncio
async def test_graph_client_get_graph_context_success():
    """Verify GraphServiceClient fetch and proto-to-domain conversion."""
    client = GraphServiceClient(host="localhost", port=50052, deadline_ms=2000)

    mock_response = graph_pb2.GetGraphContextResponse(
        nodes=[
            graph_pb2.GraphNode(
                node_id="n1",
                label="Machine Learning",
                node_type="Field",
                properties={"description": "Branch of AI"},
                relevance_score=0.95,
            )
        ],
        relationships=[
            graph_pb2.GraphRelationship(
                from_node_id="n1",
                to_node_id="n2",
                relationship_type="SUBFIELD_OF",
                properties={"weight": "1.0"},
            )
        ],
        subgraph_summary="Machine Learning is a subfield of AI.",
        total_tokens=80,
    )

    mock_stub = MagicMock()
    mock_stub.GetGraphContext = AsyncMock(return_value=mock_response)
    client.get_stub = AsyncMock(return_value=mock_stub)
    client._initialized = True

    context: GraphContext = await client.get_graph_context(
        user_id="user_123",
        conversation_id="conv_456",
        query="Machine learning concepts",
    )

    assert len(context.entities) == 1
    assert context.entities[0].id == "n1"
    assert context.entities[0].name == "Machine Learning"
    assert len(context.relationships) == 1
    assert context.relationships[0].relation_type == "SUBFIELD_OF"
    assert context.subgraph_summary == "Machine Learning is a subfield of AI."


@pytest.mark.asyncio
async def test_retrieval_client_get_relevant_chunks_success():
    """Verify RetrievalServiceClient fetch and proto-to-domain conversion."""
    client = RetrievalServiceClient(host="localhost", port=50053, deadline_ms=2000)

    mock_response = retrieval_pb2.GetRelevantChunksResponse(
        chunks=[
            retrieval_pb2.DocumentChunk(
                chunk_id="chk_001",
                content="Quantum computing uses qubits...",
                source_file_id="file_q1",
                source_name="quantum_intro.pdf",
                relevance_score=0.88,
                metadata={"page": "12"},
            )
        ],
        total_tokens=120,
    )

    mock_stub = MagicMock()
    mock_stub.GetRelevantChunks = AsyncMock(return_value=mock_response)
    client.get_stub = AsyncMock(return_value=mock_stub)
    client._initialized = True

    context: RetrievalContext = await client.get_relevant_chunks(
        user_id="user_123",
        conversation_id="conv_456",
        query="Quantum computing fundamentals",
        file_ids=["file_q1"],
    )

    assert len(context.chunks) == 1
    assert context.chunks[0].chunk_id == "chk_001"
    assert context.chunks[0].file_id == "file_q1"
    assert context.chunks[0].score == 0.88
    assert context.total_chunks == 1


@pytest.mark.asyncio
async def test_grpc_client_error_translation():
    """Verify translation of grpc.aio.AioRpcError into domain exceptions."""
    client = MemoryServiceClient(host="localhost", port=50051)

    # 1. Unavailable error
    rpc_err_unavail = grpc.aio.AioRpcError(
        code=grpc.StatusCode.UNAVAILABLE,
        initial_metadata=MagicMock(),
        trailing_metadata=MagicMock(),
        details="Connection refused",
    )
    with pytest.raises(GRPCUnavailableError, match="MemoryService unavailable"):
        client.handle_rpc_error(rpc_err_unavail, service_name="MemoryService")

    # 2. Deadline exceeded error
    rpc_err_timeout = grpc.aio.AioRpcError(
        code=grpc.StatusCode.DEADLINE_EXCEEDED,
        initial_metadata=MagicMock(),
        trailing_metadata=MagicMock(),
        details="Deadline exceeded",
    )
    with pytest.raises(GRPCTimeoutError, match="timed out"):
        client.handle_rpc_error(rpc_err_timeout, service_name="MemoryService")


@pytest.mark.asyncio
async def test_grpc_client_close():
    """Verify close cleanly terminates all connection channels."""
    client = MemoryServiceClient(host="localhost", port=50051, pool_size=2)
    mock_ch1 = AsyncMock()
    mock_ch2 = AsyncMock()
    client._channels = [mock_ch1, mock_ch2]
    client._stubs = [MagicMock(), MagicMock()]
    client._initialized = True

    await client.close()

    mock_ch1.close.assert_called_once()
    mock_ch2.close.assert_called_once()
    assert client._initialized is False
    assert len(client._channels) == 0
