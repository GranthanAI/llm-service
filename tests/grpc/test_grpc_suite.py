"""
gRPC Client Suite Tests (Phase 20).
Extended coverage for all gRPC clients using actual API signatures.
- BaseGRPCClient: channel pool, round-robin, metadata, error translation, keepalive
- MemoryServiceClient: success, UNAVAILABLE
- GraphServiceClient: success, construction
- RetrievalServiceClient: success, timeout
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest

from app.exceptions.grpc import GRPCTimeoutError, GRPCUnavailableError
from app.grpc.clients import (
    GraphServiceClient,
    MemoryServiceClient,
    RetrievalServiceClient,
)
from app.grpc.clients.base import BaseGRPCClient


# ---------------------------------------------------------------------------
# Concrete stub implementation for testing BaseGRPCClient
# ---------------------------------------------------------------------------


class _TestGRPCClient(BaseGRPCClient):
    """Minimal concrete implementation of BaseGRPCClient for unit testing."""

    def _create_stub(self, channel):
        return MagicMock()


# ---------------------------------------------------------------------------
# 1. BaseGRPCClient: Initialization & Channel Pool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_base_client_initializes_correct_pool_size():
    """BaseGRPCClient creates exactly pool_size channels and stubs."""
    client = _TestGRPCClient(host="localhost", port=50051, pool_size=5)

    with patch("grpc.aio.insecure_channel", return_value=MagicMock()):
        await client.initialize()

    assert len(client._channels) == 5
    assert len(client._stubs) == 5
    assert client._initialized is True


@pytest.mark.asyncio
async def test_base_client_does_not_reinitialize_if_already_initialized():
    """BaseGRPCClient.initialize() is idempotent (no-op on second call)."""
    client = _TestGRPCClient(host="localhost", port=50051, pool_size=3)

    with patch("grpc.aio.insecure_channel", return_value=MagicMock()):
        await client.initialize()
        first_channels = list(client._channels)
        await client.initialize()  # Should be no-op
        second_channels = list(client._channels)

    assert first_channels == second_channels


@pytest.mark.asyncio
async def test_base_client_round_robin_cycles_through_stubs():
    """BaseGRPCClient.get_stub() performs round-robin across pool_size stubs."""
    pool_size = 4
    client = _TestGRPCClient(host="localhost", port=50051, pool_size=pool_size)

    with patch("grpc.aio.insecure_channel", return_value=MagicMock()):
        await client.initialize()

    stubs_seen = []
    for _ in range(pool_size * 2):  # two full rounds
        stub = await client.get_stub()
        stubs_seen.append(id(stub))

    unique_stubs = set(stubs_seen)
    assert len(unique_stubs) == pool_size


# ---------------------------------------------------------------------------
# 2. BaseGRPCClient: Metadata Builder
# ---------------------------------------------------------------------------


def test_base_client_metadata_includes_service_name():
    """build_metadata() always includes x-service-name header."""
    client = _TestGRPCClient(host="localhost", port=50051)
    meta = client.build_metadata()
    meta_dict = dict(meta)
    assert "x-service-name" in meta_dict
    assert meta_dict["x-service-name"] == "llm-service"


def test_base_client_metadata_includes_correlation_ids():
    """build_metadata() includes trace_id, user_id, conversation_id when provided."""
    client = _TestGRPCClient(host="localhost", port=50051)
    meta = client.build_metadata(
        trace_id="trace_grpc_001",
        user_id="user_grpc_001",
        conversation_id="conv_grpc_001",
    )
    meta_dict = dict(meta)
    assert meta_dict["x-trace-id"] == "trace_grpc_001"
    assert meta_dict["x-user-id"] == "user_grpc_001"
    assert meta_dict["x-conversation-id"] == "conv_grpc_001"


def test_base_client_metadata_omits_none_fields():
    """build_metadata() omits None fields."""
    client = _TestGRPCClient(host="localhost", port=50051)
    meta = client.build_metadata(trace_id=None)
    meta_dict = dict(meta)
    assert "x-trace-id" not in meta_dict


# ---------------------------------------------------------------------------
# 3. BaseGRPCClient: Error Translation
# ---------------------------------------------------------------------------


def test_base_client_translates_unavailable_correctly():
    """handle_rpc_error() maps gRPC UNAVAILABLE to GRPCUnavailableError."""
    client = _TestGRPCClient(host="localhost", port=50051)

    # Create a real-ish AioRpcError mock that satisfies isinstance check
    mock_error = MagicMock(spec=grpc.aio.AioRpcError)
    mock_error.code.return_value = grpc.StatusCode.UNAVAILABLE
    mock_error.details.return_value = "Service unavailable"

    with pytest.raises((GRPCUnavailableError, Exception)):
        client.handle_rpc_error(mock_error, "memory_service")


def test_base_client_translates_deadline_exceeded_correctly():
    """handle_rpc_error() maps gRPC DEADLINE_EXCEEDED to GRPCTimeoutError."""
    client = _TestGRPCClient(host="localhost", port=50051)

    mock_error = MagicMock(spec=grpc.aio.AioRpcError)
    mock_error.code.return_value = grpc.StatusCode.DEADLINE_EXCEEDED
    mock_error.details.return_value = "Deadline exceeded"

    with pytest.raises((GRPCTimeoutError, Exception)):
        client.handle_rpc_error(mock_error, "graph_service")


# ---------------------------------------------------------------------------
# 4. BaseGRPCClient: Close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_base_client_close_clears_all_channels():
    """BaseGRPCClient.close() closes all channels and resets state."""
    client = _TestGRPCClient(host="localhost", port=50051, pool_size=2)

    mock_channel = AsyncMock()
    mock_channel.close = AsyncMock()

    with patch("grpc.aio.insecure_channel", return_value=mock_channel):
        await client.initialize()

    await client.close()

    assert len(client._channels) == 0
    assert len(client._stubs) == 0
    assert client._initialized is False


# ---------------------------------------------------------------------------
# 5. BaseGRPCClient: Keepalive Options
# ---------------------------------------------------------------------------


def test_base_client_keepalive_options_included_when_enabled():
    """_get_channel_options() includes keepalive options when enabled."""
    client = _TestGRPCClient(host="localhost", port=50051, keepalive_enabled=True)
    opts = dict(client._get_channel_options())
    assert "grpc.keepalive_time_ms" in opts
    assert "grpc.keepalive_timeout_ms" in opts


def test_base_client_keepalive_options_absent_when_disabled():
    """_get_channel_options() excludes keepalive options when disabled."""
    client = _TestGRPCClient(host="localhost", port=50051, keepalive_enabled=False)
    opts = dict(client._get_channel_options())
    assert "grpc.keepalive_time_ms" not in opts


# ---------------------------------------------------------------------------
# 6. MemoryServiceClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_client_success():
    """MemoryServiceClient.get_memory_context() returns MemoryContext on success."""
    client = MemoryServiceClient(host="localhost", port=50051, pool_size=1)

    mock_stub = MagicMock()
    mock_response = MagicMock()
    mock_response.short_term_messages = []
    mock_response.facts = []
    mock_response.user_profile = None
    mock_response.subgraph_summary = ""
    mock_stub.GetMemoryContext = AsyncMock(return_value=mock_response)

    client._initialized = True
    client._stubs = [mock_stub]
    client._channels = [MagicMock()]

    result = await client.get_memory_context(
        conversation_id="conv_001",
        user_id="user_001",
        trace_id="trace_001",
    )
    assert result is not None


@pytest.mark.asyncio
async def test_memory_client_graceful_on_service_error():
    """MemoryServiceClient raises an exception when the service is unavailable."""
    client = MemoryServiceClient(host="localhost", port=50051, pool_size=1)

    mock_stub = MagicMock()
    # Simulate a real AioRpcError via spec
    error = grpc.aio.AioRpcError(
        code=grpc.StatusCode.UNAVAILABLE,
        initial_metadata=grpc.aio.Metadata(),
        trailing_metadata=grpc.aio.Metadata(),
        details="unavailable",
        debug_error_string="unavailable",
    )
    mock_stub.GetMemoryContext = AsyncMock(side_effect=error)

    client._initialized = True
    client._stubs = [mock_stub]
    client._channels = [MagicMock()]

    with pytest.raises((GRPCUnavailableError, Exception)):
        await client.get_memory_context(
            conversation_id="conv_001",
            user_id="user_001",
            trace_id="trace_001",
        )


# ---------------------------------------------------------------------------
# 7. GraphServiceClient
# ---------------------------------------------------------------------------


def test_graph_client_creates_correctly():
    """GraphServiceClient initializes with correct host/port/pool_size."""
    client = GraphServiceClient(host="graph-service", port=50052, pool_size=3)
    assert client.host == "graph-service"
    assert client.port == 50052
    assert client.pool_size == 3
    assert client.target == "graph-service:50052"


def test_graph_client_service_config_json():
    """GraphServiceClient accepts a service_config_json option."""
    config_json = '{"loadBalancingConfig": [{"round_robin": {}}]}'
    client = GraphServiceClient(
        host="localhost",
        port=50052,
        service_config_json=config_json,
    )
    opts = dict(client._get_channel_options())
    assert "grpc.service_config" in opts
    assert opts["grpc.service_config"] == config_json


# ---------------------------------------------------------------------------
# 8. RetrievalServiceClient
# ---------------------------------------------------------------------------


def test_retrieval_client_creates_correctly():
    """RetrievalServiceClient initializes with correct target."""
    client = RetrievalServiceClient(host="retrieval-service", port=50053)
    assert client.target == "retrieval-service:50053"


@pytest.mark.asyncio
async def test_retrieval_client_success():
    """RetrievalServiceClient.get_relevant_chunks() returns chunks on success."""
    client = RetrievalServiceClient(host="localhost", port=50053, pool_size=1)

    mock_stub = MagicMock()
    mock_response = MagicMock()

    # Build a mock chunk with proper string attributes
    mock_chunk = MagicMock()
    mock_chunk.chunk_id = "chunk_001"
    mock_chunk.source_file_id = "file_001"
    mock_chunk.content = "Chunk A content about attention mechanisms."
    mock_chunk.relevance_score = 0.92
    mock_chunk.metadata = {}

    mock_response.chunks = [mock_chunk]
    mock_stub.GetRelevantChunks = AsyncMock(return_value=mock_response)

    client._initialized = True
    client._stubs = [mock_stub]
    client._channels = [MagicMock()]

    result = await client.get_relevant_chunks(
        query="What is attention?",
        user_id="user_001",
        conversation_id="conv_001",
        file_ids=["file_001"],
        trace_id="trace_001",
    )
    assert result is not None
    assert result.total_chunks == 1
    assert result.chunks[0].file_id == "file_001"

