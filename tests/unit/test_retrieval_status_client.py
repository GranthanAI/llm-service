"""RetrievalServiceClient maps the Retrieval Service status onto RetrievalContext, and the
ContextCollector retries a transient status once but never retries a definitive one."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.context.collector import ContextCollector  # noqa: F401  (import order: avoids a package cycle)
from app.context.schemas import RetrievalContext
from app.grpc.clients.retrieval_client import RetrievalServiceClient
from app.grpc.proto import retrieval_pb2
from app.models.pipeline_context import PipelineContext


def _client(response):
    client = RetrievalServiceClient(host="localhost", port=50060, pool_size=1)
    stub = MagicMock()
    stub.GetRelevantChunks = AsyncMock(return_value=response)
    client._initialized, client._stubs, client._channels = True, [stub], [MagicMock()]
    return client, stub


def _response(status, chunks=(), detail="", requested=1, authorized=1, ready=1):
    return retrieval_pb2.GetRelevantChunksResponse(
        chunks=list(chunks), status=status, status_detail=detail,
        requested_count=requested, authorized_count=authorized, ready_count=ready,
    )


def _chunk():
    return retrieval_pb2.DocumentChunk(
        chunk_id="c1", content="The vault code is X", source_file_id="doc-1", source_name="handbook.pdf",
        relevance_score=0.8, metadata={"page_number": "3", "section": "Security"},
    )


@pytest.mark.parametrize("name", ["ACCESS_DENIED", "NOT_INDEXED", "INDEXING", "INDEXING_FAILED", "UNAVAILABLE", "TIMEOUT"])
async def test_status_is_carried_into_the_context(name):
    client, _ = _client(_response(getattr(retrieval_pb2, name), detail="why"))
    ctx = await client.get_relevant_chunks(user_id="u", query="q", file_ids=["doc-1"])
    assert ctx.status == name and ctx.status_detail == "why" and ctx.chunks == []


async def test_chunks_and_counts_and_citation_fields_are_mapped():
    client, _ = _client(_response(retrieval_pb2.OK, [_chunk()], requested=2, authorized=2, ready=1))
    ctx = await client.get_relevant_chunks(user_id="u", query="q", file_ids=["doc-1", "doc-2"])
    assert ctx.status == "OK" and (ctx.requested_count, ctx.authorized_count, ctx.ready_count) == (2, 2, 1)
    c = ctx.chunks[0]
    assert (c.file_id, c.source_name, c.page_number, c.section) == ("doc-1", "handbook.pdf", 3, "Security")


async def test_old_server_without_status_and_no_chunks_reads_as_no_match():
    client, _ = _client(retrieval_pb2.GetRelevantChunksResponse())  # status unset == OK, zero chunks
    ctx = await client.get_relevant_chunks(user_id="u", query="q", file_ids=["doc-1"])
    assert ctx.status == "NO_MATCH"


def _collector(client):
    c = ContextCollector(memory_client=MagicMock(), graph_client=MagicMock(), retrieval_client=client)
    c.logger = MagicMock()
    return c


def _ctx():
    return PipelineContext(conversation_id="c", user_id="u", message_id="m", request_id="r",
                           user_message="what is the code?", file_ids=["doc-1"])


async def test_collector_retries_a_transient_status_once_then_succeeds():
    rc = MagicMock()
    rc.get_relevant_chunks = AsyncMock(side_effect=[
        RetrievalContext(status="TIMEOUT"),
        RetrievalContext(status="OK", chunks=[], total_chunks=0, ready_count=1),
    ])
    res = await _collector(rc)._fetch_retrieval(_ctx())
    assert res.status == "OK" and rc.get_relevant_chunks.await_count == 2


async def test_collector_gives_up_after_one_retry():
    rc = MagicMock()
    rc.get_relevant_chunks = AsyncMock(return_value=RetrievalContext(status="UNAVAILABLE"))
    res = await _collector(rc)._fetch_retrieval(_ctx())
    assert res.status == "UNAVAILABLE" and rc.get_relevant_chunks.await_count == 2


@pytest.mark.parametrize("status", ["ACCESS_DENIED", "INDEXING", "NO_MATCH", "OK"])
async def test_definitive_statuses_are_not_retried(status):
    rc = MagicMock()
    rc.get_relevant_chunks = AsyncMock(return_value=RetrievalContext(status=status))
    await _collector(rc)._fetch_retrieval(_ctx())
    assert rc.get_relevant_chunks.await_count == 1
