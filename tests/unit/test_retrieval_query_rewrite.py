"""Standalone/English retrieval query, previous-turn chunks, full-text for small documents, top_k config and the feedback log.
All against doubles of the retrieval client and the fast model; no live calls."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from app.config.settings import LLMServiceConfig
from app.context.collector import ContextCollector
from app.context.previous_chunks import PreviousChunkStore
from app.context.retrieval_query import RetrievalQueryRewriter, heuristic_query, needs_context
from app.context.schemas import DocumentChunk, RetrievalContext
from app.grpc.clients.retrieval_client import RetrievalServiceClient
from app.grpc.proto import retrieval_pb2
from app.models.pipeline_context import PipelineContext

HIST = [("user", "what is phase 6 about?"), ("assistant", "Phase 6 covers the billing cutover.")]


class FakeModel:
    model = "fake"

    def __init__(self, reply=None, error=None, delay=0.0):
        self.reply, self.error, self.delay, self.calls = reply, error, delay, 0

    async def complete(self, messages):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return json.dumps(self.reply)


# ---- rewriter -------------------------------------------------------------------------------------------------------

async def test_follow_up_is_made_standalone_by_the_model():
    m = FakeModel({"standalone": "check again what phase 6 is about", "english": "what is phase 6 about"})
    rq = await RetrievalQueryRewriter([m]).build("Check again", HIST)
    assert rq.query == "check again what phase 6 is about" and rq.dense_query == "what is phase 6 about"
    assert rq.method == "model" and rq.rewritten


async def test_hinglish_gets_an_english_dense_query_and_keeps_original_as_query():
    m = FakeModel({"standalone": "Phase 6 mein kya karna hai?", "english": "what does phase 6 involve"})
    rq = await RetrievalQueryRewriter([m]).build("Phase 6 mein kya karna hai?", [])
    assert rq.query == "Phase 6 mein kya karna hai?"
    assert rq.dense_query == "what does phase 6 involve"


@pytest.mark.parametrize("model", [FakeModel(error=RuntimeError("429")), FakeModel(delay=1.0), FakeModel(reply="not a dict")])
async def test_model_error_timeout_or_bad_output_falls_back_to_the_heuristic(model):
    rq = await RetrievalQueryRewriter([model], timeout_s=0.05).build("Check again", HIST)
    assert rq.method == "heuristic"
    assert rq.query == "what is phase 6 about? Check again" and rq.dense_query is None


async def test_no_models_uses_heuristic_and_self_contained_message_is_untouched():
    rw = RetrievalQueryRewriter([])
    assert (await rw.build("explain the retry policy for subscription renewals", HIST)).query == \
        "explain the retry policy for subscription renewals"
    assert (await rw.build("explain point 3", HIST)).query.startswith("what is phase 6 about?")


def test_short_message_heuristic():
    assert needs_context("Check again") and needs_context("ok") and needs_context("explain point 3")
    assert not needs_context("what is the MAX_RETRY_ATTEMPTS default value")
    assert heuristic_query("Check again", HIST) == "what is phase 6 about? Check again"
    assert heuristic_query("Check again", []) == "Check again"


async def test_turn_window_is_limited():
    seen = {}

    class Spy(FakeModel):
        async def complete(self, messages):
            seen["p"] = messages[1]["content"]
            return json.dumps({"standalone": "x", "english": "y"})

    hist = [(r, f"{r}-{i}") for i in range(10) for r in ("user", "assistant")]
    await RetrievalQueryRewriter([Spy()], turns=2).build("hello there friend", hist)
    assert "user-9" in seen["p"] and "user-8" in seen["p"] and "user-7" not in seen["p"]


# ---- collector ------------------------------------------------------------------------------------------------------

def _chunk(i):
    return DocumentChunk(chunk_id=f"c{i}", file_id="f1", content="secret document text", score=0.5)


def _ctx(msg="Check again", history=HIST, conv="conv-1"):
    return PipelineContext(conversation_id=conv, user_id="u", message_id="m", request_id="r", trace_id="t1",
                           user_message=msg, file_ids=["f1"], event_history=list(history))


def _collector(retrieval, rewriter=None, **cfg):
    c = ContextCollector(memory_client=MagicMock(), graph_client=MagicMock(), retrieval_client=retrieval,
                         query_rewriter=rewriter, config=LLMServiceConfig(**cfg))
    c.logger = MagicMock()
    return c


def _retrieval(chunks=(1, 2), **kw):
    rc = MagicMock()
    rc.get_relevant_chunks = AsyncMock(return_value=RetrievalContext(status="OK", chunks=[_chunk(i) for i in chunks], **kw))
    rc.get_document_chunks = AsyncMock(side_effect=grpc.aio.AioRpcError(
        grpc.StatusCode.UNIMPLEMENTED, grpc.aio.Metadata(), grpc.aio.Metadata(), "unimplemented"))
    return rc


async def test_collector_sends_original_query_and_english_dense_query():
    rc = _retrieval()
    m = FakeModel({"standalone": "Phase 6 mein kya karna hai?", "english": "what does phase 6 involve"})
    await _collector(rc, RetrievalQueryRewriter([m]))._fetch_retrieval(_ctx("Phase 6 mein kya karna hai?", []))
    kw = rc.get_relevant_chunks.await_args.kwargs
    assert kw["query"] == "Phase 6 mein kya karna hai?" and kw["dense_query"] == "what does phase 6 involve"
    assert kw["top_k"] == 8 and kw["max_tokens"] == 4000


async def test_follow_up_query_contains_the_previous_question():
    rc = _retrieval()
    await _collector(rc, RetrievalQueryRewriter([FakeModel(error=RuntimeError())]))._fetch_retrieval(_ctx())
    assert "phase 6" in rc.get_relevant_chunks.await_args.kwargs["query"]


async def test_rewrite_disabled_sends_the_raw_message_and_never_calls_the_model():
    rc, m = _retrieval(), FakeModel({"standalone": "x", "english": "y"})
    await _collector(rc, RetrievalQueryRewriter([m]), retrieval_query_rewrite_enabled=False)._fetch_retrieval(_ctx())
    kw = rc.get_relevant_chunks.await_args.kwargs
    assert kw["query"] == "Check again" and not kw["dense_query"] and m.calls == 0


async def test_top_k_and_max_tokens_come_from_config():
    rc = _retrieval()
    await _collector(rc, retrieval_top_k=10, retrieval_max_tokens=1234)._fetch_retrieval(_ctx())
    kw = rc.get_relevant_chunks.await_args.kwargs
    assert (kw["top_k"], kw["max_tokens"]) == (10, 1234)


async def test_previous_chunk_ids_are_sent_on_the_next_turn_of_the_same_conversation_only():
    rc = _retrieval(chunks=(1, 2, 3))
    col = _collector(rc)
    await col._fetch_retrieval(_ctx(conv="A"))
    assert rc.get_relevant_chunks.await_args.kwargs["previous_chunk_ids"] == []
    await col._fetch_retrieval(_ctx("explain point 3", conv="A"))
    assert rc.get_relevant_chunks.await_args.kwargs["previous_chunk_ids"] == ["c1", "c2", "c3"]
    await col._fetch_retrieval(_ctx(conv="B"))
    assert rc.get_relevant_chunks.await_args.kwargs["previous_chunk_ids"] == []


async def test_previous_chunks_flag_off_sends_nothing():
    rc = _retrieval()
    col = _collector(rc, retrieval_previous_chunks_enabled=False)
    await col._fetch_retrieval(_ctx())
    await col._fetch_retrieval(_ctx())
    assert rc.get_relevant_chunks.await_args.kwargs["previous_chunk_ids"] == []


def test_previous_chunk_store_ttl_and_empty_turn():
    s = PreviousChunkStore(ttl_s=-1.0)
    s.record("c", ["a"])
    assert s.get("c") == []
    s = PreviousChunkStore()
    s.record("c", ["a"])
    s.record("c", [])
    assert s.get("c") == ["a"]


async def test_full_text_small_documents_skip_topk():
    rc = _retrieval()
    full = RetrievalContext(status="OK", chunks=[_chunk(1), _chunk(2)], full_text=True, truncated=False)
    rc.get_document_chunks = AsyncMock(return_value=full)
    res = await _collector(rc, retrieval_full_text_enabled=True, prompt_token_budget=8000)._fetch_retrieval(_ctx())
    assert res.full_text and rc.get_document_chunks.await_args.kwargs["max_tokens"] == 2000   # 0.25 x 8000
    rc.get_relevant_chunks.assert_not_called()


async def test_full_text_truncated_or_unimplemented_falls_back_to_topk():
    rc = _retrieval()
    res = await _collector(rc, retrieval_full_text_enabled=True)._fetch_retrieval(_ctx())     # UNIMPLEMENTED
    assert not res.full_text and rc.get_relevant_chunks.await_count == 1
    rc.get_document_chunks = AsyncMock(return_value=RetrievalContext(status="OK", chunks=[_chunk(1)], full_text=True, truncated=True))
    res = await _collector(rc, retrieval_full_text_enabled=True)._fetch_retrieval(_ctx())
    assert not res.full_text and rc.get_relevant_chunks.await_count == 2


async def test_full_text_off_by_default_never_calls_the_rpc():
    rc = _retrieval()
    await _collector(rc)._fetch_retrieval(_ctx())
    rc.get_document_chunks.assert_not_called()


async def test_feedback_log_has_hash_not_text_unless_flagged_and_never_content():
    rc = _retrieval(coverage="WEAK", top_score=0.2)
    col = _collector(rc)
    await col._fetch_retrieval(_ctx("secret question here"))
    call = [c for c in col.logger.info.call_args_list if c.args[0] == "retrieval_feedback"][0]
    f = call.kwargs
    assert f["trace_id"] == "t1" and len(f["query_hash"]) == 16 and f["coverage"] == "WEAK"
    assert f["chunk_ids"] == ["c1", "c2"] and "latency_ms" in f
    assert "query_text" not in f and "secret" not in json.dumps(f, default=str)

    col = _collector(rc, log_query_text=True)
    await col._fetch_retrieval(_ctx("secret question here"))
    f = [c for c in col.logger.info.call_args_list if c.args[0] == "retrieval_feedback"][0].kwargs
    assert f["query_text"] == "secret question here"


# ---- gRPC client against the (possibly old) server ------------------------------------------------------------------

def _client(stub):
    c = RetrievalServiceClient(host="localhost", port=50060, pool_size=1)
    c._initialized, c._stubs, c._channels = True, [stub], [MagicMock()]
    return c


async def test_client_sends_dense_query_and_previous_ids_and_reads_coverage():
    stub = MagicMock()
    stub.GetRelevantChunks = AsyncMock(return_value=retrieval_pb2.GetRelevantChunksResponse(
        coverage=retrieval_pb2.WEAK, top_score=0.3,
        chunks=[retrieval_pb2.DocumentChunk(chunk_id="c1", content="x", source_file_id="f1")]))
    res = await _client(stub).get_relevant_chunks(user_id="u", query="q", file_ids=["f1"], top_k=8,
                                                  dense_query="english q", previous_chunk_ids=["a", "b"])
    req = stub.GetRelevantChunks.await_args.args[0]
    assert req.dense_query == "english q" and list(req.previous_chunk_ids) == ["a", "b"] and req.top_k == 8
    assert res.coverage == "WEAK" and res.top_score == pytest.approx(0.3)


async def test_old_server_response_reads_as_no_signal_and_dense_query_is_unset_when_absent():
    stub = MagicMock()
    stub.GetRelevantChunks = AsyncMock(return_value=retrieval_pb2.GetRelevantChunksResponse(
        chunks=[retrieval_pb2.DocumentChunk(chunk_id="c1", content="x", source_file_id="f1")]))
    res = await _client(stub).get_relevant_chunks(user_id="u", query="q", file_ids=["f1"])
    assert res.coverage == "COVERAGE_UNSPECIFIED" and res.top_score is None
    assert not stub.GetRelevantChunks.await_args.args[0].HasField("dense_query")


async def test_get_document_chunks_maps_truncated():
    stub = MagicMock()
    stub.GetDocumentChunks = AsyncMock(return_value=retrieval_pb2.GetDocumentChunksResponse(
        truncated=True, total_tokens=900, chunks=[retrieval_pb2.DocumentChunk(chunk_id="c1", content="x", source_file_id="f1")]))
    res = await _client(stub).get_document_chunks(user_id="u", file_ids=["f1"], max_tokens=1000)
    assert res.truncated and res.full_text and res.total_tokens == 900 and len(res.chunks) == 1
