"""The additive retrieval.proto fields and RPC (query rewrite, previous-turn chunks, coverage, full-text fetch). llm-service vendors this
proto; the retrieval-service copy is the source of truth and the two must stay byte-identical."""
from app.context.collector import ContextCollector  # noqa: F401  (import order: avoids a package cycle)
from app.grpc.proto import retrieval_pb2 as r


def test_request_has_optional_dense_query_and_previous_chunk_ids():
    req = r.GetRelevantChunksRequest(user_id="u", query="q", top_k=8, dense_query="dq", previous_chunk_ids=["a", "b"])
    assert req.dense_query == "dq" and list(req.previous_chunk_ids) == ["a", "b"]
    assert not r.GetRelevantChunksRequest(query="q").HasField("dense_query")  # absent stays absent: old callers are unaffected


def test_response_has_optional_top_score_and_a_coverage_enum_that_defaults_to_unspecified():
    empty = r.GetRelevantChunksResponse()
    assert empty.coverage == r.COVERAGE_UNSPECIFIED and not empty.HasField("top_score")
    weak = r.GetRelevantChunksResponse(coverage=r.WEAK, top_score=0.3)
    assert weak.coverage == r.Coverage.Value("WEAK") and weak.HasField("top_score")
    assert {v.name for v in r.Coverage.DESCRIPTOR.values} == {"COVERAGE_UNSPECIFIED", "STRONG", "WEAK", "NONE"}


def test_get_document_chunks_rpc_exists_and_get_relevant_chunks_is_unchanged():
    methods = {m.name for m in r.DESCRIPTOR.services_by_name["RetrievalService"].methods}
    assert methods == {"GetRelevantChunks", "GetDocumentChunks"}
    req = r.GetDocumentChunksRequest(user_id="u", file_ids=["f1"], max_tokens=1000, trace_id="t")
    assert list(req.file_ids) == ["f1"]
    assert hasattr(r.GetDocumentChunksResponse(), "truncated")
