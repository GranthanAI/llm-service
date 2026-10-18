"""Memory layers survive the gRPC boundary and reach the prompt as separate, labelled sections:
conversation summary, branch-valid ancestor summaries, long-term facts. Branch lineage travels from
the request event to the memory service."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.context.merger import ContextMerger
from app.context.schemas import ContextBundle, DocumentChunk, Fact, MemoryContext, RetrievalContext
from app.grpc.clients.memory_client import MemoryServiceClient
from app.grpc.proto import memory_pb2
from app.models.pipeline_context import PipelineContext
from app.models.request import ChatMessageCreatedEvent
from app.prompts.builder import PromptBuilder
from app.prompts.loader import PromptLoader
from app.prompts.registry import PromptRegistry
from app.workflow_engine.workflow_result import WorkflowResult


def _response():
    return memory_pb2.GetMemoryContextResponse(
        short_term_messages=[memory_pb2.Message(role="user", content="recent question", timestamp=1700000000000, message_id="m9")],
        medium_summary="Earlier the user set up a FastAPI service.",
        summary_version=4,
        summary_through_message_id="m5",
        long_term_facts=[memory_pb2.Fact(content="User prefers concise answers", confidence=0.8, category="preferences", fact_id="f-1")],
        ancestor_summaries=[
            memory_pb2.AncestorSummary(conversation_id="A", summary="A: discussed KV cache.", covers_through_message_id="a3", summary_version=2),
            memory_pb2.AncestorSummary(conversation_id="B", summary="B: asked about batching.", covers_through_message_id="b2", summary_version=1),
        ],
    )


@pytest.mark.asyncio
async def test_every_memory_layer_survives_the_grpc_round_trip_and_lineage_is_sent():
    client = MemoryServiceClient(host="localhost", port=50051, pool_size=1)
    stub = MagicMock()
    # the response really goes through protobuf serialization
    wire = memory_pb2.GetMemoryContextResponse.FromString(_response().SerializeToString())
    stub.GetMemoryContext = AsyncMock(return_value=wire)
    client._initialized, client._stubs, client._channels = True, [stub], [MagicMock()]

    ctx = await client.get_memory_context(
        user_id="u", conversation_id="C", query="q", trace_id="t",
        ancestors=[{"conversation_id": "A", "upto_message_id": "a3"}, {"conversation_id": "B", "upto_message_id": "b2"}],
    )

    request = stub.GetMemoryContext.call_args.args[0]
    assert [(a.conversation_id, a.upto_message_id) for a in request.ancestors] == [("A", "a3"), ("B", "b2")]
    assert [m.content for m in ctx.short_term_messages] == ["recent question"]
    assert ctx.summary == "Earlier the user set up a FastAPI service."
    assert (ctx.summary_version, ctx.summary_through_message_id) == (4, "m5")
    assert [(a.conversation_id, a.summary, a.covers_through_message_id) for a in ctx.ancestor_summaries] == [
        ("A", "A: discussed KV cache.", "a3"), ("B", "B: asked about batching.", "b2")]
    assert ctx.long_term_facts[0].fact_id == "f-1"


def test_the_event_carries_lineage_into_the_pipeline_context():
    event = ChatMessageCreatedEvent(
        conversation_id="C", user_id="u", message_id="m", response_message_id="r", content="hi",
        ancestors=[{"conversation_id": "A", "upto_message_id": "a3"}])
    ctx = PipelineContext.from_event(event, request_id="r1")
    assert ctx.ancestors == [{"conversation_id": "A", "upto_message_id": "a3"}]
    older = ChatMessageCreatedEvent(conversation_id="C", user_id="u", message_id="m", response_message_id="r", content="hi")
    assert PipelineContext.from_event(older, request_id="r1").ancestors == []


def test_merger_keeps_summaries_and_does_not_collapse_facts_that_only_share_a_prefix():
    prefix = "User is working on a large distributed system and prefers " + "x" * 150
    memory = MemoryContext(
        summary="s", summary_version=2,
        long_term_facts=[Fact(fact_id="a", statement=prefix + " Python"), Fact(fact_id="b", statement=prefix + " Rust"),
                         Fact(fact_id="c", statement="  user PREFERS concise answers. "), Fact(fact_id="d", statement="User prefers concise answers")])
    merged = ContextMerger().merge(memory=memory).memory
    assert merged.summary == "s" and merged.summary_version == 2
    assert [f.fact_id for f in merged.long_term_facts] == ["a", "b", "c"]          # d is a true duplicate of c


def test_long_chunks_with_the_same_opening_are_both_kept():
    head = "Section 4 describes the retry policy in detail. " * 8
    chunks = [DocumentChunk(chunk_id="1", file_id="f", content=head + "Retries are capped at three.", score=0.9),
              DocumentChunk(chunk_id="2", file_id="f", content=head + "Backoff is exponential.", score=0.8),
              DocumentChunk(chunk_id="3", file_id="f", content=head + "Backoff is exponential.", score=0.7)]
    merged = ContextMerger().merge(retrieval=RetrievalContext(chunks=chunks, total_chunks=3, status="OK")).retrieval
    assert [c.chunk_id for c in merged.chunks] == ["1", "2"]


def _compose(memory, mode="default"):
    ctx = PipelineContext(conversation_id="C", user_id="u", message_id="m", request_id="r", user_message="and the second one?",
                          context_bundle=ContextBundle(memory=memory))
    result = WorkflowResult(mode=mode, engine_type="mode_handler", draft_content=None, user_message=ctx.user_message)
    return PromptBuilder(registry=PromptRegistry(loader=PromptLoader())).build(result, ctx)


def test_prompt_has_separate_labelled_layers_in_a_fixed_order():
    composed = _compose(MemoryContext.model_validate({
        "summary": "Earlier: FastAPI setup.",
        "ancestor_summaries": [{"conversation_id": "A", "summary": "A: KV cache."}, {"conversation_id": "B", "summary": "B: batching."}],
        "long_term_facts": [{"fact_id": "f", "statement": "User prefers concise answers"}],
        "short_term_messages": [{"role": "user", "content": "recent question"}]}))
    user = composed.messages[1]["content"]
    order = [user.index(h) for h in ("## What I Know About You", "## Conversation Summary", "## Inherited Context",
                                     "## Recent Conversation History", "## Current User Message")]
    assert order == sorted(order)
    assert "### root conversation (1/2)" in user and "### parent conversation (2/2)" in user
    assert "nothing said after the branch point is included" in user
    assert {s.name for s in composed.sections} >= {"conversation_summary", "inherited_context", "long_term_memory", "user_query"}


def test_empty_layers_are_not_rendered_and_untrusted_text_is_cleaned():
    assert {"conversation_summary", "inherited_context"}.isdisjoint({s.name for s in _compose(MemoryContext()).sections})
    composed = _compose(MemoryContext.model_validate({"summary": "ok\x00 text\x07"}))
    assert "\x00" not in composed.messages[1]["content"] and "\x07" not in composed.messages[1]["content"]


def test_the_current_request_is_never_trimmed_away_with_huge_summaries():
    big = "word " * 20000
    composed = _compose(MemoryContext.model_validate({"summary": big, "ancestor_summaries": [{"conversation_id": "A", "summary": big}]}))
    user_query = next(s for s in composed.sections if s.name == "user_query")
    assert "and the second one?" in user_query.content and user_query.trimmable is False
    assert next(s for s in composed.sections if s.name == "system").trimmable is False
