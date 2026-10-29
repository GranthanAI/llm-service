"""A NEW llm-service talking to an OLD memory-service (one that predates retrieved_history / anchors / ancestor summaries):
the missing fields arrive empty and nothing errors. Deploy order is memory-service -> llm-service -> conversation-service, but
the reverse pairing must degrade gracefully, never fail."""
from unittest.mock import AsyncMock

import pytest

import app.pipeline.request_pipeline  # noqa: F401  (import order: avoids a pre-existing cycle in app.grpc.clients)
from app.grpc.clients.memory_client import MemoryServiceClient
from app.grpc.proto import memory_pb2
from app.models.pipeline_context import PipelineContext
from app.prompts.builder import PromptBuilder
from app.prompts.loader import PromptLoader
from app.prompts.registry import PromptRegistry
from app.context.schemas import ContextBundle
from app.workflow_engine.workflow_result import WorkflowResult


def old_server_bytes() -> bytes:
    """What an old memory-service puts on the wire: only the original fields (no retrieved_history, no anchors, no
    ancestor summaries, no summary_version). Built from the new message class but with those fields left unset, which
    serializes to exactly the old wire format (unset repeated/scalar fields are absent)."""
    resp = memory_pb2.GetMemoryContextResponse(
        short_term_messages=[memory_pb2.Message(role="user", content="hello", message_id="m1")],
        medium_summary="An older summary.",
        long_term_facts=[memory_pb2.Fact(content="likes tea", confidence=0.8)],
        total_tokens=0, request_id="r")
    return resp.SerializeToString()


@pytest.mark.asyncio
async def test_new_client_parses_an_old_servers_response_with_empty_new_fields():
    client = MemoryServiceClient.__new__(MemoryServiceClient)
    client.deadline_ms = 1000
    client.build_metadata = lambda **k: []
    stub = type("Stub", (), {})()
    stub.GetMemoryContext = AsyncMock(return_value=memory_pb2.GetMemoryContextResponse.FromString(old_server_bytes()))
    client.get_stub = AsyncMock(return_value=stub)
    ctx = await client.get_memory_context("u", "c", query="q")
    assert [m.content for m in ctx.short_term_messages] == ["hello"] and ctx.summary == "An older summary."
    assert ctx.anchors == [] and ctx.retrieved_history == [] and ctx.ancestor_summaries == []
    assert ctx.summary_version == 0 and [f.statement for f in ctx.long_term_facts] == ["likes tea"]


def test_prompt_builds_without_the_new_sections_when_memory_is_old():
    from app.grpc.proto.memory_pb2 import GetMemoryContextResponse
    resp = GetMemoryContextResponse.FromString(old_server_bytes())
    assert len(resp.anchors) == 0 and len(resp.retrieved_history) == 0
    from app.context.schemas import MemoryContext
    ctx = PipelineContext(conversation_id="C", user_id="u", message_id="m", request_id="r", user_message="q",
                          context_bundle=ContextBundle(memory=MemoryContext(summary=resp.medium_summary)))
    result = WorkflowResult(mode="default", engine_type="mode_handler", draft_content=None, user_message="q")
    composed = PromptBuilder(registry=PromptRegistry(loader=PromptLoader())).build(result, ctx)
    names = {s.name for s in composed.sections}
    assert "conversation_summary" in names and not ({"conversation_anchors", "retrieved_history"} & names)
