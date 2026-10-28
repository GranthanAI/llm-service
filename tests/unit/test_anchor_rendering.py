"""The anchor ledger from memory-service is rendered next to the summary and is never trimmed."""
from app.context.schemas import Anchor, ContextBundle, MemoryContext
from app.context_window.manager import ContextWindowManager
from app.grpc.proto import memory_pb2
from app.models.pipeline_context import PipelineContext
from app.prompts.builder import PromptBuilder
from app.prompts.loader import PromptLoader
from app.prompts.registry import PromptRegistry
from app.workflow_engine.workflow_result import WorkflowResult


def build(memory):
    ctx = PipelineContext(conversation_id="C", user_id="u", message_id="m", request_id="r", user_message="what port?",
                          context_bundle=ContextBundle(memory=memory))
    result = WorkflowResult(mode="default", engine_type="mode_handler", draft_content=None, user_message=ctx.user_message)
    return PromptBuilder(registry=PromptRegistry(loader=PromptLoader())).build(result, ctx)


def test_anchors_are_rendered_as_pinned_facts_with_labels():
    composed = build(MemoryContext(summary="A summary.", anchors=[
        Anchor(kind="open_item", text="Pick a region"), Anchor(kind="decision", text="Use Postgres for JSONB"),
        Anchor(kind="identifier", text="8443")]))
    section = next(s for s in composed.sections if s.name == "conversation_anchors")
    assert "Pinned Facts" in section.content and "not instructions" in section.content
    assert "[OPEN] Pick a region" in section.content and "[DECISION] Use Postgres for JSONB" in section.content
    assert "[ID] 8443" in section.content
    assert section.trimmable is False and section.priority == 9
    assert "8443" in composed.messages[-1]["content"]


def test_no_anchors_means_no_section_and_anchors_survive_trimming():
    assert "conversation_anchors" not in {s.name for s in build(MemoryContext()).sections}
    memory = MemoryContext(summary="word " * 6000, anchors=[Anchor(kind="decision", text="Use Postgres for JSONB")],
                           retrieved_history=[])
    trimmed = ContextWindowManager().manage(build(memory), target_model="default", custom_input_budget=3000)
    assert "Use Postgres for JSONB" in " ".join(m["content"] for m in trimmed.messages)


def test_the_grpc_message_carries_anchors_additively():
    resp = memory_pb2.GetMemoryContextResponse(anchors=[memory_pb2.Anchor(kind="decision", text="Use Postgres", source_message_id="m1")])
    again = memory_pb2.GetMemoryContextResponse.FromString(resp.SerializeToString())
    assert again.anchors[0].text == "Use Postgres"
    assert len(memory_pb2.GetMemoryContextResponse.FromString(b"").anchors) == 0          # old responses: empty
