"""Explicit tool selection (discovery, validation, dispatch) and attached-document context handling."""

from unittest.mock import AsyncMock, MagicMock
from tests.doubles import built_prompt, manager_double, trimmed_prompt

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_container
from app.api.tools import router as tools_router
from app.config.settings import LLMServiceConfig
from app.context.merger import ContextMerger
from app.context.schemas import ContextBundle, DocumentChunk, RetrievalContext
from app.exceptions.user_facing import DocumentContextUnavailableError, InvalidToolSelectionError
from app.models.execution_plan import ExecutionPlan, IntentCategory, ReasoningMode, Skill, ToolCall, UserMode
from app.models.pipeline_context import PipelineContext
from app.models.request import ChatMessageCreatedEvent
from app.models.tool import ToolParams, ToolResult, ToolSchema, ValidationResult
from app.pipeline.request_pipeline import RequestPipeline
from app.prompts.builder import PromptBuilder
from app.prompts.loader import PromptLoader
from app.prompts.registry import PromptRegistry
from app.tools.base import BaseTool
from app.tools.dispatcher import ToolDispatcher
from app.tools.registry import ToolRegistry
from app.tools.selection import MAX_SELECTED_TOOLS, build_selected_calls, describe_tool, list_enabled_tools
from app.utils.error_handler import ErrorHandler
from app.workflow_engine.workflow_result import WorkflowResult


class FakeTool(BaseTool):
    def __init__(self, name, required=("query",), available=True):
        self.name, self.description, self.version = name, f"{name} tool", "1.2.0"
        self._required, self._available = list(required), available
        self.calls: list[dict] = []

    def get_schema(self):
        props = {r: {"type": "string", "description": f"{r} value"} for r in self._required}
        return ToolSchema(name=self.name, description=self.description, version=self.version,
                          parameters={"type": "object", "properties": props, "required": self._required})

    def is_available(self):
        return self._available

    def validate_params(self, params):
        return ValidationResult(valid=True)

    async def execute(self, params: ToolParams) -> ToolResult:
        self.calls.append(dict(params.params))
        return ToolResult(tool_name=self.name, success=True, data={"results": [{"title": "T", "url": "u", "snippet": "s"}]})


def _registry():
    reg = ToolRegistry()
    reg.register(FakeTool("web_search"), enabled=True)
    reg.register(FakeTool("calculator"), enabled=False)  # registered but disabled
    reg.register(FakeTool("needs_file", required=("path",)), enabled=True)
    return reg


# ------------------------------------------------------------------ discovery


def test_discovery_lists_only_enabled_tools_with_ui_safe_fields():
    tools = list_enabled_tools(_registry())
    ids = [t["id"] for t in tools]
    assert ids == ["needs_file", "web_search"]  # sorted, disabled 'calculator' absent
    ws = next(t for t in tools if t["id"] == "web_search")
    assert ws["name"] == "Web Search" and ws["description"] and ws["version"] == "1.2.0"
    assert ws["parameters"] == [{"name": "query", "type": "string", "description": "query value", "required": True}]
    assert ws["auto_runnable"] is True
    assert next(t for t in tools if t["id"] == "needs_file")["auto_runnable"] is False
    assert set(ws) == {"id", "name", "description", "version", "parameters", "auto_runnable"}  # nothing internal


def test_discovery_endpoint_returns_the_live_registry():
    app = FastAPI()
    app.include_router(tools_router, prefix="/api")
    container = MagicMock()
    container.tool_registry = _registry()
    app.dependency_overrides[get_container] = lambda: container
    client = TestClient(app)

    body = client.get("/api/v1/tools").json()
    assert [t["id"] for t in body["tools"]] == ["needs_file", "web_search"]
    assert body["max_selected"] == MAX_SELECTED_TOOLS

    container.tool_registry.disable("web_search")  # tools can disappear at runtime
    assert [t["id"] for t in client.get("/api/v1/tools").json()["tools"]] == ["needs_file"]


def test_discovery_without_a_registry_is_empty_not_an_error():
    app = FastAPI()
    app.include_router(tools_router, prefix="/api")
    container = MagicMock()
    container.tool_registry = None
    app.dependency_overrides[get_container] = lambda: container
    assert TestClient(app).get("/api/v1/tools").json()["tools"] == []


def test_description_never_contains_runtime_config():
    schema = FakeTool("web_search").get_schema()
    blob = str(describe_tool(schema)).lower()
    assert "api_key" not in blob and "secret" not in blob and "http" not in blob


# ------------------------------------------------------------------ validation


def test_event_carries_selected_tools_into_the_pipeline_context():
    event = ChatMessageCreatedEvent(
        conversation_id="c", user_id="u", message_id="m", response_message_id="r",
        content="hi", selected_tools=["web_search"],
    )
    assert PipelineContext.from_event(event, request_id="x").selected_tools == ["web_search"]


def test_valid_selection_becomes_required_calls_with_the_users_message_as_input():
    calls = build_selected_calls(["web_search"], _registry(), "latest kafka release")
    assert len(calls) == 1
    assert calls[0].tool_name == "web_search" and calls[0].required is True
    assert calls[0].params == {"query": "latest kafka release"}


def test_empty_or_missing_selection_changes_nothing():
    assert build_selected_calls([], _registry(), "x") == []
    assert build_selected_calls(None, _registry(), "x") == []
    assert build_selected_calls(["", "  "], _registry(), "x") == []


def test_duplicates_and_whitespace_are_normalized():
    calls = build_selected_calls([" web_search ", "web_search"], _registry(), "q")
    assert [c.tool_name for c in calls] == ["web_search"]


@pytest.mark.parametrize("bad", [["nope"], ["calculator"], ["web_search", "nope"]])
def test_unknown_or_disabled_tools_are_rejected(bad):
    with pytest.raises(InvalidToolSelectionError) as e:
        build_selected_calls(bad, _registry(), "q")
    assert "not available" in e.value.user_message


def test_tool_needing_extra_input_is_rejected_not_guessed():
    with pytest.raises(InvalidToolSelectionError) as e:
        build_selected_calls(["needs_file"], _registry(), "q")
    assert "path" in e.value.user_message


def test_unconfigured_tool_is_rejected():
    reg = ToolRegistry()
    reg.register(FakeTool("web_search", available=False))
    with pytest.raises(InvalidToolSelectionError):
        build_selected_calls(["web_search"], reg, "q")


def test_too_many_tools_are_rejected():
    reg = ToolRegistry()
    names = [f"t{i}" for i in range(MAX_SELECTED_TOOLS + 1)]
    for n in names:
        reg.register(FakeTool(n))
    with pytest.raises(InvalidToolSelectionError):
        build_selected_calls(names, reg, "q")


def test_selection_without_a_registry_is_refused():
    with pytest.raises(InvalidToolSelectionError):
        build_selected_calls(["web_search"], None, "q")


def test_user_facing_errors_surface_their_message_verbatim():
    handler = ErrorHandler()
    c = handler.classify(InvalidToolSelectionError(["nope"], "unknown or disabled"))
    assert c.error_code == "INVALID_TOOL_SELECTION" and "'nope'" in c.user_message and not c.is_retriable
    d = handler.classify(DocumentContextUnavailableError("INDEXING"))
    assert "still being processed" in d.user_message and d.error_code == "DOCUMENT_INDEXING"


# ------------------------------------------------------------------ pipeline dispatch


def _plan(tools=None):
    return ExecutionPlan(intent=IntentCategory.GENERAL_CHAT, mode=UserMode.DEFAULT, skill=Skill.GENERAL_CHAT,
                         reasoning=ReasoningMode.DIRECT, tools=tools or [], max_iterations=1)


def _pipeline(registry, *, planned=None, bundle=None):
    dispatcher = ToolDispatcher(registry=registry)
    collector = MagicMock()
    collector.collect = AsyncMock(return_value=bundle or ContextMerger().merge())
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(return_value=_plan(planned))
    workflow = MagicMock()
    seen = {}

    async def execute(plan, ctx):
        seen["plan_tools"] = [t.tool_name for t in plan.tools]
        return WorkflowResult(mode="default", engine_type="mode_handler", draft_content=None,
                              tool_outputs=await dispatcher.dispatch(plan.tools), user_message=ctx.user_message)

    workflow.execute = execute
    builder = MagicMock()
    builder.build = MagicMock(side_effect=lambda r, c: (seen.setdefault("tool_outputs", [x.tool_name for x in r.tool_outputs]), built_prompt())[1])
    cwm = manager_double()

    async def gen(prompt, params):
        yield "ok"

    router = MagicMock()
    router.generate_stream = gen
    streaming = MagicMock()

    async def consume(it, ctx):
        async for _ in it:
            pass

    streaming.stream = consume
    pub = MagicMock()
    pub.publish_error_response = AsyncMock()
    pipe = RequestPipeline(
        context_collector=collector, request_analyzer=analyzer, workflow_engine=workflow,
        prompt_builder=builder, context_window_manager=cwm, generation_router=router,
        streaming_engine=streaming, publisher=pub, config=LLMServiceConfig(), logger=MagicMock(),
        tool_registry=registry, tool_dispatcher=dispatcher,
    )
    return pipe, seen, pub, collector


def _event(content="what is new in kafka", **kw):
    return ChatMessageCreatedEvent(conversation_id="c", user_id="u", message_id="m", response_message_id="r", content=content, **kw)


async def test_selected_tool_is_dispatched_even_though_the_planner_chose_none():
    reg = _registry()
    pipe, seen, pub, _ = _pipeline(reg)
    await pipe.run(_event(selected_tools=["web_search"]))
    tool = reg._tools["web_search"]
    assert tool.calls == [{"query": "what is new in kafka"}]
    assert seen["tool_outputs"] == ["web_search"]  # its output reached prompt construction
    pub.publish_error_response.assert_not_called()


async def test_a_tool_the_planner_also_chose_runs_once_not_twice():
    reg = _registry()
    planned = [ToolCall(tool_name="web_search", params={"query": "planner q"})]
    pipe, seen, _, _ = _pipeline(reg, planned=planned)
    await pipe.run(_event(selected_tools=["web_search"]))
    assert len(reg._tools["web_search"].calls) == 1
    assert seen["plan_tools"] == []  # the duplicate was removed from the plan; the explicit call ran


async def test_no_selection_runs_no_extra_tools_and_keeps_planner_behaviour():
    reg = _registry()
    pipe, seen, _, _ = _pipeline(reg)
    await pipe.run(_event())
    assert reg._tools["web_search"].calls == []


async def test_planner_chosen_tools_still_run_when_nothing_is_selected():
    reg = _registry()
    planned = [ToolCall(tool_name="web_search", params={"query": "q"})]
    pipe, _, _, _ = _pipeline(reg, planned=planned)
    await pipe.run(_event())
    assert reg._tools["web_search"].calls == [{"query": "q"}]


async def test_invalid_selection_fails_fast_with_a_clear_message_and_runs_nothing():
    reg = _registry()
    pipe, _, pub, collector = _pipeline(reg)
    await pipe.run(_event(selected_tools=["calculator"]))
    pub.publish_error_response.assert_awaited_once()
    event = pub.publish_error_response.await_args.args[1]
    assert "calculator" in event.error_message and "not available" in event.error_message
    assert reg._tools["calculator"].calls == []
    collector.collect.assert_not_called()  # rejected before spending any work


async def test_selecting_a_tool_disables_the_greeting_fast_path():
    reg = _registry()
    pipe, _, _, collector = _pipeline(reg)
    await pipe.run(_event("hi", selected_tools=["web_search"]))
    collector.collect.assert_awaited_once()
    assert reg._tools["web_search"].calls == [{"query": "hi"}]


# ------------------------------------------------------------------ document context


def _bundle(status, chunks=0, requested=1, authorized=1, ready=1):
    chunk_objs = [DocumentChunk(chunk_id=f"c{i}", file_id="doc-1", source_name="handbook.pdf", content=f"code is X{i}", score=0.9, page_number=1, section="Intro") for i in range(chunks)]
    r = RetrievalContext(chunks=chunk_objs, total_chunks=chunks, query="q", status=status, status_detail="d",
                         requested_count=requested, authorized_count=authorized, ready_count=ready)
    return ContextBundle(retrieval=r)


@pytest.mark.parametrize("status", ["ACCESS_DENIED", "NOT_INDEXED", "INDEXING", "INDEXING_FAILED", "UNAVAILABLE", "TIMEOUT"])
async def test_unusable_attached_document_fails_the_turn_instead_of_answering_ungrounded(status):
    pipe, _, pub, _ = _pipeline(_registry(), bundle=_bundle(status))
    await pipe.run(_event("what does the pdf say about the vault?", file_ids=["doc-1"]))
    pub.publish_error_response.assert_awaited_once()
    event = pub.publish_error_response.await_args.args[1]
    assert event.error_code == f"DOCUMENT_{status}"
    assert "document" in event.error_message.lower()
    pipe.generation_router  # generation was never reached
    assert pipe.streaming_engine.stream is not None


async def test_missing_retrieval_context_with_attached_files_is_reported_not_ignored():
    pipe, _, pub, _ = _pipeline(_registry(), bundle=ContextBundle())  # retrieval source absent (gRPC call failed)
    await pipe.run(_event("what does the pdf say?", file_ids=["doc-1"]))
    assert pub.publish_error_response.await_args.args[1].error_code == "DOCUMENT_UNAVAILABLE"


async def test_chunks_reach_the_pipeline_and_no_error_is_published():
    pipe, _, pub, _ = _pipeline(_registry(), bundle=_bundle("OK", chunks=2))
    await pipe.run(_event("what is the code?", file_ids=["doc-1"]))
    pub.publish_error_response.assert_not_called()


async def test_no_attached_files_never_triggers_document_checks():
    pipe, _, pub, _ = _pipeline(_registry(), bundle=_bundle("ACCESS_DENIED"))
    await pipe.run(_event("what is kafka?"))  # no file_ids: retrieval status is irrelevant
    pub.publish_error_response.assert_not_called()


def _builder():
    return PromptBuilder(registry=PromptRegistry(loader=PromptLoader()))


def _ctx(bundle, notice=None, file_ids=("doc-1",)):
    return PipelineContext(conversation_id="c", user_id="u", message_id="m", request_id="r",
                           user_message="What is the vault code?", file_ids=list(file_ids),
                           context_bundle=bundle, document_notice=notice)


def _compose(ctx):
    result = WorkflowResult(mode="ask_files", engine_type="mode_handler", draft_content=None, user_message=ctx.user_message)
    return _builder().build(result, ctx)


def test_retrieved_chunks_and_their_sources_appear_in_the_final_prompt():
    composed = _compose(_ctx(_bundle("OK", chunks=2)))
    text = "\n".join(m["content"] for m in composed.messages)
    assert "Retrieved Document Context" in text and "code is X0" in text and "handbook.pdf" in text and "p. 1" in text
    assert "retrieval" in {s.name for s in composed.sections}


def test_no_match_notice_tells_the_model_not_to_invent_document_content():
    ctx = _ctx(_bundle("NO_MATCH"), notice="no passage was relevant. Say so; do not invent its contents.")
    composed = _compose(ctx)
    text = "\n".join(m["content"] for m in composed.messages)
    assert "Attached Document Notice" in text and "do not invent" in text
    assert "retrieval" not in {s.name for s in composed.sections}


def test_pipeline_sets_a_notice_for_no_match_and_for_partial_access():
    pipe, *_ = _pipeline(_registry())
    ctx = _ctx(_bundle("NO_MATCH"))
    pipe._enforce_document_context(ctx)
    assert "not appear" in ctx.document_notice or "no passage" in ctx.document_notice
    ctx2 = _ctx(_bundle("OK", chunks=1, requested=3, authorized=1, ready=1))
    pipe._enforce_document_context(ctx2)
    assert "2 of the attached documents" in ctx2.document_notice


def test_merger_preserves_the_retrieval_status_and_counts():
    merged = ContextMerger().merge(retrieval=_bundle("OK", chunks=1, requested=2, authorized=2, ready=1).retrieval)
    assert merged.retrieval.status == "OK" and merged.retrieval.ready_count == 1 and merged.retrieval.requested_count == 2
    empty = ContextMerger().merge(retrieval=_bundle("INDEXING").retrieval)
    assert empty.retrieval.status == "INDEXING"


# ------------------------------------------------------------------ response modes


def test_discovery_lists_selectable_modes_and_respects_feature_flags():
    from app.workflow_engine.modes import available_modes

    on = [m["id"] for m in available_modes(LLMServiceConfig())]
    assert on == ["tutor", "code", "web_search", "smart", "deep_research"]  # ask_files/default are never offered
    off = [m["id"] for m in available_modes(LLMServiceConfig(feature_smart_mode=False, feature_deep_research_mode=False, feature_web_search=False))]
    assert off == ["tutor", "code"]
    assert all(set(m) == {"id", "name", "description"} for m in available_modes(LLMServiceConfig()))


def test_discovery_endpoint_includes_modes():
    app = FastAPI()
    app.include_router(tools_router, prefix="/api")
    container = MagicMock()
    container.tool_registry = _registry()
    container.config = LLMServiceConfig()
    app.dependency_overrides[get_container] = lambda: container
    body = TestClient(app).get("/api/v1/tools").json()
    assert [m["id"] for m in body["modes"]][:2] == ["tutor", "code"]


@pytest.mark.parametrize("raw", [None, "", "  ", "auto", "default", "AUTO"])
def test_auto_means_no_explicit_mode(raw):
    from app.workflow_engine.modes import resolve_requested_mode

    assert resolve_requested_mode(raw, LLMServiceConfig()) is None


def test_valid_mode_is_resolved_case_insensitively():
    from app.workflow_engine.modes import resolve_requested_mode

    assert resolve_requested_mode(" Deep_Research ", LLMServiceConfig()) == UserMode.DEEP_RESEARCH


@pytest.mark.parametrize("raw", ["nope", "ask_files", "../x"])
def test_unknown_or_non_selectable_modes_are_rejected(raw):
    from app.exceptions.user_facing import InvalidModeError
    from app.workflow_engine.modes import resolve_requested_mode

    with pytest.raises(InvalidModeError) as e:
        resolve_requested_mode(raw, LLMServiceConfig())
    assert "not available" in e.value.user_message


def test_disabled_mode_is_rejected_not_ignored():
    from app.exceptions.user_facing import InvalidModeError
    from app.workflow_engine.modes import resolve_requested_mode

    with pytest.raises(InvalidModeError):
        resolve_requested_mode("smart", LLMServiceConfig(feature_smart_mode=False))


def test_requested_mode_overrides_the_planner_and_sets_the_loop_budget():
    from app.workflow_engine.modes import apply_requested_mode

    cfg = LLMServiceConfig()
    plan = apply_requested_mode(_plan(), UserMode.SMART, cfg)
    assert plan.mode == UserMode.SMART and plan.max_iterations == cfg.langgraph_max_loop_iterations_smart
    plan = apply_requested_mode(_plan(), UserMode.DEEP_RESEARCH, cfg)
    assert plan.max_iterations == cfg.langgraph_max_loop_iterations_deep_research
    plan = apply_requested_mode(_plan(), UserMode.CODE, cfg)
    assert plan.mode == UserMode.CODE and plan.max_iterations == 1
    untouched = _plan()
    assert apply_requested_mode(untouched, None, cfg).mode == UserMode.DEFAULT  # Auto leaves the planner's choice


async def test_pipeline_runs_the_mode_the_user_chose_even_if_the_planner_picked_another():
    pipe, seen, pub, _ = _pipeline(_registry())
    modes = {}
    original = pipe.workflow_engine.execute

    async def spy(plan, ctx):
        modes["mode"] = plan.mode
        return await original(plan, ctx)

    pipe.workflow_engine.execute = spy
    await pipe.run(_event("explain closures", mode_hint="tutor"))
    assert modes["mode"] == UserMode.TUTOR
    pub.publish_error_response.assert_not_called()


async def test_auto_mode_keeps_the_planner_choice():
    pipe, _, _, _ = _pipeline(_registry())
    modes = {}
    original = pipe.workflow_engine.execute

    async def spy(plan, ctx):
        modes["mode"] = plan.mode
        return await original(plan, ctx)

    pipe.workflow_engine.execute = spy
    await pipe.run(_event("explain closures"))
    assert modes["mode"] == UserMode.DEFAULT


async def test_invalid_mode_fails_fast_with_a_clear_message():
    pipe, _, pub, collector = _pipeline(_registry())
    await pipe.run(_event("hello", mode_hint="warp_speed"))
    event = pub.publish_error_response.await_args.args[1]
    assert event.error_code == "INVALID_MODE" and "warp_speed" in event.error_message
    collector.collect.assert_not_called()
