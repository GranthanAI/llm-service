"""
Unit tests for RequestPipeline — the orchestrator wiring Context Collection ->
Request Analysis -> Workflow Dispatch -> Prompt Build -> Context Window
Management -> Generation Router -> Streaming Engine.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.context.schemas import ContextBundle
from app.models.execution_plan import ExecutionPlan
from app.models.request import ChatMessageCreatedEvent, TraceContext
from app.pipeline.request_pipeline import RequestPipeline
from app.prompts.schemas import ComposedPrompt
from app.workflow_engine.workflow_result import WorkflowResult


def make_event(content: str = "Explain Kafka") -> ChatMessageCreatedEvent:
    return ChatMessageCreatedEvent(
        message_id="msg_1",
        conversation_id="conv_1",
        user_id="user_1",
        content=content,
        mode_hint="default",
        selected_tools=[],
        file_ids=[],
        trace_context=TraceContext(traceparent="00-abc-def-01"),
    )


def make_pipeline() -> tuple[RequestPipeline, dict]:
    context_collector = AsyncMock()
    context_collector.collect.return_value = ContextBundle()

    request_analyzer = AsyncMock()
    plan = ExecutionPlan()
    request_analyzer.analyze.return_value = plan

    workflow_engine = AsyncMock()
    workflow_result = WorkflowResult(
        mode="default", engine_type="mode_handler", draft_content=None, user_message="Explain Kafka"
    )
    workflow_engine.execute.return_value = workflow_result

    prompt_builder = MagicMock()
    composed = ComposedPrompt(
        mode="default", engine_type="mode_handler", messages=[{"role": "user", "content": "hi"}]
    )
    prompt_builder.build.return_value = composed

    context_window_manager = MagicMock()
    trimmed = MagicMock()
    context_window_manager.manage.return_value = trimmed

    async def fake_stream_gen(*_args, **_kwargs):
        for tok in ["Hello", " world"]:
            yield tok

    generation_router = MagicMock()
    generation_router.generate_stream = MagicMock(side_effect=lambda *a, **k: fake_stream_gen())

    streaming_engine = AsyncMock()
    streaming_engine.stream.return_value = "Hello world"

    publisher = AsyncMock()

    pipeline = RequestPipeline(
        context_collector=context_collector,
        request_analyzer=request_analyzer,
        workflow_engine=workflow_engine,
        prompt_builder=prompt_builder,
        context_window_manager=context_window_manager,
        generation_router=generation_router,
        streaming_engine=streaming_engine,
        publisher=publisher,
    )

    return pipeline, {
        "context_collector": context_collector,
        "request_analyzer": request_analyzer,
        "workflow_engine": workflow_engine,
        "prompt_builder": prompt_builder,
        "context_window_manager": context_window_manager,
        "generation_router": generation_router,
        "streaming_engine": streaming_engine,
        "publisher": publisher,
        "plan": plan,
        "workflow_result": workflow_result,
    }


@pytest.mark.asyncio
async def test_run_calls_every_stage_in_order():
    pipeline, mocks = make_pipeline()
    event = make_event()

    await pipeline.run(event)

    mocks["context_collector"].collect.assert_awaited_once()
    mocks["request_analyzer"].analyze.assert_awaited_once()
    mocks["workflow_engine"].execute.assert_awaited_once_with(mocks["plan"], mocks["workflow_engine"].execute.call_args.args[1])
    mocks["prompt_builder"].build.assert_called_once()
    mocks["context_window_manager"].manage.assert_called_once()
    mocks["streaming_engine"].stream.assert_awaited_once()
    mocks["publisher"].publish_error_response.assert_not_called()


@pytest.mark.asyncio
async def test_run_populates_context_before_streaming():
    pipeline, mocks = make_pipeline()
    event = make_event()

    await pipeline.run(event)

    ctx_passed_to_stream = mocks["streaming_engine"].stream.call_args.args[1]
    assert ctx_passed_to_stream.plan is mocks["plan"]
    assert ctx_passed_to_stream.context_bundle is not None
    assert mocks["workflow_result"].engine_type == "mode_handler"


@pytest.mark.asyncio
async def test_run_publishes_error_event_on_failure_instead_of_raising():
    pipeline, mocks = make_pipeline()
    mocks["workflow_engine"].execute.side_effect = RuntimeError("boom")
    event = make_event()

    # Should not raise
    await pipeline.run(event)

    mocks["publisher"].publish_error_response.assert_awaited_once()
    published_event = mocks["publisher"].publish_error_response.call_args.args[1]
    assert published_event.status == "error"
