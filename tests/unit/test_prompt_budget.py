"""Prompt token budget: requests are trimmed to fit the free-tier Groq limits instead of being rejected.

Pure local logic and fakes: no model or network calls.
"""

from unittest.mock import AsyncMock, MagicMock
from tests.doubles import built_prompt, manager_double

from app.config.settings import LLMServiceConfig
from app.context.schemas import ContextBundle, DocumentChunk, RetrievalContext
from app.context_window.manager import ContextWindowManager
from app.models.execution_plan import ExecutionPlan, IntentCategory, ReasoningMode, Skill, UserMode
from app.models.pipeline_context import PipelineContext
from app.models.request import ChatMessageCreatedEvent
from app.pipeline.request_pipeline import RequestPipeline
from app.prompts.builder import PromptBuilder
from app.prompts.loader import PromptLoader
from app.prompts.registry import PromptRegistry
from app.prompts.schemas import ComposedPrompt, PromptSection
from app.workflow_engine.workflow_result import WorkflowResult

WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "


def _section(name, priority, tokens_hint, trimmable=True):
    text = WORDS * tokens_hint
    return PromptSection(name=name, content=text, priority=priority, token_count=len(text.split()), trimmable=trimmable)


def _big_prompt():
    secs = [
        _section("system", 10, 10, trimmable=False),
        _section("long_term_memory", 7, 150),
        _section("graph_context", 5, 150),
        _section("tool_results", 6, 300),
        _section("conversation_history", 8, 200),
        _section("user_query", 10, 2, trimmable=False),
    ]
    return ComposedPrompt(sections=secs, total_tokens=sum(s.token_count for s in secs), messages=[], mode="default", engine_type="mode_handler")


def test_a_large_prompt_is_trimmed_to_the_configured_budget_and_the_question_survives():
    mgr = ContextWindowManager()
    prompt = _big_prompt()
    assert prompt.total_tokens > 5000  # the situation from the log: far over what the free tier accepts
    trimmed = mgr.manage(prompt, target_model="nvidia/some-unknown-model", custom_input_budget=2000)
    assert trimmed.total_tokens <= 2000
    names = {s.name for s in trimmed.sections if s.content}
    assert {"system", "user_query"} <= names  # locked sections are never cut
    assert "graph_context" not in names or next(s for s in trimmed.sections if s.name == "graph_context").token_count < 1800


def test_without_a_budget_an_unknown_model_gets_the_huge_default_window_and_nothing_is_trimmed():
    """This is the original bug: the NVIDIA model id is not in the limits table."""
    trimmed = ContextWindowManager().manage(_big_prompt(), target_model="nvidia/nemotron-3.5-lightning-30b-a3b")
    assert trimmed.total_tokens > 5000


def test_lowest_priority_context_is_cut_before_history():
    trimmed = ContextWindowManager().manage(_big_prompt(), target_model="x", custom_input_budget=3000)
    by = {s.name: s.token_count for s in trimmed.sections}  # a fully cut section disappears from the list
    original = {s.name: s.token_count for s in _big_prompt().sections}
    assert by.get("graph_context", 0) < original["graph_context"]  # priority 5: cut first
    assert by["conversation_history"] > by.get("graph_context", 0)  # priority 8: kept in preference


def test_default_budget_fits_groqs_free_tier_with_room_for_an_answer():
    c = LLMServiceConfig(_env_file=None)
    # The legacy per-request cap is OFF by default: the 150k context budget applies. A rate-limited tier opts in
    # explicitly (PROMPT_TOKEN_BUDGET=5000 keeps a request within Groq's 8000 tokens/min with room for the answer).
    assert c.prompt_token_budget == 0


# ------------------------------------------------------------------ pipeline passes the budget


def _pipeline(budget):
    plan = ExecutionPlan(intent=IntentCategory.GENERAL_CHAT, mode=UserMode.DEFAULT, skill=Skill.GENERAL_CHAT, reasoning=ReasoningMode.DIRECT, tools=[], max_iterations=1)
    collector = MagicMock()
    collector.collect = AsyncMock(return_value=ContextBundle())
    analyzer = MagicMock()
    analyzer.analyze = AsyncMock(return_value=plan)
    workflow = MagicMock()
    workflow.execute = AsyncMock(return_value=MagicMock(tool_outputs=[]))
    builder = MagicMock()
    builder.build = MagicMock(return_value=built_prompt())
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
    pipe = RequestPipeline(
        context_collector=collector, request_analyzer=analyzer, workflow_engine=workflow, prompt_builder=builder,
        context_window_manager=cwm, generation_router=router, streaming_engine=streaming, publisher=MagicMock(),
        config=LLMServiceConfig(_env_file=None, prompt_token_budget=budget), logger=MagicMock(),
    )
    return pipe, cwm


def _event(text="explain closures in detail"):
    return ChatMessageCreatedEvent(conversation_id="c", user_id="u", message_id="m", response_message_id="r", content=text)


async def test_pipeline_applies_the_prompt_budget():
    pipe, cwm = _pipeline(4200)
    await pipe.run(_event())
    assert cwm.manage.call_args.kwargs["custom_input_budget"] == 4200      # the legacy per-request cap applies as a minimum


async def test_a_zero_legacy_cap_leaves_only_the_context_budget_and_the_model_window():
    pipe, cwm = _pipeline(0)
    await pipe.run(_event())
    effective = cwm.manage.call_args.kwargs["custom_input_budget"]
    assert 4200 < effective <= 32768        # unknown model: the conservative window, never an unbounded request


# ------------------------------------------------------------------ attached documents are protected


def _compose(file_ids):
    chunk = DocumentChunk(chunk_id="c1", file_id="doc-1", source_name="handbook.pdf", content="The vault code is X", score=0.9, page_number=1, section="Intro")
    bundle = ContextBundle(retrieval=RetrievalContext(chunks=[chunk], total_chunks=1, status="OK"))
    ctx = PipelineContext(conversation_id="c", user_id="u", message_id="m", request_id="r", user_message="what is the code?", file_ids=file_ids, context_bundle=bundle)
    result = WorkflowResult(mode="ask_files", engine_type="mode_handler", draft_content=None, user_message=ctx.user_message)
    return PromptBuilder(registry=PromptRegistry(loader=PromptLoader())).build(result, ctx)


def test_document_passages_outrank_graph_and_memory_when_files_are_attached():
    with_files = {s.name: s.priority for s in _compose(["doc-1"]).sections}
    without = {s.name: s.priority for s in _compose([]).sections}
    assert with_files["retrieval"] == 9 and without["retrieval"] == 5
