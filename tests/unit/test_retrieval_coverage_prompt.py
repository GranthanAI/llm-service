"""WEAK/NONE retrieval coverage is rendered in the document section; UNSPECIFIED/STRONG leave the text unchanged."""
import pytest

from app.config.settings import get_settings
from app.context.schemas import ContextBundle, DocumentChunk, RetrievalContext
from app.models.pipeline_context import PipelineContext
from app.prompts.builder import PromptBuilder
from app.prompts.loader import PromptLoader
from app.prompts.registry import PromptRegistry
from app.workflow_engine.workflow_result import WorkflowResult


def section(coverage, chunks=True):
    chunk_list = [DocumentChunk(chunk_id="c1", file_id="f", source_name="PHASES.md", content="Phase 1 text")] if chunks else []
    ctx = PipelineContext(conversation_id="C", user_id="u", message_id="m", request_id="r", user_message="what is phase 6?",
                          file_ids=["f"], context_bundle=ContextBundle(retrieval=RetrievalContext(chunks=chunk_list, coverage=coverage)))
    result = WorkflowResult(mode="default", engine_type="mode_handler", draft_content=None, user_message=ctx.user_message)
    composed = PromptBuilder(registry=PromptRegistry(loader=PromptLoader())).build(result, ctx)
    return next((s.content for s in composed.sections if s.name == "retrieval"), None)


def test_unspecified_and_strong_are_unchanged():
    base = section("COVERAGE_UNSPECIFIED")
    assert section("STRONG") == base and "coverage" not in base.lower()


@pytest.mark.parametrize("cov", ["WEAK", "NONE"])
def test_weak_and_none_tell_the_model_not_to_guess_and_that_passages_override_pinned_facts(cov):
    text = section(cov)
    assert f"Retrieval coverage: {cov}" in text
    assert "do not cover" in text and "Pinned Facts" in text and "override" in text


def test_none_without_chunks_still_renders_the_section():
    assert "Retrieval coverage: NONE" in section("NONE", chunks=False)
    assert section("STRONG", chunks=False) is None


def test_flag_off_restores_old_text(monkeypatch):
    monkeypatch.setattr(get_settings(), "retrieval_coverage_prompt_enabled", False)
    assert section("WEAK") == section("COVERAGE_UNSPECIFIED")
