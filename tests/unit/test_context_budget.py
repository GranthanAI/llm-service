"""Context budget correctness: unknown models are handled conservatively (and loudly), a configured cap can only lower the
budget, and the requested output is reserved."""

from unittest.mock import MagicMock

import pytest

from app.context_window.manager import ContextWindowManager
from app.context_window.models import ModelLimits
from app.prompts.schemas import ComposedPrompt, PromptSection

WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "


def prompt(words_per_section=200):
    text = WORDS * words_per_section
    secs = [PromptSection(name="system", content="You are helpful.", priority=10, token_count=5, trimmable=False),
            PromptSection(name="graph_context", content=text, priority=5, token_count=len(text.split()), trimmable=True),
            PromptSection(name="user_query", content="question?", priority=10, token_count=3, trimmable=False)]
    return ComposedPrompt(sections=secs, total_tokens=sum(s.token_count for s in secs), messages=[], mode="default", engine_type="mode_handler")


def manager(**kw):
    m = ContextWindowManager(logger_instance=MagicMock(), **kw)
    return m


def test_an_unknown_model_gets_conservative_limits_and_a_single_warning():
    m = manager()
    limits = m.get_model_limits("brand-new-model-x")
    assert limits.context_window == 8192 and limits.max_output_tokens == 1024     # never a silent 131k window
    m.get_model_limits("brand-new-model-x")
    assert m.logger.warning.call_count == 1 and m.logger.warning.call_args.args[0] == "unknown_model_limits"
    m.get_model_limits("another-one")
    assert m.logger.warning.call_count == 2


def test_registered_models_keep_their_real_limits_without_a_warning():
    m = manager()
    assert m.get_model_limits("llama-3.3-70b-versatile").context_window == 131072
    m.register_model_limits("custom", ModelLimits(model="custom", context_window=32000, max_output_tokens=2048, provider="groq"))
    assert m.get_model_limits("custom").context_window == 32000
    m.logger.warning.assert_not_called()


def test_a_configured_cap_cannot_exceed_the_models_window():
    m = manager()
    m.register_model_limits("tiny", ModelLimits(model="tiny", context_window=4000, max_output_tokens=500, provider="groq"))
    big = prompt(5000)                                                         # far more than fits
    trimmed = m.manage(big, target_model="tiny", custom_input_budget=100_000)    # an absurd cap
    window_budget = int(4000 * 0.95) - 500
    assert trimmed.total_tokens <= window_budget


def test_a_lower_configured_cap_still_applies():
    m = manager()
    trimmed = m.manage(prompt(5000), target_model="llama-3.3-70b-versatile", custom_input_budget=1500)
    assert trimmed.total_tokens <= 1500


def test_the_requested_output_is_reserved():
    m = manager()
    m.register_model_limits("tiny", ModelLimits(model="tiny", context_window=4000, max_output_tokens=500, provider="groq"))
    default = m.manage(prompt(5000), target_model="tiny")
    reserved = m.manage(prompt(5000), target_model="tiny", max_output_tokens=2000)
    assert reserved.total_tokens < default.total_tokens
    assert reserved.total_tokens <= int(4000 * 0.95) - 2000


def test_mandatory_sections_that_cannot_fit_fail_clearly_instead_of_overflowing():
    from app.exceptions.analysis import ContextOverflowError

    m = manager()
    m.register_model_limits("tiny", ModelLimits(model="tiny", context_window=1000, max_output_tokens=900, provider="groq"))
    locked = PromptSection(name="system", content=WORDS * 300, priority=10, token_count=3600, trimmable=False)
    with pytest.raises(ContextOverflowError):
        m.manage(ComposedPrompt(sections=[locked], total_tokens=3600, messages=[], mode="default", engine_type="mode_handler"), target_model="tiny")
