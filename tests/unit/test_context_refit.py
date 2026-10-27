"""A prompt that fits by section count but measures a few tokens over is re-trimmed until it fits (it used to fail by 1 token)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.config.settings import LLMServiceConfig
from app.context_window.fit import fit_prompt
from app.exceptions.analysis import ContextOverflowError

BUDGET = 5752


def _run(measure):
    """`measure(target)` = what the real serialized prompt measures when sections were trimmed to `target`."""
    manager = MagicMock()
    manager.resolve_limits.return_value = (None, True)
    manager.token_counter.count_messages.side_effect = lambda msgs, provider: msgs[0]["tokens"]
    manager.manage.side_effect = lambda composed, target_model, custom_input_budget, max_output_tokens: SimpleNamespace(
        messages=[{"tokens": measure(custom_input_budget)}], total_tokens=custom_input_budget, trimmed_sections=["graph_context"])
    budget = SimpleNamespace(input_budget=BUDGET, app_budget_source="provider", app_budget=0, model_window=0, window_assumed=True,
                             output_reserve=0, legacy_cap_applied=False, tpm_cap_applied=True)
    with patch("app.context_window.fit.resolve_context_budget", return_value=budget):
        trimmed, report = fit_prompt(
            config=LLMServiceConfig(context_compaction_enabled=False), manager=manager, prompt_builder=MagicMock(),
            result=SimpleNamespace(conversation_history=[]), ctx=SimpleNamespace(), composed=SimpleNamespace(messages=[{"tokens": 6135}]),
            provider="groq", model="m", max_output_tokens=None, logger=MagicMock())
    return trimmed, report, manager


def test_one_token_overshoot_after_the_first_retry_now_fits():
    # framing adds 11 tokens, and rounding makes the first correction fall exactly one token short (the logged 5753 vs 5752)
    measure = lambda b: b + 11 + -(-(BUDGET - b) // 11) if b <= BUDGET else b + 11
    trimmed, report, manager = _run(measure)
    assert report.tokens_after <= BUDGET
    assert manager.manage.call_count >= 3                      # first trim + several corrections, not just one


def test_already_fitting_prompt_is_trimmed_once():
    _, report, manager = _run(lambda b: b - 5)
    assert manager.manage.call_count == 1 and report.tokens_after <= BUDGET


def test_genuinely_oversized_locked_content_still_raises_a_controlled_error():
    with pytest.raises(ContextOverflowError):
        _run(lambda b: BUDGET + 500)                           # locked content: trimming can never get under the budget
