"""B4e: billing's max_context_tokens is a ceiling only when positive. 0, -1 and None mean "no cap from billing"
(Pay as you Go reads the model's full window); -1 used to become the budget and raise ContextBudgetError."""

from app.config.settings import LLMServiceConfig
from app.context_window.budget_policy import resolve_context_budget
from app.context_window.models import ModelLimits

LIMITS = ModelLimits(model="gemini-x", context_window=128_000, max_output_tokens=8_000, provider="gemini")


def budget(input_cap):
    return resolve_context_budget(LLMServiceConfig(), "gemini", "gemini-x", LIMITS, input_cap=input_cap)


def test_no_cap_values_leave_the_budget_to_the_model_window_and_the_app_budget():
    uncapped = budget(None).input_budget
    assert uncapped > 0
    for no_cap in (0, -1, -5):
        assert budget(no_cap).input_budget == uncapped


def test_a_positive_cap_only_lowers_the_budget():
    uncapped = budget(None).input_budget
    assert budget(16_000).input_budget == 16_000
    assert budget(uncapped * 10).input_budget == uncapped  # a cap above the budget is not a floor
