"""
Unit Tests for Context Window Manager (TokenCounter, BudgetCalculator, PriorityTrimmer, ContextWindowManager).
Implements test coverage for Phase 14 as per LLD v2.0 Section 18.
"""

import pytest

from app.context_window.budget import BudgetCalculator
from app.context_window.manager import ContextWindowManager
from app.context_window.models import ModelLimits
from app.context_window.token_counter import TokenCounter
from app.context_window.trimmer import PriorityTrimmer
from app.exceptions.analysis import ContextOverflowError
from app.prompts.schemas import ComposedPrompt, PromptSection

# --- 1. TokenCounter Tests ---


def test_token_counter_calibration_and_multipliers():
    """Verify TokenCounter accurately counts tokens with provider multipliers."""
    counter = TokenCounter()
    sample_text = "GraphGPT is an enterprise LLM orchestration service built on Python 3.12."

    count_nvidia = counter.count_text(sample_text, provider="nvidia")
    count_groq = counter.count_text(sample_text, provider="groq")
    count_gemini = counter.count_text(sample_text, provider="gemini")

    assert count_nvidia > 0
    assert count_groq == count_nvidia
    assert count_gemini <= count_nvidia

    # Message overhead counting
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    msg_tokens = counter.count_messages(messages)
    assert msg_tokens > counter.count_text("You are a helpful assistant.") + counter.count_text(
        "Hello!"
    )


def test_token_counter_truncation():
    """Verify TokenCounter truncates text to target tokens."""
    counter = TokenCounter()
    long_text = "Word " * 100
    original_tokens = counter.count_text(long_text)

    truncated = counter.truncate_text_to_tokens(long_text, target_tokens=20)
    truncated_tokens = counter.count_text(truncated)

    assert truncated_tokens <= 25
    assert len(truncated) < len(long_text)


# --- 2. BudgetCalculator Tests ---


def test_budget_calculator_proportions():
    """Verify BudgetCalculator computes effective input and section quotas."""
    model = ModelLimits(
        model="meta/llama-3.1-405b-instruct",
        context_window=100000,
        max_output_tokens=4000,
        provider="nvidia",
    )
    allocation = BudgetCalculator.calculate(model)

    assert allocation.effective_window == 95000  # 95% of 100k
    assert allocation.reserved_output == 4000
    assert allocation.input_budget == 91000  # 95000 - 4000
    assert allocation.section_budgets["retrieval"] == int(91000 * 0.15)
    assert allocation.section_budgets["tool_results"] == int(91000 * 0.15)


# --- 3. PriorityTrimmer Tests ---


def test_priority_trimmer_preserves_when_under_budget():
    """Verify PriorityTrimmer makes no modifications when total tokens <= budget."""
    trimmer = PriorityTrimmer()
    sections = [
        PromptSection(
            name="system", content="System prompt", priority=10, token_count=20, trimmable=False
        ),
        PromptSection(
            name="user_query", content="Query", priority=10, token_count=10, trimmable=False
        ),
        PromptSection(
            name="retrieval", content="Doc chunk", priority=5, token_count=50, trimmable=True
        ),
    ]
    trimmed, trimmed_names = trimmer.trim(sections, total_budget=100)

    assert len(trimmed) == 3
    assert len(trimmed_names) == 0
    assert sum(s.token_count for s in trimmed) == 80


def test_priority_trimmer_trims_lowest_priority_first():
    """Verify PriorityTrimmer trims priority 5 sections before priority 7/8/9 sections."""
    trimmer = PriorityTrimmer()
    sections = [
        PromptSection(
            name="system", content="System prompt", priority=10, token_count=20, trimmable=False
        ),
        PromptSection(
            name="graph_context", content="Graph " * 50, priority=5, token_count=100, trimmable=True
        ),
        PromptSection(
            name="long_term_memory",
            content="Fact " * 50,
            priority=7,
            token_count=100,
            trimmable=True,
        ),
        PromptSection(
            name="user_query", content="User query", priority=10, token_count=20, trimmable=False
        ),
    ]
    # Total = 240 tokens. Target budget = 180 tokens (excess = 60 tokens).
    # graph_context (priority 5) should be trimmed first down by 60 tokens, leaving long_term_memory untouched!
    trimmed, trimmed_names = trimmer.trim(sections, total_budget=180)

    assert "graph_context" in trimmed_names
    assert "long_term_memory" not in trimmed_names
    assert sum(s.token_count for s in trimmed) <= 185


def test_priority_trimmer_raises_overflow_when_locked_sections_exceed_budget():
    """Verify PriorityTrimmer raises ContextOverflowError if locked/minimum sections exceed budget."""
    trimmer = PriorityTrimmer()
    sections = [
        PromptSection(
            name="system", content="System " * 100, priority=10, token_count=200, trimmable=False
        ),
        PromptSection(
            name="user_query", content="Query " * 50, priority=10, token_count=100, trimmable=False
        ),
    ]
    with pytest.raises(ContextOverflowError):
        trimmer.trim(sections, total_budget=150)


# --- 4. ContextWindowManager Integration Tests ---


def test_context_window_manager_end_to_end_flow():
    """Verify ContextWindowManager executes manage() across target models and produces TrimmedPrompt."""
    manager = ContextWindowManager()

    prompt = ComposedPrompt(
        sections=[
            PromptSection(
                name="system",
                content="System instruction",
                priority=10,
                token_count=30,
                trimmable=False,
            ),
            PromptSection(
                name="retrieval",
                content="Document excerpt " * 40,
                priority=5,
                token_count=200,
                trimmable=True,
            ),
            PromptSection(
                name="conversation_history",
                content="Chat history " * 30,
                priority=8,
                token_count=150,
                trimmable=True,
            ),
            PromptSection(
                name="user_query",
                content="User request",
                priority=10,
                token_count=20,
                trimmable=False,
            ),
        ],
        total_tokens=400,
        mode="tutor",
        engine_type="mode_handler",
        system_prompt="System instruction",
        user_prompt="Context & query",
    )

    # 1. Normal execution (no trimming required)
    trimmed_normal = manager.manage(prompt, target_model="meta/llama-3.1-405b-instruct")
    assert not trimmed_normal.was_trimmed
    assert trimmed_normal.total_tokens == 400
    assert len(trimmed_normal.messages) == 2

    # 2. Constrained budget execution (triggers trimming on retrieval section)
    trimmed_constrained = manager.manage(
        prompt, target_model="meta/llama-3.1-405b-instruct", custom_input_budget=250
    )
    assert trimmed_constrained.was_trimmed
    assert "retrieval" in trimmed_constrained.trimmed_sections
    assert trimmed_constrained.total_tokens <= 260
    assert len(trimmed_constrained.messages) == 2


def test_context_window_manager_model_limits_registration():
    """Verify custom ModelLimits can be registered and retrieved."""
    manager = ContextWindowManager()
    custom_limits = ModelLimits(
        model="custom-claude-3-7-sonnet",
        context_window=200000,
        max_output_tokens=8192,
        provider="nvidia",
    )
    manager.register_model_limits("custom-claude-3-7-sonnet", custom_limits)

    retrieved = manager.get_model_limits("custom-claude-3-7-sonnet")
    assert retrieved.context_window == 200000
    assert retrieved.max_output_tokens == 8192
