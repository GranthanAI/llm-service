"""Shared pipeline test doubles for the context-fitting step (counter + limits + a manager that returns a fixed prompt)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.context_window.token_counter import TokenCounter


def built_prompt(**over):
    """What PromptBuilder.build returns, with real serialized messages (the fit step counts them)."""
    base = dict(sections=[], messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}], mode="default")
    base.update(over)
    return SimpleNamespace(**base)


def trimmed_prompt(**over):
    base = dict(sections=[], messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
                total_tokens=10, trimmed_sections=[], mode="default", original_tokens=10, was_trimmed=False, target_model="m")
    base.update(over)
    return SimpleNamespace(**base)


def manager_double(trimmed=None):
    """A ContextWindowManager stand-in: real token counter, unknown-model limits, fixed trimmed result."""
    cwm = MagicMock()
    cwm.token_counter = TokenCounter()
    cwm.resolve_limits.return_value = (None, True)
    cwm.manage.return_value = trimmed if trimmed is not None else trimmed_prompt()
    return cwm
