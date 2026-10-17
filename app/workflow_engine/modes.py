"""
User-selectable response modes: discovery, validation and enforcement helpers.

A mode decides HOW a question is handled (plain chat, tutoring, code help, forced web search, an iterative
"smart" loop, deep research). "Auto" (no mode) lets the request planner choose, which is the default.
`ask_files` is never selectable: it is chosen automatically when documents are attached.
"""

from typing import Any

from app.config.settings import LLMServiceConfig
from app.exceptions.user_facing import InvalidModeError
from app.models.execution_plan import UserMode

# Order = order shown to the user. Each entry: (mode, display name, description).
_SELECTABLE: list[tuple[UserMode, str, str]] = [
    (UserMode.TUTOR, "Tutor", "Explains step by step and checks understanding."),
    (UserMode.CODE, "Code", "Focused on writing, explaining and debugging code."),
    (UserMode.WEB_SEARCH, "Web search", "Always searches the web first, then answers with sources."),
    (UserMode.SMART, "Smart", "Reasons in several passes and uses tools when they help."),
    (UserMode.DEEP_RESEARCH, "Deep research", "Plans, searches and synthesises a thorough, sourced answer. Slower."),
]

# Words clients may send to mean "let the planner decide".
_AUTO = {"", "auto", "default"}


def _enabled(mode: UserMode, config: LLMServiceConfig) -> bool:
    if mode == UserMode.SMART:
        return config.enable_smart_mode
    if mode == UserMode.DEEP_RESEARCH:
        return config.enable_deep_research_mode
    if mode == UserMode.WEB_SEARCH:
        return config.enable_web_search
    return True


def available_modes(config: LLMServiceConfig) -> list[dict[str, Any]]:
    """Modes a user may pick right now (feature flags respected). Stable ids; safe to show in a UI."""
    return [
        {"id": m.value, "name": name, "description": desc}
        for m, name, desc in _SELECTABLE
        if _enabled(m, config)
    ]


def resolve_requested_mode(raw: str | None, config: LLMServiceConfig) -> UserMode | None:
    """Validate the mode the user asked for. None = Auto. Unknown or disabled modes are rejected with a
    clear message instead of being silently ignored."""
    value = (raw or "").strip().lower()
    if value in _AUTO:
        return None
    for mode, _name, _desc in _SELECTABLE:
        if mode.value == value:
            if not _enabled(mode, config):
                raise InvalidModeError(value, "this mode is turned off")
            return mode
    raise InvalidModeError(value, "unknown mode")


def apply_requested_mode(plan: Any, mode: UserMode | None, config: LLMServiceConfig) -> Any:
    """The user's explicit choice wins over the planner's. Iterative modes get their configured loop
    budget; single-pass modes stay single-pass."""
    if mode is None:
        return plan
    plan.mode = mode
    if mode == UserMode.SMART:
        plan.max_iterations = config.langgraph_max_loop_iterations_smart
    elif mode == UserMode.DEEP_RESEARCH:
        plan.max_iterations = config.langgraph_max_loop_iterations_deep_research
    else:
        plan.max_iterations = 1
    return plan
