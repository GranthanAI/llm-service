"""
Shared attempt/repair policy for visualization blocks (Mermaid in the browser, Manim in the worker).

Default: 3 TOTAL attempts per block -> attempt 1 is the original, attempts 2 and 3 are targeted repairs. After the third
failed attempt processing stops and ONLY that block falls back to text. `AttemptTracker` is the single place that counts,
so "three total, not three retries plus the original" cannot drift between the two renderers.
"""

from dataclasses import dataclass, field
from enum import StrEnum


class FailureCategory(StrEnum):
    INVALID_SPEC = "invalid_spec"
    UNSUPPORTED_TEMPLATE = "unsupported_template"
    SYNTAX = "syntax"                          # Mermaid syntax / validation failure
    RENDER_ERROR = "render_error"
    INVALID_OUTPUT = "invalid_output"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    STORAGE_FAILURE = "storage_failure"
    WORKER_UNAVAILABLE = "worker_unavailable"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    DISABLED = "disabled"
    UNAUTHORIZED = "unauthorized"
    BUDGET_EXHAUSTED = "budget_exhausted"


# A corrected specification/source can fix these.
REPAIRABLE = frozenset({
    FailureCategory.INVALID_SPEC, FailureCategory.UNSUPPORTED_TEMPLATE, FailureCategory.SYNTAX,
    FailureCategory.RENDER_ERROR, FailureCategory.INVALID_OUTPUT,
})
# The same input may succeed again later (infrastructure hiccup): retry without calling the model.
TRANSIENT = frozenset({FailureCategory.STORAGE_FAILURE, FailureCategory.WORKER_UNAVAILABLE, FailureCategory.TIMEOUT})
# Never retried: retrying cannot change the outcome or would bypass a limit.
PERMANENT = frozenset({
    FailureCategory.DISABLED, FailureCategory.UNAUTHORIZED, FailureCategory.BUDGET_EXHAUSTED,
    FailureCategory.CANCELLED, FailureCategory.EXPIRED, FailureCategory.RESOURCE_LIMIT,
})

USER_MESSAGES = {
    FailureCategory.INVALID_SPEC: "The animation description was not valid.",
    FailureCategory.UNSUPPORTED_TEMPLATE: "This kind of animation is not supported.",
    FailureCategory.SYNTAX: "The diagram could not be drawn.",
    FailureCategory.RENDER_ERROR: "The visual could not be generated.",
    FailureCategory.INVALID_OUTPUT: "The generated video was not usable.",
    FailureCategory.TIMEOUT: "Generating the visual took too long.",
    FailureCategory.RESOURCE_LIMIT: "The visual was too heavy to generate.",
    FailureCategory.STORAGE_FAILURE: "The generated visual could not be stored.",
    FailureCategory.WORKER_UNAVAILABLE: "Animations are temporarily unavailable.",
    FailureCategory.CANCELLED: "The visual was cancelled.",
    FailureCategory.EXPIRED: "The visual expired.",
    FailureCategory.DISABLED: "This kind of visual is turned off.",
    FailureCategory.UNAUTHORIZED: "You do not have access to this visual.",
    FailureCategory.BUDGET_EXHAUSTED: "The visual limit was reached.",
}


def user_message(category: FailureCategory | str) -> str:
    """Bounded, content-free text safe to show a user (never a stack trace or model/source text)."""
    try:
        return USER_MESSAGES[FailureCategory(category)]
    except (ValueError, KeyError):
        return "The visual could not be generated."


class AttemptsExhausted(Exception):
    pass


@dataclass
class AttemptTracker:
    max_attempts: int = 3
    used: int = 0
    history: list[str] = field(default_factory=list)

    def begin_attempt(self) -> int:
        """Start the next attempt and return its 1-based number. Raises once the budget is spent."""
        if self.used >= self.max_attempts:
            raise AttemptsExhausted(f"{self.max_attempts} total attempts already used")
        self.used += 1
        return self.used

    def record_failure(self, category: FailureCategory) -> None:
        self.history.append(category.value)

    @property
    def remaining(self) -> int:
        return self.max_attempts - self.used

    def next_step(self, category: FailureCategory) -> str:
        """After a failed attempt: 'repair' (ask the model for a corrected block), 'retry' (same input, transient
        failure) or 'give_up' (budget spent or a permanent failure)."""
        if self.remaining <= 0 or category in PERMANENT:
            return "give_up"
        if category in REPAIRABLE:
            return "repair"
        if category in TRANSIENT:
            return "retry"
        return "give_up"


def text_fallback(title: str | None, category: FailureCategory | str, kind: str) -> str:
    """Markdown that replaces ONLY a failed block, so the surrounding explanation stays complete."""
    label = {"mermaid": "diagram", "manim": "animation"}.get(kind, "visual")
    name = f" “{title}”" if title else ""
    return f"*({label.capitalize()}{name} unavailable: {user_message(category)} The explanation above stands on its own.)*"
