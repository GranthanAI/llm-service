"""
Per-request usage collector: one CallUsage for EVERY model call of a request (billing integration plan 4.4.2).

The collector lives in a ContextVar for the duration of one request (set by the pipeline), so call sites that have no
access to the request (shared adapters, the analysis client, query rewriters) still report into the right request. The
purpose of a call is also a ContextVar, set around the call site:

    with usage_purpose("request_analysis"):
        await client.complete(...)

Token counts are what the PROVIDER reported. When a provider reports none, they are estimated with the context-window
token counter, `usage_estimated` is true and llm_usage_estimated_total{provider,model} is incremented. Characters are
never used as tokens.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel

from app.config.logging import get_logger

logger = get_logger("usage_collector")

PURPOSES = ("answer", "request_analysis", "query_rewrite", "workflow", "research", "manim_plan")

# billing's provider names (config/seed/model_prices.yaml): our "gemini" adapter is billing's "google"
_PROVIDER_ALIASES = {"gemini": "google"}

_collector: ContextVar["UsageCollector | None"] = ContextVar("usage_collector", default=None)
_purpose: ContextVar[str] = ContextVar("usage_purpose", default="answer")


class CallUsage(BaseModel):
    provider: str
    model: str
    purpose: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    usage_estimated: bool = False


def _estimate_tokens(provider: str, messages: list[dict[str, Any]] | None, text: str | None) -> tuple[int, int]:
    """(input, output) token estimates with the existing counter; (0, 0) if it is unavailable."""
    try:
        from app.context_window.token_counter import TokenCounter

        counter = _estimate_tokens.counter  # type: ignore[attr-defined]
        if counter is None:
            counter = _estimate_tokens.counter = TokenCounter()  # type: ignore[attr-defined]
        i = counter.count_messages(messages, provider) if messages else 0
        o = counter.count_text(text, provider) if text else 0
        return int(i), int(o)
    except Exception:  # an estimate must never break a request
        return 0, 0


_estimate_tokens.counter = None  # type: ignore[attr-defined]


class UsageCollector:
    def __init__(self, model_resolver: Callable[[str, str], str] | None = None):
        self.calls: list[CallUsage] = []
        # (provider, provider_model) -> billing catalog id, when there is one
        self._resolve = model_resolver

    def record_provider_usage(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_input_tokens: int | None = 0,
        messages: list[dict[str, Any]] | None = None,
        output_text: str | None = None,
        purpose: str | None = None,
    ) -> CallUsage:
        """Record one call. None for a count means the provider did not report it: it is estimated and flagged."""
        estimated = input_tokens is None or output_tokens is None
        if estimated:
            est_in, est_out = _estimate_tokens(provider, messages, output_text)
            input_tokens = est_in if input_tokens is None else input_tokens
            output_tokens = est_out if output_tokens is None else output_tokens
            try:
                from app.utils.metrics import USAGE_ESTIMATED_TOTAL

                USAGE_ESTIMATED_TOTAL.labels(provider=provider, model=model).inc()
            except Exception:
                pass
            logger.warning("usage_estimated", provider=provider, model=model)
        canonical = _PROVIDER_ALIASES.get(provider, provider)
        resolved = self._resolve(canonical, model) if self._resolve else model
        call = CallUsage(
            provider=canonical, model=resolved or model, purpose=purpose or _purpose.get(),
            input_tokens=max(0, int(input_tokens or 0)), cached_input_tokens=max(0, int(cached_input_tokens or 0)),
            output_tokens=max(0, int(output_tokens or 0)), usage_estimated=estimated,
        )
        self.calls.append(call)
        return call

    # --- views ---------------------------------------------------------------------------------------------------
    def calls_for(self, purpose: str) -> list[CallUsage]:
        return [c for c in self.calls if c.purpose == purpose]

    def totals(self, purpose: str | None = None) -> tuple[int, int]:
        calls = self.calls_for(purpose) if purpose else self.calls
        return sum(c.input_tokens for c in calls), sum(c.output_tokens for c in calls)

    def snapshot(self) -> list[dict[str, Any]]:
        return [c.model_dump() for c in self.calls]

    def any_output(self) -> bool:
        return any(c.output_tokens > 0 for c in self.calls)


def current_collector() -> UsageCollector | None:
    return _collector.get()


@contextmanager
def use_collector(collector: UsageCollector | None) -> Iterator[UsageCollector | None]:
    token = _collector.set(collector)
    try:
        yield collector
    finally:
        _collector.reset(token)


@contextmanager
def usage_purpose(purpose: str) -> Iterator[None]:
    if purpose not in PURPOSES:
        raise ValueError(f"unknown usage purpose {purpose!r}")
    token = _purpose.set(purpose)
    try:
        yield
    finally:
        _purpose.reset(token)
