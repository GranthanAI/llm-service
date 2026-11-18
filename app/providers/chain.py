"""
Ordered provider chain for answer generation.

Providers are tried in order until one produces a token. A provider that errors, is rate limited, or
produces no first token before its deadline is skipped and the next one is tried; once any text has been
streamed the turn never switches provider (that would splice two different answers together).
Each entry has its own circuit breaker, so a provider that keeps failing is skipped immediately.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import structlog

from app.config.logging import get_logger
from app.context_window.models import TrimmedPrompt
from app.exceptions.analysis import ContextOverflowError
from app.exceptions.provider import AllProvidersFailedError
from app.providers.base import BaseProviderAdapter
from app.providers.groq_budget import BUDGET_WAITS
from app.providers.router import with_deadlines
from app.utils.circuit_breaker import CircuitBreaker
from app.utils.metrics import GENERATION_FALLBACK_TOTAL

logger = get_logger("provider_chain")
MAX_BUDGET_WAIT_S = 8.0   # wait for a short gap in every Groq budget; a longer one fails the turn (hold released)


@dataclass
class ChainEntry:
    name: str
    adapter: BaseProviderAdapter
    breaker: CircuitBreaker | None
    first_token_timeout_s: float | None
    # Set only for billing group chains: the registry entry (catalog id, window, output limit) and the grant's
    # context cap. None for the legacy GENERATION_PROVIDER_ORDER chain.
    spec: Any = None
    input_cap: int | None = None


class ProviderChain:
    def __init__(
        self,
        entries: list[ChainEntry],
        idle_timeout_s: float | None = 20.0,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
        fitter: Callable[[TrimmedPrompt, str], TrimmedPrompt] | None = None,
    ):
        if not entries:
            raise ValueError("ProviderChain needs at least one provider")
        self.entries = entries
        # Re-validates the prompt against each provider's own context budget right before it is tried (see
        # app.context_window.fit.build_chain_fitter). None = send the prompt as it is (legacy behaviour).
        self.fitter = fitter
        self.idle_timeout_s = idle_timeout_s
        self.logger = logger_instance or logger

    async def generate_stream(
        self, prompt: TrimmedPrompt, params: dict[str, Any] | None = None
    ) -> AsyncIterator[str]:
        params = params if params is not None else {}
        route: dict[str, Any] = params.setdefault("_route", {})
        failures: list[tuple[str, str]] = []
        short_waits: list[float] = []   # Groq models skipped only for lack of budget, and how long until they fit
        for round_ in range(2):
            started = False
            async for token in self._round(prompt, params, route, failures, short_waits, last=round_ == 1):
                started = True
                yield token
            if started or params.pop("_chain_done", False):
                return
            if round_ == 0 and short_waits and min(short_waits) <= MAX_BUDGET_WAIT_S:
                wait = min(short_waits)
                BUDGET_WAITS.labels(purpose="answer").inc()
                self.logger.warning("groq_budget_wait", wait_s=round(wait, 1))
                await asyncio.sleep(wait + 0.25)
                short_waits.clear()
                failures.clear()
                continue
            break
        raise AllProvidersFailedError(
            "All generation providers failed. " + "; ".join(f"{n}: {m}" for n, m in failures)
        )

    async def _round(self, prompt: TrimmedPrompt, params: dict[str, Any], route: dict[str, Any],
                     failures: list[tuple[str, str]], short_waits: list[float], last: bool) -> AsyncIterator[str]:
        """One pass over the entries in order. Yields tokens when one answers; otherwise yields nothing."""
        for idx, entry in enumerate(self.entries):
            if entry.breaker and entry.breaker.is_open():
                failures.append((entry.name, "circuit breaker open"))
                self._record(entry.name, idx)
                continue

            attempt_prompt = prompt
            if self.fitter is not None:
                try:
                    attempt_prompt = (
                        self.fitter(prompt, entry.name, entry) if entry.spec is not None else self.fitter(prompt, entry.name)
                    )
                except ContextOverflowError as exc:
                    # Not a provider failure and not a rate limit: the request cannot fit THIS provider's budget.
                    failures.append((entry.name, f"context_too_large: {exc}"))
                    self.logger.warning("Skipping provider: prompt does not fit its context budget", provider=entry.name)
                    self._record(entry.name, idx)
                    continue

            hook = getattr(entry.adapter, "budget_wait", None)
            if hook is not None:
                wait = await hook(attempt_prompt.messages, params)
                if wait > 0:
                    short_waits.append(wait)
                    failures.append((entry.name, f"no token budget for {wait:.0f}s"))
                    self.logger.info("groq_budget_skip", provider=entry.name, wait_s=round(wait, 1))
                    self._record(entry.name, idx)
                    continue

            emitted = False
            try:
                async for token in with_deadlines(
                    entry.adapter.stream(attempt_prompt.messages, params),
                    entry.first_token_timeout_s,
                    self.idle_timeout_s,
                    entry.name,
                ):
                    if not emitted:
                        emitted = True
                        if entry.breaker:
                            await entry.breaker.on_success()
                        route.update(
                            provider=entry.name,
                            fallback_used=bool(failures),
                            fallback_reason=failures[-1][1] if failures else None,
                        )
                        if entry.spec is not None:     # billing group chain only: the catalog id that answered
                            route["model_id"] = entry.spec.id
                    yield token
                params["_chain_done"] = True   # finished (even with no text): not a failure
                return
            except Exception as exc:
                if entry.breaker:
                    await entry.breaker.on_failure()
                if emitted:
                    self.logger.error(
                        "Provider failed after streaming started; not switching mid-answer",
                        provider=entry.name,
                        error=str(exc),
                    )
                    raise
                failures.append((entry.name, f"{type(exc).__name__}: {exc}"))
                self.logger.warning(
                    "Provider produced no answer; trying the next one",
                    provider=entry.name,
                    reason=type(exc).__name__,
                    error=str(exc)[:200],
                    remaining=len(self.entries) - idx - 1,
                )
                self._record(entry.name, idx)

    def _record(self, from_name: str, idx: int) -> None:
        to_name = self.entries[idx + 1].name if idx + 1 < len(self.entries) else "none"
        try:
            GENERATION_FALLBACK_TOTAL.labels(from_provider=from_name, to_provider=to_name).inc()
        except Exception:
            pass
