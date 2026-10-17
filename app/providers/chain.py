"""
Ordered provider chain for answer generation.

Providers are tried in order until one produces a token. A provider that errors, is rate limited, or
produces no first token before its deadline is skipped and the next one is tried; once any text has been
streamed the turn never switches provider (that would splice two different answers together).
Each entry has its own circuit breaker, so a provider that keeps failing is skipped immediately.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import structlog

from app.config.logging import get_logger
from app.context_window.models import TrimmedPrompt
from app.exceptions.provider import AllProvidersFailedError
from app.providers.base import BaseProviderAdapter
from app.providers.router import with_deadlines
from app.utils.circuit_breaker import CircuitBreaker
from app.utils.metrics import GENERATION_FALLBACK_TOTAL

logger = get_logger("provider_chain")


@dataclass
class ChainEntry:
    name: str
    adapter: BaseProviderAdapter
    breaker: CircuitBreaker | None
    first_token_timeout_s: float | None


class ProviderChain:
    def __init__(
        self,
        entries: list[ChainEntry],
        idle_timeout_s: float | None = 20.0,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        if not entries:
            raise ValueError("ProviderChain needs at least one provider")
        self.entries = entries
        self.idle_timeout_s = idle_timeout_s
        self.logger = logger_instance or logger

    async def generate_stream(
        self, prompt: TrimmedPrompt, params: dict[str, Any] | None = None
    ) -> AsyncIterator[str]:
        params = params if params is not None else {}
        route: dict[str, Any] = params.setdefault("_route", {})
        failures: list[tuple[str, str]] = []

        for idx, entry in enumerate(self.entries):
            if entry.breaker and entry.breaker.is_open():
                failures.append((entry.name, "circuit breaker open"))
                self._record(entry.name, idx)
                continue

            emitted = False
            try:
                async for token in with_deadlines(
                    entry.adapter.stream(prompt.messages, params),
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
                    yield token
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

        raise AllProvidersFailedError(
            "All generation providers failed. " + "; ".join(f"{n}: {m}" for n, m in failures)
        )

    def _record(self, from_name: str, idx: int) -> None:
        to_name = self.entries[idx + 1].name if idx + 1 < len(self.entries) else "none"
        try:
            GENERATION_FALLBACK_TOTAL.labels(from_provider=from_name, to_provider=to_name).inc()
        except Exception:
            pass
