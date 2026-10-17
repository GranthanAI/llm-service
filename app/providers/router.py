"""
Generation Router with Circuit Breaker and Provider Fallback.
Implements LLD v2.0 Section 19.1 and Section 19.2.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import structlog

from app.config.logging import get_logger
from app.context_window.models import TrimmedPrompt
from app.exceptions.provider import (
    AllProvidersFailedError,
    ProviderError,
    ProviderTimeoutError,
)
from app.providers.base import BaseProviderAdapter
from app.providers.models import GenerationResponse
from app.utils.circuit_breaker import CircuitBreaker
from app.utils.metrics import GENERATION_FALLBACK_TOTAL
from app.utils.retry import RetryManager
from app.utils.tracing import get_tracer

logger = get_logger("generation_router")


async def with_deadlines(
    source: AsyncIterator[str],
    first_timeout_s: float | None,
    idle_timeout_s: float | None,
    label: str,
) -> AsyncIterator[str]:
    """Yield tokens from `source`, enforcing a first-token and an inter-token deadline."""
    it = source.__aiter__()
    first = True
    try:
        while True:
            limit = first_timeout_s if first else idle_timeout_s
            try:
                token = await asyncio.wait_for(it.__anext__(), timeout=limit)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                stage = "first token" if first else "next token"
                raise ProviderTimeoutError(
                    f"{label} provider produced no {stage} within {limit:g}s",
                    provider=label,
                ) from exc
            first = False
            yield token
    finally:
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass


class GenerationRouter:
    """
    Routes generation requests to primary provider (NVIDIA NIM), falling back
    transparently to secondary provider (Gemini) on circuit open or transient failures.
    """

    def __init__(
        self,
        nvidia_adapter: BaseProviderAdapter,
        gemini_adapter: BaseProviderAdapter,
        circuit_breakers: dict[str, CircuitBreaker],
        retry_manager: RetryManager | None = None,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
        fallback_enabled: bool = True,
        fallback_name: str = "gemini",
        first_token_timeout_s: float | None = 8.0,
        idle_timeout_s: float | None = 20.0,
        fallback_first_token_timeout_s: float | None = 30.0,
    ):
        self.primary = nvidia_adapter
        # The fallback slot is historically called `gemini_adapter`; it can hold any
        # adapter (GENERATION_FALLBACK_PROVIDER = gemini | groq). `fallback_name` keys its
        # circuit breaker and labels logs/metrics.
        self.fallback = gemini_adapter
        self.fallback_name = fallback_name
        self.fallback_label = fallback_name.capitalize()
        # When False the fallback provider is never called (GENERATION_FALLBACK_PROVIDER=none).
        self.fallback_enabled = fallback_enabled
        self.circuit_breakers = circuit_breakers
        # Deadlines on the visible token stream (None disables). See settings.py.
        self.first_token_timeout_s = first_token_timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.fallback_first_token_timeout_s = fallback_first_token_timeout_s
        self.retry_manager = retry_manager or RetryManager()
        self.logger = logger_instance or logger
        self.tracer = get_tracer()

    async def generate_stream(
        self,
        prompt: TrimmedPrompt,
        params: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Streams generated tokens, routing to NVIDIA primary and falling back to Gemini.
        """
        params = params if params is not None else {}
        with self.tracer.start_as_current_span("generation_router.generate_stream") as span:
            span.set_attribute("mode", prompt.mode)
            span.set_attribute("target_model", prompt.target_model)

            nvidia_cb = self.circuit_breakers.get("nvidia")
            gemini_cb = self.circuit_breakers.get(self.fallback_name)

            route: dict[str, Any] = params.setdefault("_route", {})
            route.update(provider="primary", fallback_used=False, fallback_reason=None)

            # 1. Check if Primary (NVIDIA) is open
            if nvidia_cb and nvidia_cb.is_open():
                self.logger.warning(
                    f"NVIDIA circuit breaker OPEN, routing directly to fallback ({self.fallback_label})"
                )
                self._record_fallback("nvidia", self.fallback_name)
                route.update(provider=self.fallback_name, fallback_used=True, fallback_reason="circuit_open")
                async for token in self._stream_fallback(prompt, params, gemini_cb):
                    yield token
                return

            # 2. Try Primary (NVIDIA) with fallback. The primary is held to a first-visible-token
            # deadline; once any token has been emitted we never switch provider (that would
            # splice two different answers together) and a failure is surfaced instead.
            emitted = False
            try:
                async for token in self._with_deadlines(
                    self.primary.stream(prompt.messages, params),
                    self.first_token_timeout_s,
                    self.idle_timeout_s,
                    "primary",
                ):
                    if not emitted:
                        emitted = True
                        if nvidia_cb:
                            await nvidia_cb.on_success()
                    yield token
                return

            except (ProviderError, Exception) as e:
                if nvidia_cb:
                    await nvidia_cb.on_failure()
                if emitted:
                    self.logger.error(
                        "Primary provider failed after streaming started; not falling back mid-answer",
                        error=str(e),
                    )
                    raise
                self.logger.warning(
                    f"Primary provider (NVIDIA) produced no answer, falling back to {self.fallback_label}",
                    error=str(e),
                    reason=type(e).__name__,
                )
                self._record_fallback("nvidia", self.fallback_name)
                route.update(
                    provider=self.fallback_name, fallback_used=True, fallback_reason=type(e).__name__
                )

            async for token in self._stream_fallback(prompt, params, gemini_cb):
                yield token

    def _with_deadlines(self, source, first_timeout_s, idle_timeout_s, label):
        return with_deadlines(source, first_timeout_s, idle_timeout_s, label)

    async def _stream_fallback(
        self,
        prompt: TrimmedPrompt,
        params: dict[str, Any],
        gemini_cb: CircuitBreaker | None,
    ) -> AsyncIterator[str]:
        """Streams tokens from the fallback adapter."""
        if not self.fallback_enabled:
            self.logger.error("Primary provider failed and the generation fallback is disabled")
            raise AllProvidersFailedError("Primary provider failed and the generation fallback is disabled")
        if gemini_cb and gemini_cb.is_open():
            self.logger.error(f"Both NVIDIA and {self.fallback_label} circuit breakers are OPEN")
            raise AllProvidersFailedError(f"Both NVIDIA and {self.fallback_label} providers are unavailable")

        try:
            first_token_received = False
            async for token in self._with_deadlines(
                self.fallback.stream(prompt.messages, params),
                self.fallback_first_token_timeout_s,
                self.idle_timeout_s,
                self.fallback_name,
            ):
                if not first_token_received:
                    first_token_received = True
                    if gemini_cb:
                        await gemini_cb.on_success()
                yield token
        except Exception as e:
            if gemini_cb:
                await gemini_cb.on_failure()
            self.logger.error(f"Fallback provider ({self.fallback_label}) failed as well", error=str(e))
            raise AllProvidersFailedError(
                f"All generation providers failed. {self.fallback_label} error: {e}"
            ) from e

    async def generate(
        self,
        prompt: TrimmedPrompt,
        params: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        """
        Executes non-streaming generation with primary -> fallback routing.
        """
        params = params or {}
        with self.tracer.start_as_current_span("generation_router.generate") as span:
            span.set_attribute("mode", prompt.mode)

            nvidia_cb = self.circuit_breakers.get("nvidia")
            gemini_cb = self.circuit_breakers.get(self.fallback_name)

            # 1. If NVIDIA circuit is open, route to Gemini
            if nvidia_cb and nvidia_cb.is_open():
                self._record_fallback("nvidia", self.fallback_name)
                return await self._execute_fallback(prompt, params, gemini_cb)

            # 2. Try NVIDIA
            try:
                resp = await self.primary.execute(prompt.messages, params)
                if nvidia_cb:
                    await nvidia_cb.on_success()
                return resp
            except (ProviderError, Exception) as e:
                self.logger.warning(
                    f"Primary provider (NVIDIA) execution failed, falling back to {self.fallback_label}",
                    error=str(e),
                )
                if nvidia_cb:
                    await nvidia_cb.on_failure()
                self._record_fallback("nvidia", self.fallback_name)
                return await self._execute_fallback(prompt, params, gemini_cb)

    async def _execute_fallback(
        self,
        prompt: TrimmedPrompt,
        params: dict[str, Any],
        gemini_cb: CircuitBreaker | None,
    ) -> GenerationResponse:
        """Executes non-streaming completion using fallback provider."""
        if not self.fallback_enabled:
            raise AllProvidersFailedError("Primary provider failed and the generation fallback is disabled")
        if gemini_cb and gemini_cb.is_open():
            raise AllProvidersFailedError(f"Both NVIDIA and {self.fallback_label} providers are unavailable")

        try:
            resp = await self.fallback.execute(prompt.messages, params)
            if gemini_cb:
                await gemini_cb.on_success()
            return resp
        except Exception as e:
            if gemini_cb:
                await gemini_cb.on_failure()
            raise AllProvidersFailedError(
                f"All generation providers failed. {self.fallback_label} error: {e}"
            ) from e

    def _record_fallback(self, from_provider: str, to_provider: str) -> None:
        """Records Prometheus fallback counter metric."""
        try:
            GENERATION_FALLBACK_TOTAL.labels(
                from_provider=from_provider, to_provider=to_provider
            ).inc()
        except Exception:
            pass
