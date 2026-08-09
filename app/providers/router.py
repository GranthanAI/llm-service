"""
Generation Router with Circuit Breaker and Provider Fallback.
Implements LLD v2.0 Section 19.1 and Section 19.2.
"""

from collections.abc import AsyncIterator
from typing import Any

import structlog

from app.config.logging import get_logger
from app.context_window.models import TrimmedPrompt
from app.exceptions.provider import (
    AllProvidersFailedError,
    ProviderError,
)
from app.providers.base import BaseProviderAdapter
from app.providers.models import GenerationResponse
from app.utils.circuit_breaker import CircuitBreaker
from app.utils.metrics import GENERATION_FALLBACK_TOTAL
from app.utils.retry import RetryManager
from app.utils.tracing import get_tracer

logger = get_logger("generation_router")


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
    ):
        self.primary = nvidia_adapter
        self.fallback = gemini_adapter
        self.circuit_breakers = circuit_breakers
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
        params = params or {}
        with self.tracer.start_as_current_span("generation_router.generate_stream") as span:
            span.set_attribute("mode", prompt.mode)
            span.set_attribute("target_model", prompt.target_model)

            nvidia_cb = self.circuit_breakers.get("nvidia")
            gemini_cb = self.circuit_breakers.get("gemini")

            # 1. Check if Primary (NVIDIA) is open
            if nvidia_cb and nvidia_cb.is_open():
                self.logger.warning(
                    "NVIDIA circuit breaker OPEN, routing directly to fallback (Gemini)"
                )
                self._record_fallback("nvidia", "gemini")
                async for token in self._stream_fallback(prompt, params, gemini_cb):
                    yield token
                return

            # 2. Try Primary (NVIDIA) with fallback
            try:
                first_token_received = False
                async for token in self.primary.stream(prompt.messages, params):
                    if not first_token_received:
                        first_token_received = True
                        if nvidia_cb:
                            await nvidia_cb.on_success()
                    yield token

            except (ProviderError, Exception) as e:
                self.logger.warning(
                    "Primary provider (NVIDIA) failed during generation, initiating fallback to Gemini",
                    error=str(e),
                )
                if nvidia_cb:
                    await nvidia_cb.on_failure()

                self._record_fallback("nvidia", "gemini")

                async for token in self._stream_fallback(prompt, params, gemini_cb):
                    yield token

    async def _stream_fallback(
        self,
        prompt: TrimmedPrompt,
        params: dict[str, Any],
        gemini_cb: CircuitBreaker | None,
    ) -> AsyncIterator[str]:
        """Streams tokens from the fallback adapter."""
        if gemini_cb and gemini_cb.is_open():
            self.logger.error("Both NVIDIA and Gemini circuit breakers are OPEN")
            raise AllProvidersFailedError("Both NVIDIA and Gemini providers are unavailable")

        try:
            first_token_received = False
            async for token in self.fallback.stream(prompt.messages, params):
                if not first_token_received:
                    first_token_received = True
                    if gemini_cb:
                        await gemini_cb.on_success()
                yield token
        except Exception as e:
            if gemini_cb:
                await gemini_cb.on_failure()
            self.logger.error("Fallback provider (Gemini) failed as well", error=str(e))
            raise AllProvidersFailedError(
                f"All generation providers failed. Gemini error: {e}"
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
            gemini_cb = self.circuit_breakers.get("gemini")

            # 1. If NVIDIA circuit is open, route to Gemini
            if nvidia_cb and nvidia_cb.is_open():
                self._record_fallback("nvidia", "gemini")
                return await self._execute_fallback(prompt, params, gemini_cb)

            # 2. Try NVIDIA
            try:
                resp = await self.primary.execute(prompt.messages, params)
                if nvidia_cb:
                    await nvidia_cb.on_success()
                return resp
            except (ProviderError, Exception) as e:
                self.logger.warning(
                    "Primary provider (NVIDIA) execution failed, falling back to Gemini",
                    error=str(e),
                )
                if nvidia_cb:
                    await nvidia_cb.on_failure()
                self._record_fallback("nvidia", "gemini")
                return await self._execute_fallback(prompt, params, gemini_cb)

    async def _execute_fallback(
        self,
        prompt: TrimmedPrompt,
        params: dict[str, Any],
        gemini_cb: CircuitBreaker | None,
    ) -> GenerationResponse:
        """Executes non-streaming completion using fallback provider."""
        if gemini_cb and gemini_cb.is_open():
            raise AllProvidersFailedError("Both NVIDIA and Gemini providers are unavailable")

        try:
            resp = await self.fallback.execute(prompt.messages, params)
            if gemini_cb:
                await gemini_cb.on_success()
            return resp
        except Exception as e:
            if gemini_cb:
                await gemini_cb.on_failure()
            raise AllProvidersFailedError(
                f"All generation providers failed. Gemini error: {e}"
            ) from e

    def _record_fallback(self, from_provider: str, to_provider: str) -> None:
        """Records Prometheus fallback counter metric."""
        try:
            GENERATION_FALLBACK_TOTAL.labels(
                from_provider=from_provider, to_provider=to_provider
            ).inc()
        except Exception:
            pass
