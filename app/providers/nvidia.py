"""
NVIDIA NIM Provider Adapter.
Implements LLD v2.0 Section 20.3.
"""

import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from openai import AsyncOpenAI

from app.config.logging import get_logger
from app.exceptions.provider import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.providers.base import BaseProviderAdapter
from app.providers.models import GenerationResponse
from app.utils.metrics import TOKENS_TOTAL
from app.utils.tracing import get_tracer

logger = get_logger("nvidia_adapter")


class NVIDIAAdapter(BaseProviderAdapter):
    """
    NVIDIA NIM OpenAI-compatible adapter for high-performance generation (Primary).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "meta/llama-3.3-70b-instruct",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout_s: float = 30.0,
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 1024,
        client: AsyncOpenAI | None = None,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        super().__init__(provider_name="nvidia", model_name=model)
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.default_max_tokens = max_tokens
        self.client = client or AsyncOpenAI(
            api_key=api_key or "not-set",
            base_url=base_url,
            timeout=timeout_s,
        )
        self.logger = logger_instance or logger
        self.tracer = get_tracer()

    async def execute(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        """Executes a synchronous/complete completion request."""
        params = params or {}
        model = params.get("model", self.model_name)
        temperature = params.get("temperature", self.default_temperature)
        top_p = params.get("top_p", self.default_top_p)
        max_tokens = params.get("max_tokens", self.default_max_tokens)

        start_time = time.perf_counter()
        with self.tracer.start_as_current_span("nvidia.execute") as span:
            span.set_attribute("provider", "nvidia")
            span.set_attribute("model", model)

            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=False,
                )
                duration = time.perf_counter() - start_time
                choice = response.choices[0]
                content = choice.message.content or ""
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                total_tokens = response.usage.total_tokens if response.usage else 0

                try:
                    TOKENS_TOTAL.labels(provider="nvidia", token_type="prompt").inc(prompt_tokens)
                    TOKENS_TOTAL.labels(provider="nvidia", token_type="completion").inc(
                        completion_tokens
                    )
                except Exception:
                    pass

                return GenerationResponse(
                    content=content,
                    provider="nvidia",
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    duration_s=duration,
                )
            except Exception as e:
                self._handle_exception(e)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Streams tokens incrementally from NVIDIA NIM."""
        params = params or {}
        model = params.get("model", self.model_name)
        temperature = params.get("temperature", self.default_temperature)
        top_p = params.get("top_p", self.default_top_p)
        max_tokens = params.get("max_tokens", self.default_max_tokens)

        with self.tracer.start_as_current_span("nvidia.stream") as span:
            span.set_attribute("provider", "nvidia")
            span.set_attribute("model", model)

            try:
                stream_resp = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=True,
                )
                async for chunk in stream_resp:
                    if chunk.choices and chunk.choices[0].delta:
                        content_piece = chunk.choices[0].delta.content
                        if content_piece:
                            yield content_piece
            except Exception as e:
                self._handle_exception(e)

    def _handle_exception(self, e: Exception) -> None:
        """Translates external SDK errors to internal ProviderError hierarchies."""
        err_msg = str(e)
        self.logger.error("NVIDIA generation failed", error=err_msg)
        if "429" in err_msg or "rate limit" in err_msg.lower():
            raise ProviderRateLimitError(f"NVIDIA Rate Limit: {err_msg}", provider="nvidia") from e
        if "timeout" in err_msg.lower() or "timed out" in err_msg.lower():
            raise ProviderTimeoutError(f"NVIDIA Timeout: {err_msg}", provider="nvidia") from e
        raise ProviderError(f"NVIDIA Error: {err_msg}", provider="nvidia") from e
