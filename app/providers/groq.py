"""
Groq Provider Adapter.
Implements LLD v2.0 Section 20.2.
"""

import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from groq import AsyncGroq

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

logger = get_logger("groq_adapter")


class GroqAdapter(BaseProviderAdapter):
    """
    Groq adapter used primarily for high-speed Request Analysis (Call 1).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout_s: float = 5.0,
        client: AsyncGroq | None = None,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        super().__init__(provider_name="groq", model_name=model)
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.client = client or AsyncGroq(
            api_key=api_key or "not-set",
            timeout=timeout_s,
        )
        self.logger = logger_instance or logger
        self.tracer = get_tracer()

    async def execute(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        """Executes a JSON or text completion via Groq."""
        params = params or {}
        model = params.get("model", self.model_name)
        temperature = params.get("temperature", 0.0)
        max_tokens = params.get("max_tokens", 2048)
        response_format = params.get("response_format")

        start_time = time.perf_counter()
        with self.tracer.start_as_current_span("groq.execute") as span:
            span.set_attribute("provider", "groq")
            span.set_attribute("model", model)

            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                response = await self.client.chat.completions.create(**kwargs)
                duration = time.perf_counter() - start_time
                choice = response.choices[0]
                content = choice.message.content or ""

                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                total_tokens = response.usage.total_tokens if response.usage else 0

                try:
                    TOKENS_TOTAL.labels(provider="groq", token_type="prompt").inc(prompt_tokens)
                    TOKENS_TOTAL.labels(provider="groq", token_type="completion").inc(
                        completion_tokens
                    )
                except Exception:
                    pass

                return GenerationResponse(
                    content=content,
                    provider="groq",
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
        """Streams tokens from Groq."""
        params = params or {}
        model = params.get("model", self.model_name)
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 4096)

        with self.tracer.start_as_current_span("groq.stream") as span:
            span.set_attribute("provider", "groq")
            span.set_attribute("model", model)

            try:
                stream_resp = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
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
        """Translates Groq SDK errors to internal ProviderError hierarchies."""
        err_msg = str(e)
        self.logger.error("Groq execution failed", error=err_msg)
        if "429" in err_msg or "rate limit" in err_msg.lower():
            raise ProviderRateLimitError(f"Groq Rate Limit: {err_msg}", provider="groq") from e
        if "timeout" in err_msg.lower() or "timed out" in err_msg.lower():
            raise ProviderTimeoutError(f"Groq Timeout: {err_msg}", provider="groq") from e
        raise ProviderError(f"Groq Error: {err_msg}", provider="groq") from e
