"""
Google Gemini Provider Adapter.
Implements LLD v2.0 Section 20.4.
"""

import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from google import genai
from google.genai import types

from app.config.logging import get_logger
from app.exceptions.provider import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.providers.base import BaseProviderAdapter
from app.providers.models import GenerationResponse
from app.providers.usage_hook import report_usage
from app.utils.metrics import TOKENS_TOTAL
from app.utils.tracing import get_tracer

logger = get_logger("gemini_adapter")


def gemini_usage_numbers(meta: Any) -> tuple[int | None, int | None, int]:
    """(input, output, cached) from usage_metadata. Gemini bills its thinking tokens as output, and they are NOT in
    candidates_token_count, so they are added to the output."""
    if meta is None or getattr(meta, "prompt_token_count", None) is None:
        return None, None, 0
    out = getattr(meta, "candidates_token_count", None)
    thoughts = getattr(meta, "thoughts_token_count", None) or 0
    return (int(meta.prompt_token_count), None if out is None else int(out) + int(thoughts),
            int(getattr(meta, "cached_content_token_count", 0) or 0))


class GeminiAdapter(BaseProviderAdapter):
    """
    Google Gemini adapter for resilient fallback generation (Fallback).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
        client: genai.Client | None = None,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        super().__init__(provider_name="gemini", model_name=model)
        self.api_key = api_key
        self.client = client or genai.Client(api_key=api_key or "not-set")
        self.logger = logger_instance or logger
        self.tracer = get_tracer()

    def _convert_messages(
        self, messages: list[dict[str, Any]], temperature: float, max_tokens: int
    ) -> tuple[str | None, list[types.Content], types.GenerateContentConfig]:
        """Translates OpenAI message format into Gemini system instruction and content parts."""
        system_instructions: list[str] = []
        contents: list[types.Content] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))

            if role == "system":
                system_instructions.append(content)
            elif role == "assistant":
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=content)],
                    )
                )
            else:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content)],
                    )
                )

        sys_inst = "\n\n".join(system_instructions) if system_instructions else None
        config = types.GenerateContentConfig(
            system_instruction=sys_inst,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        return sys_inst, contents, config

    async def execute(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        """Executes a complete (non-streaming) completion request via Gemini."""
        params = params or {}
        model = params.get("model", self.model_name)
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 8192)

        start_time = time.perf_counter()
        with self.tracer.start_as_current_span("gemini.execute") as span:
            span.set_attribute("provider", "gemini")
            span.set_attribute("model", model)

            try:
                _, contents, config = self._convert_messages(messages, temperature, max_tokens)
                response = await self.client.aio.models.generate_content(
                    model=model,
                    contents=contents,  # type: ignore[arg-type]
                    config=config,
                )
                duration = time.perf_counter() - start_time
                content = response.text or ""

                prompt_tokens = (
                    response.usage_metadata.prompt_token_count
                    if response.usage_metadata
                    else len(content) // 4
                )
                completion_tokens = (
                    response.usage_metadata.candidates_token_count
                    if response.usage_metadata
                    else len(content) // 4
                )
                total_tokens = prompt_tokens + completion_tokens
                in_t, out_t, cached = gemini_usage_numbers(response.usage_metadata)
                report_usage(params, provider="gemini", model=model, input_tokens=in_t, output_tokens=out_t,
                             cached_input_tokens=cached, messages=messages, output_text=content)

                try:
                    TOKENS_TOTAL.labels(provider="gemini", token_type="prompt").inc(prompt_tokens)
                    TOKENS_TOTAL.labels(provider="gemini", token_type="completion").inc(
                        completion_tokens
                    )
                except Exception:
                    pass

                return GenerationResponse(
                    content=content,
                    provider="gemini",
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
        """Streams tokens incrementally from Google Gemini."""
        params = params or {}
        model = params.get("model", self.model_name)
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 8192)

        with self.tracer.start_as_current_span("gemini.stream") as span:
            span.set_attribute("provider", "gemini")
            span.set_attribute("model", model)

            try:
                _, contents, config = self._convert_messages(messages, temperature, max_tokens)
                stream_resp = await self.client.aio.models.generate_content_stream(
                    model=model,
                    contents=contents,  # type: ignore[arg-type]
                    config=config,
                )
                meta = None
                parts: list[str] = []
                try:
                    async for chunk in stream_resp:
                        meta = getattr(chunk, "usage_metadata", None) or meta     # the last chunk carries the totals
                        if chunk.text:
                            parts.append(chunk.text)
                            yield chunk.text
                except Exception:
                    if parts:   # died midway: tokens were consumed; the provider's figures are unknown
                        report_usage(params, provider="gemini", model=model, input_tokens=None, output_tokens=None,
                                     messages=messages, output_text="".join(parts))
                    raise
                in_t, out_t, cached = gemini_usage_numbers(meta)
                report_usage(params, provider="gemini", model=model, input_tokens=in_t, output_tokens=out_t,
                             cached_input_tokens=cached, messages=messages, output_text="".join(parts))
            except Exception as e:
                self._handle_exception(e)

    def _handle_exception(self, e: Exception) -> None:
        """Translates Gemini SDK errors to internal ProviderError hierarchies."""
        err_msg = str(e)
        self.logger.error("Gemini generation failed", error=err_msg)
        if "429" in err_msg or "resource exhausted" in err_msg.lower():
            raise ProviderRateLimitError(f"Gemini Rate Limit: {err_msg}", provider="gemini") from e
        if "timeout" in err_msg.lower() or "deadline exceeded" in err_msg.lower():
            raise ProviderTimeoutError(f"Gemini Timeout: {err_msg}", provider="gemini") from e
        raise ProviderError(f"Gemini Error: {err_msg}", provider="gemini") from e
