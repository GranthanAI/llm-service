"""
OpenAI provider adapter, over the async OpenAI SDK (chat completions).

Same contract as the Groq / Gemini adapters: one retry for a short rate limit, mapped errors, a visible notice when
the answer is cut off by the output limit, no retry once text has been shown. Streaming asks for usage
(stream_options.include_usage); the usage arrives on the final chunk, which has no choices.
"""

import asyncio
import random
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from openai import AsyncOpenAI

from app.config.logging import get_logger
from app.exceptions.provider import ProviderError, ProviderRateLimitError, ProviderTimeoutError
from app.providers.base import TRUNCATION_NOTICE, BaseProviderAdapter
from app.providers.groq import MAX_RATE_LIMIT_WAIT_S, is_context_limit_error, rate_limit_wait_s
from app.providers.models import GenerationResponse
from app.providers.usage_hook import report_usage
from app.utils import metrics
from app.utils.metrics import TOKENS_TOTAL
from app.utils.tracing import get_tracer

logger = get_logger("openai_adapter")

DEFAULT_MAX_TOKENS = 4096


def openai_usage_numbers(usage: Any) -> tuple[int | None, int | None, int]:
    """(prompt_tokens, completion_tokens, cached_tokens) from an OpenAI-style usage object (also NVIDIA's). Prompt
    tokens include the cached ones."""
    if usage is None:
        return None, None, 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    return (None if prompt is None else int(prompt)), (None if completion is None else int(completion)), int(cached or 0)


class OpenAIAdapter(BaseProviderAdapter):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-6-luna",
        timeout_s: float = 60.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: AsyncOpenAI | None = None,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        super().__init__(provider_name="openai", model_name=model)
        self.api_key = api_key
        self.default_max_tokens = max_tokens
        self.client = client or AsyncOpenAI(api_key=api_key or "not-set", timeout=timeout_s, max_retries=0)
        self.logger = logger_instance or logger
        self.tracer = get_tracer()

    def _kwargs(self, messages: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": params.get("model", self.model_name),
            "messages": messages,
            # current OpenAI models take max_completion_tokens (max_tokens is rejected by reasoning models)
            "max_completion_tokens": int(params.get("max_tokens") or self.default_max_tokens),
        }
        if params.get("temperature") is not None and params.get("_send_temperature"):
            kwargs["temperature"] = params["temperature"]
        return kwargs

    async def execute(self, messages: list[dict[str, Any]], params: dict[str, Any] | None = None) -> GenerationResponse:
        params = params or {}
        kwargs = self._kwargs(messages, params)
        if params.get("response_format"):
            kwargs["response_format"] = params["response_format"]
        start = time.perf_counter()
        with self.tracer.start_as_current_span("openai.execute") as span:
            span.set_attribute("provider", "openai")
            span.set_attribute("model", kwargs["model"])
            try:
                response = await self.client.chat.completions.create(stream=False, **kwargs)
            except Exception as e:
                self._handle_exception(e)
            content = (response.choices[0].message.content or "") if response.choices else ""
            prompt, completion, cached = openai_usage_numbers(getattr(response, "usage", None))
            report_usage(params, provider="openai", model=kwargs["model"], input_tokens=prompt,
                         output_tokens=completion, cached_input_tokens=cached)
            self._count(prompt or 0, completion or 0)
            return GenerationResponse(
                content=content, provider="openai", model=kwargs["model"], prompt_tokens=prompt or 0,
                completion_tokens=completion or 0, total_tokens=(prompt or 0) + (completion or 0),
                duration_s=time.perf_counter() - start,
            )

    async def stream(self, messages: list[dict[str, Any]], params: dict[str, Any] | None = None) -> AsyncIterator[str]:
        params = params or {}
        kwargs = self._kwargs(messages, params)
        model = kwargs["model"]
        with self.tracer.start_as_current_span("openai.stream") as span:
            span.set_attribute("provider", "openai")
            span.set_attribute("model", model)
            emitted = False
            rate_limited = False
            while True:
                prompt = completion = None
                cached = 0
                truncated = False
                try:
                    stream_resp = await self.client.chat.completions.create(
                        stream=True, stream_options={"include_usage": True}, **kwargs
                    )
                    async for chunk in stream_resp:
                        if getattr(chunk, "usage", None) is not None:      # the final chunk (no choices)
                            prompt, completion, cached = openai_usage_numbers(chunk.usage)
                        if chunk.choices:
                            choice = chunk.choices[0]
                            piece = choice.delta.content if choice.delta else None
                            if piece:
                                emitted = True
                                yield piece
                            if choice.finish_reason == "length":
                                truncated = True
                    report_usage(params, provider="openai", model=model, input_tokens=prompt,
                                 output_tokens=completion, cached_input_tokens=cached)
                    self._count(prompt or 0, completion or 0)
                    if truncated:
                        self.logger.warning("OpenAI answer hit the output token limit", max_tokens=kwargs["max_completion_tokens"])
                        yield TRUNCATION_NOTICE
                    return
                except Exception as e:
                    wait = rate_limit_wait_s(e)
                    if wait is not None and not emitted and wait <= MAX_RATE_LIMIT_WAIT_S and not rate_limited:
                        rate_limited = True
                        self.logger.warning("OpenAI rate limit reached; waiting briefly before one retry", wait_s=wait)
                        metrics.PROVIDER_RATE_LIMITED_TOTAL.labels(provider="openai", action="retry").inc()
                        await asyncio.sleep(wait + 0.25 + random.uniform(0.0, min(0.25, 0.1 * wait)))
                        continue
                    if wait is not None:
                        metrics.PROVIDER_RATE_LIMITED_TOTAL.labels(provider="openai", action="fallback").inc()
                    self._handle_exception(e)

    @staticmethod
    def _count(prompt: int, completion: int) -> None:
        try:
            TOKENS_TOTAL.labels(provider="openai", token_type="prompt").inc(prompt)
            TOKENS_TOTAL.labels(provider="openai", token_type="completion").inc(completion)
        except Exception:
            pass

    def _handle_exception(self, e: Exception) -> None:
        err = str(e)
        self.logger.error("OpenAI request failed", error=err[:300])
        name = type(e).__name__
        status = getattr(e, "status_code", None)
        if is_context_limit_error(e):
            raise ProviderError(f"OpenAI rejected the request size (context limit): {err[:200]}", provider="openai") from e
        if status == 429 or name == "RateLimitError" or "429" in err or "rate limit" in err.lower():
            raise ProviderRateLimitError(f"OpenAI Rate Limit: {err}", provider="openai") from e
        if name in ("APITimeoutError", "TimeoutError") or "timeout" in err.lower() or "timed out" in err.lower():
            raise ProviderTimeoutError(f"OpenAI Timeout: {err}", provider="openai") from e
        raise ProviderError(f"OpenAI Error: {err}", provider="openai", status_code=status) from e
