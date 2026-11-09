"""
Anthropic (Claude) provider adapter, over the async Anthropic SDK.

Same contract as the Groq / Gemini adapters: one retry for a short rate limit (a long one is reported at once so the
chain moves on), SDK errors mapped to ProviderError / ProviderRateLimitError / ProviderTimeoutError, a visible
notice when the answer is cut off by the output limit, no retry once text has been shown.
"""

import asyncio
import random
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from anthropic import AsyncAnthropic

from app.config.logging import get_logger
from app.exceptions.provider import ProviderError, ProviderRateLimitError, ProviderTimeoutError
from app.providers.base import TRUNCATION_NOTICE, BaseProviderAdapter
from app.providers.groq import MAX_RATE_LIMIT_WAIT_S, rate_limit_wait_s
from app.providers.models import GenerationResponse
from app.providers.usage_hook import report_usage
from app.utils import metrics
from app.utils.metrics import TOKENS_TOTAL
from app.utils.tracing import get_tracer

logger = get_logger("anthropic_adapter")

DEFAULT_MAX_TOKENS = 4096


def to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """OpenAI-style messages -> (system, messages) for the Messages API.

    The system prompt is a separate parameter (system messages are joined), roles must alternate and start with
    `user` (consecutive same-role messages are merged; a leading assistant turn gets a short user lead-in), and empty
    messages are dropped (the API rejects empty content)."""
    system_parts: list[str] = []
    merged: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        text = str(msg.get("content", "") or "")
        if role == "system":
            if text.strip():
                system_parts.append(text)
            continue
        role = "assistant" if role == "assistant" else "user"
        if not text.strip():
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n\n" + text
        else:
            merged.append({"role": role, "content": text})
    if not merged:
        merged.append({"role": "user", "content": "Hello"})
    elif merged[0]["role"] == "assistant":
        merged.insert(0, {"role": "user", "content": "(earlier conversation)"})
    return ("\n\n".join(system_parts) or None), merged


def _usage_numbers(usage: Any) -> tuple[int | None, int | None, int]:
    """(input, output, cached) from an Anthropic usage object. Input counts the cache-read tokens as well, so cost
    code can split them: input_tokens is the TOTAL prompt (uncached + cache creation + cache read)."""
    if usage is None:
        return None, None, 0
    uncached = getattr(usage, "input_tokens", None)
    created = getattr(usage, "cache_creation_input_tokens", 0) or 0
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    total_in = None if uncached is None else int(uncached) + int(created) + int(read)
    out = getattr(usage, "output_tokens", None)
    return total_in, (None if out is None else int(out)), int(read)


class AnthropicAdapter(BaseProviderAdapter):
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5-5",
        timeout_s: float = 60.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: AsyncAnthropic | None = None,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        super().__init__(provider_name="anthropic", model_name=model)
        self.api_key = api_key
        self.default_max_tokens = max_tokens
        # max_retries=0: a silent SDK retry would hide a stall from the chain's first-token deadline.
        self.client = client or AsyncAnthropic(api_key=api_key or "not-set", timeout=timeout_s, max_retries=0)
        self.logger = logger_instance or logger
        self.tracer = get_tracer()

    def _kwargs(self, messages: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
        system, converted = to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": params.get("model", self.model_name),
            "messages": converted,
            "max_tokens": int(params.get("max_tokens") or self.default_max_tokens),  # required by the API
        }
        if system:
            kwargs["system"] = system
        return kwargs

    async def execute(self, messages: list[dict[str, Any]], params: dict[str, Any] | None = None) -> GenerationResponse:
        params = params or {}
        kwargs = self._kwargs(messages, params)
        start = time.perf_counter()
        with self.tracer.start_as_current_span("anthropic.execute") as span:
            span.set_attribute("provider", "anthropic")
            span.set_attribute("model", kwargs["model"])
            try:
                response = await self.client.messages.create(**kwargs)
            except Exception as e:
                self._handle_exception(e)
            content = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
            input_tokens, output_tokens, cached = _usage_numbers(getattr(response, "usage", None))
            report_usage(params, provider="anthropic", model=kwargs["model"], input_tokens=input_tokens,
                         output_tokens=output_tokens, cached_input_tokens=cached)
            self._count(input_tokens or 0, output_tokens or 0)
            return GenerationResponse(
                content=content, provider="anthropic", model=kwargs["model"], prompt_tokens=input_tokens or 0,
                completion_tokens=output_tokens or 0, total_tokens=(input_tokens or 0) + (output_tokens or 0),
                duration_s=time.perf_counter() - start,
            )

    async def stream(self, messages: list[dict[str, Any]], params: dict[str, Any] | None = None) -> AsyncIterator[str]:
        params = params or {}
        kwargs = self._kwargs(messages, params)
        model = kwargs["model"]
        with self.tracer.start_as_current_span("anthropic.stream") as span:
            span.set_attribute("provider", "anthropic")
            span.set_attribute("model", model)
            emitted = False
            rate_limited = False
            while True:
                input_tokens: int | None = None
                output_tokens: int | None = None
                cached = 0
                truncated = False
                try:
                    stream_resp = await self.client.messages.create(stream=True, **kwargs)
                    async for event in stream_resp:
                        etype = getattr(event, "type", "")
                        if etype == "message_start":
                            input_tokens, output_tokens, cached = _usage_numbers(event.message.usage)
                        elif etype == "content_block_delta":
                            text = getattr(event.delta, "text", None)
                            if text and getattr(event.delta, "type", "") == "text_delta":
                                emitted = True
                                yield text
                        elif etype == "message_delta":
                            # the final message_delta carries the cumulative output tokens (and may carry the rest)
                            _, out, _ = _usage_numbers(event.usage)
                            if out is not None:
                                output_tokens = out
                            if getattr(event.delta, "stop_reason", None) == "max_tokens":
                                truncated = True
                    report_usage(params, provider="anthropic", model=model, input_tokens=input_tokens,
                                 output_tokens=output_tokens, cached_input_tokens=cached)
                    self._count(input_tokens or 0, output_tokens or 0)
                    if truncated:
                        self.logger.warning("Anthropic answer hit the output token limit", max_tokens=kwargs["max_tokens"])
                        yield TRUNCATION_NOTICE
                    return
                except Exception as e:
                    # An attempt that reached the provider may have used tokens even when it failed midway.
                    if input_tokens is not None or output_tokens is not None:
                        report_usage(params, provider="anthropic", model=model, input_tokens=input_tokens,
                                     output_tokens=output_tokens, cached_input_tokens=cached)
                    wait = rate_limit_wait_s(e)
                    if wait is not None and not emitted and wait <= MAX_RATE_LIMIT_WAIT_S and not rate_limited:
                        rate_limited = True
                        self.logger.warning("Anthropic rate limit reached; waiting briefly before one retry", wait_s=wait)
                        metrics.PROVIDER_RATE_LIMITED_TOTAL.labels(provider="anthropic", action="retry").inc()
                        await asyncio.sleep(wait + 0.25 + random.uniform(0.0, min(0.25, 0.1 * wait)))
                        continue
                    if wait is not None:
                        metrics.PROVIDER_RATE_LIMITED_TOTAL.labels(provider="anthropic", action="fallback").inc()
                    self._handle_exception(e)

    @staticmethod
    def _count(input_tokens: int, output_tokens: int) -> None:
        try:
            TOKENS_TOTAL.labels(provider="anthropic", token_type="prompt").inc(input_tokens)
            TOKENS_TOTAL.labels(provider="anthropic", token_type="completion").inc(output_tokens)
        except Exception:
            pass

    def _handle_exception(self, e: Exception) -> None:
        err = str(e)
        self.logger.error("Anthropic request failed", error=err[:300])
        name = type(e).__name__
        status = getattr(e, "status_code", None)
        if status == 429 or name == "RateLimitError" or "429" in err or "rate limit" in err.lower():
            raise ProviderRateLimitError(f"Anthropic Rate Limit: {err}", provider="anthropic") from e
        if name in ("APITimeoutError", "TimeoutError") or "timeout" in err.lower() or "timed out" in err.lower():
            raise ProviderTimeoutError(f"Anthropic Timeout: {err}", provider="anthropic") from e
        if status == 529 or name == "OverloadedError" or "overloaded" in err.lower():
            raise ProviderRateLimitError(f"Anthropic overloaded: {err}", provider="anthropic") from e
        raise ProviderError(f"Anthropic Error: {err}", provider="anthropic", status_code=status) from e
