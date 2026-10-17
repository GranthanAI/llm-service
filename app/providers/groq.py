"""
Groq Provider Adapter.
Implements LLD v2.0 Section 20.2.
"""

import asyncio
import re
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
from app.providers.base import TRUNCATION_NOTICE, BaseProviderAdapter
from app.providers.models import GenerationResponse
from app.utils.metrics import TOKENS_TOTAL
from app.utils.tracing import get_tracer

logger = get_logger("groq_adapter")


# Groq's gpt-oss models sometimes try to call a built-in tool (browsing, code) even though none were
# offered; the API then aborts the stream with this message. It is a model quirk, not a bad request, and
# it happens before any text is produced, so the call can be retried safely.
_NO_TOOLS_NUDGE = (
    "\n\nAnswer in plain text using only the information in the conversation. "
    "Do not call, request or mention any tools or functions."
)
_MAX_TOOL_CALL_RETRIES = 2
GENERATION_MAX_TOKENS = 4096
# A 429 says how long to wait ("Please try again in 16.4s"). Short waits are worth honouring once; long
# ones are reported straight away so the user is told instead of watching a stalled spinner.
MAX_RATE_LIMIT_WAIT_S = 8.0
_RETRY_AFTER = re.compile(r"try again in (\d+(?:\.\d+)?)\s*(ms|s)\b", re.IGNORECASE)


def rate_limit_wait_s(exc: Exception) -> float | None:
    """Seconds Groq asks us to wait if `exc` is a rate-limit error, else None."""
    msg = str(exc)
    if "429" not in msg and "rate limit" not in msg.lower():
        return None
    m = _RETRY_AFTER.search(msg)
    if not m:
        return 60.0  # unknown: treat as long
    value = float(m.group(1))
    return value / 1000.0 if m.group(2).lower() == "ms" else value
MIN_OUTPUT_TOKENS = 1024
_CHARS_PER_TOKEN = 3.0  # deliberately pessimistic so the estimate over-counts the prompt


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    return int(sum(len(str(m.get("content", ""))) for m in messages) / _CHARS_PER_TOKEN) + 50


def output_budget(requested: int, prompt_tokens: int, tpm_limit: int) -> int:
    """Largest output budget that still fits Groq's per-request tokens-per-minute limit.

    Groq counts prompt + max_tokens against the limit, so asking for more than fits means a guaranteed
    429 with no answer at all. The caller's value is a floor (gpt-oss spends part of the budget on
    reasoning), capped by the per-request ceiling and by what the limit allows for this prompt."""
    want = min(max(requested, GENERATION_MAX_TOKENS), 8192)
    if tpm_limit <= 0:
        return want
    return max(MIN_OUTPUT_TOKENS, min(want, tpm_limit - prompt_tokens - 200))


def _is_spurious_tool_call(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "tool choice is none" in msg or "model called a tool" in msg


def _with_nudge(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of `messages` with a no-tools instruction appended to the system message (or added)."""
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] = m["content"] + _NO_TOOLS_NUDGE
            return out
    return [{"role": "system", "content": _NO_TOOLS_NUDGE.strip()}, *out]


class GroqAdapter(BaseProviderAdapter):
    """
    Groq adapter used primarily for high-speed Request Analysis (Call 1).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout_s: float = 5.0,
        tpm_limit: int = 0,
        client: AsyncGroq | None = None,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        super().__init__(provider_name="groq", model_name=model)
        self.api_key = api_key
        self.timeout_s = timeout_s
        # Per-request tokens-per-minute ceiling of the account's Groq tier (0 = unlimited).
        self.tpm_limit = tpm_limit
        # max_retries=0: the SDK would silently back off and retry 429s, hiding a stall from the router's
        # deadlines. Rate limits are handled explicitly in `stream`.
        self.client = client or AsyncGroq(
            api_key=api_key or "not-set",
            timeout=timeout_s,
            max_retries=0,
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
        # gpt-oss counts its reasoning tokens inside the same budget as the visible answer, so the limit
        # passed down for another provider is a floor, never a ceiling, for this adapter.
        max_tokens = output_budget(
            int(params.get("max_tokens", 0) or 0), estimate_prompt_tokens(messages), self.tpm_limit
        )
        extra: dict[str, Any] = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}

        with self.tracer.start_as_current_span("groq.stream") as span:
            span.set_attribute("provider", "groq")
            span.set_attribute("model", model)

            emitted = False
            rate_limited = False
            attempt = 0
            current = messages
            while True:
                try:
                    stream_resp = await self.client.chat.completions.create(
                        model=model,
                        messages=current,  # type: ignore[arg-type]
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        **extra,
                    )
                    truncated = False
                    async for chunk in stream_resp:
                        if chunk.choices and chunk.choices[0].delta:
                            content_piece = chunk.choices[0].delta.content
                            if content_piece:
                                emitted = True
                                yield content_piece
                        if chunk.choices and chunk.choices[0].finish_reason == "length":
                            truncated = True
                    if truncated:
                        self.logger.warning("Groq answer hit the output token limit", max_tokens=max_tokens)
                        yield TRUNCATION_NOTICE
                    return
                except Exception as e:
                    wait = rate_limit_wait_s(e)
                    if wait is not None and not emitted and wait <= MAX_RATE_LIMIT_WAIT_S and not rate_limited:
                        rate_limited = True
                        self.logger.warning("Groq rate limit reached; waiting briefly before one retry", wait_s=wait)
                        await asyncio.sleep(wait + 0.25)
                        continue
                    # Retry only when nothing has been shown to the user yet (never splice two answers).
                    if _is_spurious_tool_call(e) and not emitted and attempt < _MAX_TOOL_CALL_RETRIES:
                        attempt += 1
                        self.logger.warning(
                            "Groq model attempted a tool call; retrying with a no-tools instruction",
                            attempt=attempt,
                        )
                        current = _with_nudge(messages)
                        continue
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
