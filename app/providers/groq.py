"""
Groq Provider Adapter.
Implements LLD v2.0 Section 20.2.
"""

import asyncio
import random
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from groq import AsyncGroq

from app.config.logging import get_logger
from app.utils.rate_limit import retry_after_of
from app.exceptions.provider import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.providers.base import TRUNCATION_NOTICE, BaseProviderAdapter
from app.providers.models import GenerationResponse
from app.providers.usage_hook import report_usage
from app.utils import metrics
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


def _retry_after_header_s(exc: Exception) -> float | None:
    """Seconds from a `Retry-After` response header (numeric form), when the SDK error carries the response."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        raw = headers.get("retry-after") if headers is not None else None
        return max(0.0, float(raw)) if raw is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def is_context_limit_error(exc: Exception) -> bool:
    """A request rejected for its SIZE (context window), which is not a rate limit and must not be retried as one."""
    msg = str(exc).lower()
    return "context_length_exceeded" in msg or "maximum context length" in msg or "reduce the length of the messages" in msg


def rate_limit_wait_s(exc: Exception) -> float | None:
    """Seconds Groq asks us to wait if `exc` is a rate-limit error, else None (a context-size rejection is None)."""
    msg = str(exc)
    if is_context_limit_error(exc):
        return None
    if "429" not in msg and "rate limit" not in msg.lower():
        return None
    header = _retry_after_header_s(exc)
    if header is not None:
        return header
    m = _RETRY_AFTER.search(msg)
    if not m:
        return 60.0  # unknown: treat as long
    value = float(m.group(1))
    return value / 1000.0 if m.group(2).lower() == "ms" else value
MIN_OUTPUT_TOKENS = 1024
_CHARS_PER_TOKEN = 3.0  # deliberately pessimistic so the estimate over-counts the prompt


_COUNTER = None


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    """Prompt size for the tokens-per-minute arithmetic: a real tokenizer count (+5%) when available, else the
    pessimistic characters/3 fallback. (Characters/3 alone over-counts by ~30% and shrank answers needlessly.)"""
    global _COUNTER
    try:
        if _COUNTER is None:
            from app.context_window.token_counter import TokenCounter

            _COUNTER = TokenCounter()
        if _COUNTER.tokenizer is not None:
            return int(_COUNTER.count_messages(messages, "groq") * 1.05) + 50
    except Exception:
        pass
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


def _groq_cached(usage: Any) -> int:
    """Cached prompt tokens when Groq reports them (prompt_tokens_details.cached_tokens), else 0."""
    details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
    return int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0


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
        budget: Any = None,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        super().__init__(provider_name="groq", model_name=model)
        self.api_key = api_key
        self.timeout_s = timeout_s
        # Per-request tokens-per-minute ceiling of the account's Groq tier (0 = unlimited).
        self.tpm_limit = tpm_limit
        self.budget = budget  # shared per-model token budget (groq_budget.GroqBudget) or None
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
                report_usage(
                    params, provider="groq", model=model,
                    input_tokens=response.usage.prompt_tokens if response.usage else None,
                    output_tokens=response.usage.completion_tokens if response.usage else None,
                    cached_input_tokens=_groq_cached(response.usage), messages=messages, output_text=content)

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

    def _estimate(self, messages: list[dict[str, Any]], params: dict[str, Any]) -> int:
        """Tokens this call will count against the minute: the prompt plus the output it asks for."""
        max_tokens = output_budget(int(params.get("max_tokens", 0) or 0), estimate_prompt_tokens(messages), self.tpm_limit)
        from app.providers.groq_budget import estimate_tokens

        return estimate_tokens(estimate_prompt_tokens(messages), max_tokens, getattr(self.budget, "output_estimate", 600))

    async def budget_wait(self, messages: list[dict[str, Any]], params: dict[str, Any] | None = None) -> float:
        """Seconds until this answer fits this model's shared budget (0.0 = it fits and is claimed). The chain calls this
        first so it can move to the next model instead of waiting."""
        params = params if params is not None else {}
        if self.budget is None:
            return 0.0
        estimate = self._estimate(messages, params)   # input + min(max_tokens, 600): Groq does not count the full max_tokens
        wait = await self.budget.acquire(self.model_name, estimate, "answer")
        if wait <= 0:
            params["_budget_claimed"] = self.model_name
            params["_budget_estimate"] = estimate
        return wait

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
        claimed = self.budget is None or params.get("_budget_claimed") == model
        if not claimed:  # called outside the chain (legacy order): claim here, wait only for a short gap
            params["_budget_estimate"] = self._estimate(messages, params)
            gap = await self.budget.acquire(model, params["_budget_estimate"], "answer")
            if gap > MAX_RATE_LIMIT_WAIT_S:
                raise ProviderRateLimitError(f"Groq budget for {model} is short for {gap:.0f}s", provider="groq", retry_after_seconds=gap)
            if gap > 0:
                await asyncio.sleep(gap + 0.1)

        with self.tracer.start_as_current_span("groq.stream") as span:
            span.set_attribute("provider", "groq")
            span.set_attribute("model", model)

            emitted = False
            rate_limited = False
            attempt = 0
            current = messages
            attempt_text: list[str] = []
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
                    usage = None
                    if self.budget is not None:  # the real remaining tokens, for every other caller of this model
                        await self.budget.record_headers(model, getattr(getattr(stream_resp, "response", None), "headers", None))
                    attempt_text.clear()
                    async for chunk in stream_resp:
                        # usage arrives on the final chunk: `usage`, or `x_groq.usage`
                        usage = getattr(chunk, "usage", None) or getattr(getattr(chunk, "x_groq", None), "usage", None) or usage
                        if chunk.choices and chunk.choices[0].delta:
                            content_piece = chunk.choices[0].delta.content
                            if content_piece:
                                emitted = True
                                attempt_text.append(content_piece)
                                yield content_piece
                        if chunk.choices and chunk.choices[0].finish_reason == "length":
                            truncated = True
                    report_usage(
                        params, provider="groq", model=model,
                        input_tokens=getattr(usage, "prompt_tokens", None), output_tokens=getattr(usage, "completion_tokens", None),
                        cached_input_tokens=_groq_cached(usage), messages=current, output_text="".join(attempt_text))
                    if self.budget is not None and params.get("_budget_estimate"):
                        # the call is over: the minute was charged for what it really used, not for prompt + max_tokens
                        claimed = int(params["_budget_estimate"])
                        used = int(getattr(usage, "total_tokens", 0) or 0) or claimed
                        await self.budget.settle_window(model, claimed, used)
                        await self.budget.settle(model, claimed, used)
                    if truncated:
                        self.logger.warning("Groq answer hit the output token limit", max_tokens=max_tokens)
                        yield TRUNCATION_NOTICE
                    return
                except Exception as e:
                    if self.budget is not None:
                        await self.budget.record_error(model, e)
                    if attempt_text:
                        # an answer that died midway still consumed tokens the provider will bill: estimate them
                        report_usage(params, provider="groq", model=model, input_tokens=None, output_tokens=None,
                                     messages=current, output_text="".join(attempt_text))
                    wait = rate_limit_wait_s(e)
                    if wait is not None and not emitted and wait <= MAX_RATE_LIMIT_WAIT_S and not rate_limited:
                        rate_limited = True
                        self.logger.warning("Groq rate limit reached; waiting briefly before one retry", wait_s=wait)
                        metrics.PROVIDER_RATE_LIMITED_TOTAL.labels(provider="groq", action="retry").inc()
                        # bounded, jittered backoff so concurrent requests do not retry in lock step
                        await asyncio.sleep(wait + 0.25 + random.uniform(0.0, min(0.25, 0.1 * wait)))
                        continue
                    if wait is not None:
                        metrics.PROVIDER_RATE_LIMITED_TOTAL.labels(provider="groq", action="fallback").inc()
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
        if is_context_limit_error(e):
            raise ProviderError(f"Groq rejected the request size (context limit): {err_msg[:200]}", provider="groq") from e
        if "429" in err_msg or "rate limit" in err_msg.lower():
            raise ProviderRateLimitError(f"Groq Rate Limit: {err_msg}", provider="groq", retry_after_seconds=retry_after_of(e)) from e
        if "timeout" in err_msg.lower() or "timed out" in err_msg.lower():
            raise ProviderTimeoutError(f"Groq Timeout: {err_msg}", provider="groq") from e
        raise ProviderError(f"Groq Error: {err_msg}", provider="groq") from e
