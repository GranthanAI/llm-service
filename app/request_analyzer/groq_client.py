"""
Groq Analysis Async Client for Request Planning (Call 1).
Implements LLD v2.0 Section 10.1 and HLD v2.0 Section 9.
"""

from typing import Any

import groq
import structlog
from groq import AsyncGroq

from app.config.logging import get_logger
from app.providers.usage_hook import report_usage
from app.utils.rate_limit import retry_after_of
from app.exceptions.provider import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)


class AnalysisBudgetSkipped(Exception):
    """The model's Groq budget cannot take an analysis call without eating the answer reserve: skip analysis."""


class GroqAnalysisClient:
    """
    Dedicated client for ultra-low-latency planning inference via Groq.
    Forces JSON mode and deterministic temperature for structured ExecutionPlan generation.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-20b",
        timeout_ms: int = 3000,
        client: AsyncGroq | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
        budget: Any = None,
        max_tokens: int = 512,
    ):
        self.budget = budget
        self.max_tokens = max_tokens
        self.api_key: str = api_key
        self.model: str = model
        self.timeout_ms: int = timeout_ms
        self._client: AsyncGroq = client or AsyncGroq(api_key=api_key)
        self.logger = logger or get_logger("groq_analysis_client")

    async def reserve(self, messages: list[dict[str, str]]) -> int:
        """Claim this call's tokens from the shared budget BEFORE it runs (outside the circuit breaker, so a skip is not a
        failure). Raises AnalysisBudgetSkipped when the call would eat the answer reserve; returns the claimed estimate."""
        if self.budget is None:
            return 0
        from app.providers.groq import estimate_prompt_tokens

        from app.providers.groq_budget import estimate_tokens

        estimate = estimate_tokens(estimate_prompt_tokens(messages), self.max_tokens, getattr(self.budget, "output_estimate", 600))
        wait = await self.budget.acquire(self.model, estimate, "analysis")
        if wait > 0:
            raise AnalysisBudgetSkipped(f"{self.model} budget short for {wait:.0f}s")
        return estimate

    async def complete(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.1,
        estimate: int | None = None,
    ) -> str:
        """Execute chat completion in JSON mode against Groq."""
        fmt = response_format or {"type": "json_object"}
        timeout_sec = self.timeout_ms / 1000.0
        if estimate is None:  # called without reserve(): claim here
            estimate = await self.reserve(messages)

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                response_format=fmt,
                max_tokens=self.max_tokens,
                timeout=timeout_sec,
            )
            if self.budget is not None and getattr(response, "usage", None) is not None:
                await self.budget.settle(self.model, estimate, int(getattr(response.usage, "total_tokens", estimate) or estimate))

            usage = getattr(response, "usage", None)
            report_usage(
                None, provider="groq", model=self.model,
                input_tokens=getattr(usage, "prompt_tokens", None), output_tokens=getattr(usage, "completion_tokens", None),
                messages=messages, output_text=response.choices[0].message.content or "")
            content = response.choices[0].message.content
            if not content:
                raise ProviderError("Groq returned empty response content", provider="groq")

            return content
        except groq.RateLimitError as exc:
            if self.budget is not None:
                await self.budget.record_error(self.model, exc)
            self.logger.warning("Groq rate limit exceeded", error=str(exc))
            raise ProviderRateLimitError(
                "Groq rate limit exceeded", provider="groq", retry_after_seconds=retry_after_of(exc)
            ) from exc
        except (groq.APITimeoutError, TimeoutError) as exc:
            self.logger.warning(
                "Groq request timed out", timeout_ms=self.timeout_ms, error=str(exc)
            )
            raise ProviderTimeoutError(
                f"Groq request timed out after {self.timeout_ms}ms", provider="groq"
            ) from exc
        except groq.APIError as exc:
            self.logger.error(
                "Groq API error", error=str(exc), status_code=getattr(exc, "status_code", None)
            )
            raise ProviderError(
                f"Groq API error: {exc}",
                provider="groq",
                status_code=getattr(exc, "status_code", None),
            ) from exc
        except Exception as exc:
            self.logger.error("Unexpected error in Groq client", error=str(exc))
            raise ProviderError(f"Unexpected Groq client failure: {exc}", provider="groq") from exc
