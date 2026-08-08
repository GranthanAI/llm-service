"""
Groq Analysis Async Client for Request Planning (Call 1).
Implements LLD v2.0 Section 10.1 and HLD v2.0 Section 9.
"""

from typing import Any

import groq
import structlog
from groq import AsyncGroq

from app.config.logging import get_logger
from app.exceptions.provider import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)


class GroqAnalysisClient:
    """
    Dedicated client for ultra-low-latency planning inference via Groq.
    Forces JSON mode and deterministic temperature for structured ExecutionPlan generation.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout_ms: int = 3000,
        client: AsyncGroq | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.api_key: str = api_key
        self.model: str = model
        self.timeout_ms: int = timeout_ms
        self._client: AsyncGroq = client or AsyncGroq(api_key=api_key)
        self.logger = logger or get_logger("groq_analysis_client")

    async def complete(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.1,
    ) -> str:
        """Execute chat completion in JSON mode against Groq."""
        fmt = response_format or {"type": "json_object"}
        timeout_sec = self.timeout_ms / 1000.0

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                response_format=fmt,
                timeout=timeout_sec,
            )

            content = response.choices[0].message.content
            if not content:
                raise ProviderError("Groq returned empty response content", provider="groq")

            return content
        except groq.RateLimitError as exc:
            self.logger.warning("Groq rate limit exceeded", error=str(exc))
            raise ProviderRateLimitError(
                "Groq rate limit exceeded", provider="groq", status_code=429
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
