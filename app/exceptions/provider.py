"""
Provider Exceptions.
Implements LLD v2.0 Section 28.1.
"""

from app.exceptions.base import BaseLLMServiceError, PermanentError, RetriableError


class ProviderError(BaseLLMServiceError):
    """Base exception for LLM provider errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        status_code: int | None = None,
        error_code: str | None = None,
    ):
        super().__init__(message, error_code=error_code or "PROVIDER_ERROR")
        self.provider = provider
        self.status_code = status_code


class AllProvidersFailedError(PermanentError, ProviderError):
    """Raised when both primary (NVIDIA) and fallback (Gemini) providers fail."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider=provider, error_code="PROVIDER_ALL_FAILED")


class ProviderTimeoutError(RetriableError, ProviderError):
    """Raised when a provider API call exceeds timeout."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider=provider, error_code="PROVIDER_TIMEOUT")


class ProviderRateLimitError(RetriableError, ProviderError):
    """Raised when receiving HTTP 429 (or an overload) from a provider.

    `retry_after_seconds` is what the provider asked for (Retry-After / x-ratelimit-reset-*), when it said. The status code is
    always 429, so callers never pass it."""

    def __init__(
        self, message: str, provider: str | None = None, retry_after_seconds: float | None = None
    ):
        super().__init__(
            message, provider=provider, error_code="PROVIDER_RATE_LIMIT", status_code=429
        )
        self.retry_after_seconds = retry_after_seconds
        try:  # every 429 is counted where it is raised, whichever provider or wrapper raised it
            from app.utils.metrics import PROVIDER_RATE_LIMITED_TOTAL

            PROVIDER_RATE_LIMITED_TOTAL.labels(provider=provider or "unknown", action="raised").inc()
        except Exception:  # noqa: BLE001, S110 - metrics must never turn a 429 into another error
            pass


class CircuitOpenError(PermanentError, ProviderError):
    """Raised when attempting a call through an open circuit breaker."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider=provider, error_code="CIRCUIT_OPEN")
