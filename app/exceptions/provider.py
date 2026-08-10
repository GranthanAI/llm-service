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
    """Raised when receiving HTTP 429 from provider."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(
            message, provider=provider, error_code="PROVIDER_RATE_LIMIT", status_code=429
        )


class CircuitOpenError(PermanentError, ProviderError):
    """Raised when attempting a call through an open circuit breaker."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider=provider, error_code="CIRCUIT_OPEN")
