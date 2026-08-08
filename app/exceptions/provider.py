"""
Provider Exceptions.
"""


class ProviderError(Exception):
    """Base exception for LLM provider errors."""

    def __init__(self, message: str, provider: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class AllProvidersFailedError(ProviderError):
    """Raised when both primary (NVIDIA) and fallback (Gemini) providers fail."""

    pass


class ProviderTimeoutError(ProviderError):
    """Raised when a provider API call exceeds timeout."""

    pass


class ProviderRateLimitError(ProviderError):
    """Raised when receiving HTTP 429 from provider."""

    pass


class CircuitOpenError(ProviderError):
    """Raised when attempting a call through an open circuit breaker."""

    pass
