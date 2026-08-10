"""
Base Service Exception Hierarchy and Taxonomy.
Implements LLD v2.0 Section 28.1 (Error Taxonomy: Retriable, Permanent, Fatal).
"""


class BaseLLMServiceError(Exception):
    """Root base exception for all LLM service-defined exceptions."""

    def __init__(self, message: str, error_code: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.message: str = message
        self.error_code: str = error_code or self.__class__.__name__
        self.details: dict = details or {}

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class RetriableError(BaseLLMServiceError):
    """
    Marker / base class for transient errors that can be safely retried.
    Examples: network timeouts, HTTP 429/503, gRPC UNAVAILABLE.
    """

    pass


class PermanentError(BaseLLMServiceError):
    """
    Marker / base class for deterministic errors that will not succeed on retry.
    Examples: Pydantic validation failure, HTTP 400/401/403, context overflow.
    """

    pass


class FatalError(BaseLLMServiceError):
    """
    Marker / base class for fatal system-level errors requiring service halt.
    Examples: missing startup configuration, unrecoverable boot failure.
    """

    pass
