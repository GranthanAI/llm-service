"""
gRPC Client Exceptions.
Implements LLD v2.0 Section 28.1.
"""

from app.exceptions.base import BaseLLMServiceError, RetriableError


class GRPCError(BaseLLMServiceError):
    """Base exception for baseline context gRPC service errors."""

    def __init__(
        self,
        message: str,
        service: str | None = None,
        code: int | None = None,
        error_code: str | None = None,
    ):
        super().__init__(message, error_code=error_code or "GRPC_ERROR")
        self.service = service
        self.code = code


class GRPCUnavailableError(RetriableError, GRPCError):
    """Raised when gRPC status is UNAVAILABLE."""

    def __init__(self, message: str, service: str | None = None, code: int | None = None):
        super().__init__(message, service=service, code=code, error_code="GRPC_UNAVAILABLE")


class GRPCTimeoutError(RetriableError, GRPCError):
    """Raised when gRPC call exceeds deadline."""

    def __init__(self, message: str, service: str | None = None, code: int | None = None):
        super().__init__(message, service=service, code=code, error_code="GRPC_TIMEOUT")
