"""
gRPC Client Exceptions.
"""


class GRPCError(Exception):
    """Base exception for baseline context gRPC service errors."""

    def __init__(self, message: str, service: str | None = None, code: int | None = None):
        super().__init__(message)
        self.service = service
        self.code = code


class GRPCUnavailableError(GRPCError):
    """Raised when gRPC status is UNAVAILABLE."""

    pass


class GRPCTimeoutError(GRPCError):
    """Raised when gRPC call exceeds deadline."""

    pass
