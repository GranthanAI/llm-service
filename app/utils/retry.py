"""
Retry Manager and Transient Failure Classification.
Implements LLD v2.0 Section 22.
"""

import asyncio
import random
from collections.abc import Callable
from typing import Any

import structlog
from pydantic import BaseModel, Field

from app.config.logging import get_logger
from app.exceptions.base import RetriableError
from app.exceptions.grpc import GRPCUnavailableError
from app.exceptions.provider import ProviderRateLimitError, ProviderTimeoutError


class RetryPolicy(BaseModel):
    """Configuration policy for retry backoff, attempts, and jitter."""

    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_delay_ms: int = Field(default=100, ge=10)
    max_delay_ms: int = Field(default=2000, ge=10)
    backoff_factor: float = 2.0
    jitter: bool = True
    retriable_status_codes: list[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])
    retriable_exceptions: list[type[Exception]] = Field(default_factory=list)


def is_retriable(exc: Exception, policy: RetryPolicy | None = None) -> bool:
    """
    Classifies whether an exception represents a transient failure eligible for retry.
    Implements LLD v2.0 Section 22.3.
    """
    # 1. Direct RetriableError marker inheritance
    if isinstance(exc, RetriableError):
        return True

    # 2. Known domain exception classes
    if isinstance(
        exc,
        (
            ProviderRateLimitError,
            ProviderTimeoutError,
            GRPCUnavailableError,
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
        ),
    ):
        return True

    # 3. Status code check if present on exception
    if policy and hasattr(exc, "status_code") and exc.status_code is not None:
        if exc.status_code in policy.retriable_status_codes:
            return True

    # 4. User-configured custom retriable exception types
    if policy and policy.retriable_exceptions:
        if isinstance(exc, tuple(policy.retriable_exceptions)):
            return True

    # 5. Check for known SDK / library exception class names
    exc_name = type(exc).__name__
    if exc_name in (
        "RateLimitError",
        "InternalServerError",
        "APIConnectionError",
        "ServiceUnavailable",
        "DeadlineExceeded",
        "TooManyRequests",
        "AioRpcError",
    ):
        # If gRPC AioRpcError, inspect status code if available
        if hasattr(exc, "code") and callable(exc.code):
            try:
                code_name = exc.code().name
                if code_name in ("UNAVAILABLE", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED"):
                    return True
            except Exception:
                pass
        return True

    # 6. Standard OS network / socket exceptions
    if isinstance(exc, (OSError,)):
        return True

    return False  # Authentication, BadRequest, Pydantic ValidationError: do NOT retry


class RetryManager:
    """
    Executes async operations with exponential backoff and jitter for transient errors.
    Used for pre-streaming provider calls, tool calls, and Kafka publish operations.
    Retries NEVER cross the streaming boundary (LLD Section 22.4).
    """

    def __init__(
        self,
        policy: RetryPolicy | None = None,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        self.policy: RetryPolicy = policy or RetryPolicy()
        self.logger = logger_instance or get_logger("retry_manager")

    async def execute_with_retry(
        self,
        func: Callable[[], Any],
        operation_name: str = "operation",
        provider: str | None = None,
    ) -> Any:
        """
        Execute an async callable with retries based on policy.
        Retries never cross streaming boundaries.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func()
                res = func()
                if asyncio.iscoroutine(res):
                    return await res
                return res
            except Exception as exc:
                if attempt >= self.policy.max_attempts or not is_retriable(exc, self.policy):
                    self.logger.warning(
                        "Retry exhausted or non-retriable error",
                        operation=operation_name,
                        provider=provider,
                        attempt=attempt,
                        error=str(exc),
                        exc_type=type(exc).__name__,
                    )
                    raise

                # Calculate backoff delay with exponential backoff and ±10% jitter (LLD Section 22.5)
                base_delay = min(
                    (self.policy.initial_delay_ms / 1000.0)
                    * (self.policy.backoff_factor ** (attempt - 1)),
                    self.policy.max_delay_ms / 1000.0,
                )
                jitter_val = (
                    random.uniform(-0.1 * base_delay, 0.1 * base_delay)
                    if self.policy.jitter
                    else 0.0
                )
                delay = max(0.01, base_delay + jitter_val)

                self.logger.info(
                    "Retrying operation after transient failure",
                    operation=operation_name,
                    provider=provider,
                    attempt=attempt,
                    delay_seconds=round(delay, 3),
                    error=str(exc),
                )
                await asyncio.sleep(delay)
