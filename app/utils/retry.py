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
from app.exceptions.grpc import GRPCUnavailableError
from app.exceptions.provider import ProviderRateLimitError, ProviderTimeoutError


class RetryPolicy(BaseModel):
    """Configuration policy for retry backoff and attempts."""

    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_delay_ms: int = Field(default=100, ge=10)
    max_delay_ms: int = Field(default=2000, ge=100)
    backoff_factor: float = 2.0
    jitter: bool = True


def is_retriable(exc: Exception) -> bool:
    """Classifies whether an exception represents a transient failure eligible for retry."""
    if isinstance(
        exc,
        (ProviderRateLimitError, ProviderTimeoutError, GRPCUnavailableError, asyncio.TimeoutError),
    ):
        return True

    # Check for known SDK exception types if imported
    exc_name = type(exc).__name__
    if exc_name in (
        "RateLimitError",
        "InternalServerError",
        "APIConnectionError",
        "ServiceUnavailable",
        "DeadlineExceeded",
    ):
        return True

    # Standard network / socket exceptions
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True

    return False


class RetryManager:
    """
    Executes async operations with exponential backoff and jitter for transient errors.
    Used for pre-streaming provider calls, tool calls, and Kafka publish operations.
    """

    def __init__(
        self, policy: RetryPolicy | None = None, logger: structlog.stdlib.BoundLogger | None = None
    ):
        self.policy: RetryPolicy = policy or RetryPolicy()
        self.logger = logger or get_logger("retry_manager")

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
                if attempt >= self.policy.max_attempts or not is_retriable(exc):
                    self.logger.warning(
                        "Retry exhausted or non-retriable error",
                        operation=operation_name,
                        provider=provider,
                        attempt=attempt,
                        error=str(exc),
                        exc_type=type(exc).__name__,
                    )
                    raise

                # Calculate backoff delay with exponential backoff and jitter
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
