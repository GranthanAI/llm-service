"""
Circuit Breaker Implementation.
Implements LLD v2.0 Section 23 and HLD v2.0 Section 19.3.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from pydantic import BaseModel, Field

from app.config.logging import get_logger
from app.exceptions.provider import CircuitOpenError
from app.models.provider import CircuitState
from app.utils.metrics import CIRCUIT_BREAKER_STATE
from app.utils.retry import is_retriable


class CircuitBreakerConfig(BaseModel):
    """Configuration parameters for CircuitBreaker."""

    failure_threshold: int = Field(default=5, ge=1)
    success_threshold: int = Field(default=2, ge=1)
    window_seconds: int = Field(default=60, ge=1)
    recovery_timeout_seconds: int = Field(default=30, ge=1)


class CircuitBreaker:
    """
    Per-dependency Circuit Breaker managing CLOSED, OPEN, and HALF_OPEN state transitions.
    Protects downstream dependencies from cascading failures.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.name: str = name
        self.config: CircuitBreakerConfig = config or CircuitBreakerConfig()
        self.logger = logger or get_logger(f"circuit_breaker_{name}")

        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.success_count: int = 0
        self.last_failure_time: float = 0.0
        self._lock: asyncio.Lock = asyncio.Lock()

    def is_open(self) -> bool:
        """Check if circuit is currently open and not yet ready for half-open probe."""
        if self.state == CircuitState.OPEN:
            now = time.time()
            if (now - self.last_failure_time) < self.config.recovery_timeout_seconds:
                return True
        return False

    def trip(self) -> None:
        """Manually trip circuit breaker to OPEN state."""
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
        try:
            CIRCUIT_BREAKER_STATE.labels(name=self.name).set(2)
        except Exception:
            pass
        self.logger.warning("Circuit breaker tripped to OPEN", name=self.name)

    async def on_success(self) -> None:
        """Handle successful call and transition state if applicable."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    CIRCUIT_BREAKER_STATE.labels(name=self.name).set(0)
                    self.logger.info("Circuit breaker closed — service recovered", name=self.name)
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)

    async def on_failure(self) -> None:
        """Handle failed call and trip circuit if threshold exceeded."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                CIRCUIT_BREAKER_STATE.labels(name=self.name).set(2)
                self.logger.warning("Circuit reopened during half-open probe", name=self.name)
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                CIRCUIT_BREAKER_STATE.labels(name=self.name).set(2)
                self.logger.warning(
                    "Circuit breaker opened due to failure threshold",
                    name=self.name,
                    failure_count=self.failure_count,
                    threshold=self.config.failure_threshold,
                )

    @asynccontextmanager
    async def call(self) -> AsyncIterator[None]:
        """Context manager protecting async callable with circuit breaker."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                now = time.time()
                if (now - self.last_failure_time) >= self.config.recovery_timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    CIRCUIT_BREAKER_STATE.labels(name=self.name).set(1)
                    self.logger.info("Circuit entered half-open probe state", name=self.name)
                else:
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is OPEN", provider=self.name
                    )

        try:
            yield
            await self.on_success()
        except Exception as exc:
            if is_retriable(exc) or isinstance(exc, (TimeoutError, ConnectionError)):
                await self.on_failure()
            raise
