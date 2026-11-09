"""
Billing gRPC client (billing integration plan 4.5): CheckAndReserve / Release for Manim renders.

Caller rules: CheckAndReserve deadline 150 ms; Release 500 ms x 3 retries; a circuit breaker that opens after 5
consecutive failures or timeouts and half-opens after 30 s; metadata `authorization: Bearer <service token>` and
`x-correlation-id`. "Unreachable" (breaker open, timeout, any transport error) is reported as None so the caller can
apply its degraded behaviour; it is never raised.
"""

import asyncio
from dataclasses import dataclass

import grpc

from app.config.logging import get_logger
from app.grpc.proto import billing_pb2, billing_pb2_grpc
from app.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

logger = get_logger("billing_client")

RESERVE_DEADLINE_S = 0.150
RELEASE_DEADLINE_S = 0.500
RELEASE_ATTEMPTS = 3

DENY_NAMES = {
    billing_pb2.DENY_REASON_DAILY_LIMIT: "daily_limit",
    billing_pb2.DENY_REASON_MONTHLY_LIMIT: "monthly_limit",
    billing_pb2.DENY_REASON_MODEL_NOT_IN_PLAN: "model_not_in_plan",
    billing_pb2.DENY_REASON_NO_CREDIT: "no_credit",
    billing_pb2.DENY_REASON_SPEND_CAP: "spend_cap",
    billing_pb2.DENY_REASON_ACCOUNT_BLOCKED: "account_blocked",
    billing_pb2.DENY_REASON_MEMBER_SPEND_CAP: "member_spend_cap",
}


@dataclass(frozen=True)
class ReserveResult:
    allowed: bool
    hold_id: str | None = None
    deny_reason: str | None = None         # daily_limit | monthly_limit | ... (plan 4.6)
    message_key: str | None = None


class BillingClient:
    def __init__(self, target: str, token: str, *, channel: grpc.aio.Channel | None = None, stub=None,
                 breaker: CircuitBreaker | None = None):
        self.target = target
        self._token = token
        self._channel = channel
        self._stub = stub
        self.breaker = breaker or CircuitBreaker(
            "billing_grpc", CircuitBreakerConfig(failure_threshold=5, success_threshold=1, recovery_timeout_seconds=30))

    def _get_stub(self):
        if self._stub is None:
            self._channel = self._channel or grpc.aio.insecure_channel(self.target)
            self._stub = billing_pb2_grpc.BillingInternalStub(self._channel)
        return self._stub

    def _metadata(self, correlation_id: str | None) -> tuple[tuple[str, str], ...]:
        md = [("authorization", f"Bearer {self._token}")]
        if correlation_id:
            md.append(("x-correlation-id", correlation_id))
        return tuple(md)

    async def check_and_reserve_manim(self, user_id: str, idempotency_key: str,
                                      correlation_id: str | None = None) -> ReserveResult | None:
        """The decision, or None when billing cannot be reached (breaker open / timeout / error)."""
        if self.breaker.is_open():
            logger.warning("billing_breaker_open", op="CheckAndReserve")
            return None
        request = billing_pb2.CheckAndReserveRequest(
            user_id=user_id, idempotency_key=idempotency_key, kind=billing_pb2.RESERVE_KIND_MANIM_VIDEO)
        try:
            resp = await self._get_stub().CheckAndReserve(
                request, timeout=RESERVE_DEADLINE_S, metadata=self._metadata(correlation_id))
        except Exception as exc:                       # DEADLINE_EXCEEDED, UNAVAILABLE, anything else
            await self.breaker.on_failure()
            logger.warning("billing_reserve_unreachable", error=type(exc).__name__)
            return None
        await self.breaker.on_success()
        if resp.allowed:
            return ReserveResult(True, hold_id=resp.hold_id or None)
        return ReserveResult(False, deny_reason=DENY_NAMES.get(resp.deny_reason, "account_blocked"),
                             message_key=resp.message_key or None)

    async def release(self, user_id: str, hold_id: str, reason: str, correlation_id: str | None = None) -> bool:
        """Give a hold back (idempotent in billing). 500 ms deadline, 3 attempts; False if it could not be done."""
        request = billing_pb2.ReleaseRequest(user_id=user_id, hold_id=hold_id, reason=reason)
        for attempt in range(1, RELEASE_ATTEMPTS + 1):
            if self.breaker.is_open():
                break
            try:
                resp = await self._get_stub().Release(
                    request, timeout=RELEASE_DEADLINE_S, metadata=self._metadata(correlation_id))
                await self.breaker.on_success()
                return bool(resp.released)
            except Exception as exc:
                await self.breaker.on_failure()
                logger.warning("billing_release_failed", attempt=attempt, error=type(exc).__name__)
                await asyncio.sleep(0.05 * attempt)
        return False

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
