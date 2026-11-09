"""
Manim x billing: reserve before a render is queued, publish artifact.manim.rendered when the job ends (plan 4.4.3, 4.6).

    BILLING_RESERVE_MANIM=off      no reserve call (the event is still published, with hold_id null)
                      shadow       call billing, log billing_shadow_decision, always proceed
                      enforce      a deny stops the request before anything is enqueued (429 / 403 / 402)
    Billing unreachable (breaker open or timeout): proceed with hold_id null (renders are unlimited on plans).

`attempt` is the billing attempt of a job id: 1 for a new job, +1 each time a failed/cancelled/expired job is submitted
again. (The render attempts inside one job, repairs included, belong to the same billing attempt.)
"""

from contextvars import ContextVar
from typing import Any

from fastapi import HTTPException

from app.config.logging import get_logger
from app.usage.collector import UsageCollector, usage_purpose, use_collector

logger = get_logger("manim_billing")

# (job_id, billing_attempt) of the job being run, so repair calls are attributed to it
CURRENT_JOB: ContextVar[tuple[str, int] | None] = ContextVar("manim_current_job", default=None)

# plan 4.6: reason -> (HTTP status, default message_key, English message)
DENY_RESPONSES = {
    "daily_limit": (429, "limit.daily", "Daily limit reached. Resets at midnight, or upgrade / top up."),
    "monthly_limit": (429, "limit.monthly", "You've used this month's allowance. It resets on the 1st, or upgrade."),
    "model_not_in_plan": (403, "plan.model_not_included", "This model isn't included in your plan."),
    "account_blocked": (403, "account.blocked", "Your account can't send messages right now. Please contact support."),
    "no_credit": (402, "credits.empty", "Top up to continue."),
    "spend_cap": (402, "credits.spend_cap",
                  "Monthly spend cap reached. Sending is paused until the cap is raised or next month."),
    "member_spend_cap": (402, "credits.member_cap",
                         "Your organisation's monthly limit for you has been reached. Ask your organisation admin to raise it."),
}


def deny_error(reason: str, message_key: str | None) -> HTTPException:
    status, default_key, message = DENY_RESPONSES.get(reason, DENY_RESPONSES["account_blocked"])
    return HTTPException(status_code=status, detail={
        "code": "billing_denied", "reason": reason, "message_key": message_key or default_key, "message": message,
        "resets_at": None, "upgrade_to": None})


class ManimBilling:
    def __init__(self, mode: str, client: Any | None, publisher: Any | None):
        self.mode = mode if mode in ("off", "shadow", "enforce") else "off"
        self.client = client
        self.publisher = publisher

    # --- reserve -------------------------------------------------------------------------------------------------
    async def reserve(self, user_id: str, job_id: str, attempt: int) -> str | None:
        """The hold id to store on the job (None = no hold). Raises HTTPException on an enforced deny."""
        if self.mode == "off" or self.client is None:
            return None
        result = await self.client.check_and_reserve_manim(user_id, f"{job_id}:{attempt}", correlation_id=job_id)
        if result is None:                                     # unreachable: render proceeds, nothing reserved
            logger.warning("manim_reserve_unavailable_proceeding", job_id=job_id, mode=self.mode)
            return None
        if self.mode == "shadow":
            logger.info("billing_shadow_decision", kind="manim", job_id=job_id, attempt=attempt,
                        allowed=result.allowed, deny_reason=result.deny_reason)
            return result.hold_id if result.allowed else None
        if not result.allowed:
            raise deny_error(result.deny_reason or "account_blocked", result.message_key)
        return result.hold_id

    async def release(self, user_id: str, hold_id: str | None, reason: str) -> None:
        if hold_id and self.client is not None and self.mode != "off":
            await self.client.release(user_id, hold_id, reason)

    # --- repair calls ----------------------------------------------------------------------------------------------
    @staticmethod
    async def record_repair_calls(store: Any, collector: UsageCollector, job: tuple[str, int] | None) -> None:
        if job is not None and collector.calls:
            await store.add_calls(job[0], job[1], collector.snapshot())

    # --- final event -----------------------------------------------------------------------------------------------
    async def finalize(self, store: Any, record: Any) -> None:
        """Called once a job is in a final state (ready / failed / cancelled / expired): releases the hold if the job
        died before rendering, then publishes artifact.manim.rendered, exactly once per billing attempt."""
        if record is None or not await store.claim_event(record.id, record.billing_attempt):
            return
        calls = await store.get_calls(record.id, record.billing_attempt)
        status = "success" if getattr(record.state, "value", record.state) == "ready" else "error"
        if status == "error" and record.hold_id and not record.render_started:
            await self.release(record.owner, record.hold_id, "provider_error")
        if self.publisher is None:
            return
        try:
            await self.publisher.publish_manim_rendered(
                user_id=record.owner, job_id=record.id, attempt=record.billing_attempt, hold_id=record.hold_id,
                status=status, calls=calls)
        except Exception as exc:                               # let the next finalize() retry the publish
            await store.unclaim_event(record.id, record.billing_attempt)
            logger.error("manim_rendered_publish_failed", job_id=record.id, error=type(exc).__name__)


def new_collector() -> UsageCollector:
    return UsageCollector()


__all__ = ["CURRENT_JOB", "ManimBilling", "deny_error", "new_collector", "usage_purpose", "use_collector"]
