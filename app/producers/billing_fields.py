"""
The billing fields every terminal chat event carries (billing integration plan 4.4.2): hold_id, model_group, model,
downgraded and calls. Built the same way for success, error and cancel, and for legacy requests (no billing grant:
hold_id and model_group are null, the rest is filled, so billing can cost legacy traffic too).
"""

from typing import Any


def _answered_model(ctx: Any, *, answered: bool) -> str | None:
    """The catalog id of the model that actually answered, after any fallback. For a legacy request the provider model
    string. None when nothing answered (an error before the first token)."""
    if getattr(ctx, "answered_model", None):
        return ctx.answered_model
    collector = getattr(ctx, "usage_collector", None)
    if collector is not None:
        for call in reversed(collector.calls_for("answer")):
            if call.output_tokens > 0:
                return call.model
    if answered and getattr(ctx, "selected_provider", None):
        name = str(ctx.selected_provider)
        return name.split(":", 1)[1] if ":" in name else name
    return None


def billing_fields(ctx: Any, *, answered: bool = False) -> dict[str, Any]:
    grant = getattr(ctx, "billing", None)
    collector = getattr(ctx, "usage_collector", None)
    return {
        "hold_id": grant.hold_id if grant is not None else None,
        "model_group": grant.granted_group if grant is not None else None,
        "model": _answered_model(ctx, answered=answered),
        "downgraded": bool(grant.downgraded) if grant is not None else False,
        "calls": collector.snapshot() if collector is not None else [],
    }
