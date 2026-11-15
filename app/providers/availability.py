"""
Which models can answer right now (GET /internal/models/availability, called by billing-service every 5 minutes).

A model is available when it is enabled in models.yaml AND its provider has credentials AND an adapter exists AND its
circuit breaker is not open. `reason` is "" when available, else disabled | no_credentials | no_adapter | breaker_open.
"""

from datetime import UTC, datetime
from typing import Any

from app.utils.circuit_breaker import CircuitState


def build_availability(group_chains: Any) -> dict[str, Any]:
    registry = group_chains.registry
    models = []
    for model_id, spec in registry.models.items():
        reason = ""
        if not spec.enabled:
            reason = "disabled"
        elif not group_chains._has_credentials(spec.provider):
            reason = "no_credentials"
        elif not group_chains.can_serve(spec):
            reason = "no_adapter"
        else:
            factory = getattr(group_chains, "_breaker_factory", None)
            if factory is not None:
                from app.providers.group_chain import ENTRY_PREFIX

                breaker = factory(f"{ENTRY_PREFIX[spec.provider]}:{spec.provider_model}")
                if breaker is not None and breaker.state == CircuitState.OPEN:
                    reason = "breaker_open"
        models.append({"id": model_id, "available": reason == "", "reason": reason})
    return {"models": models, "checked_at": datetime.now(UTC).isoformat()}
