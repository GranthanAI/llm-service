"""
Which models can answer right now (GET /internal/models/availability, called by billing-service every 5 minutes).

A model is available when it is enabled in models.yaml AND its provider has credentials AND an adapter exists AND its
circuit breaker is not open AND (Google models) the provider still lists the id. `reason` is "" when available, else
disabled | no_credentials | no_adapter | breaker_open | "retired by provider".
"""

from datetime import UTC, datetime
from typing import Any

from app.providers.provider_models import RETIRED_REASON, ProviderModels
from app.utils.circuit_breaker import CircuitState


async def build_availability(group_chains: Any, provider_models: ProviderModels | None = None) -> dict[str, Any]:
    registry = group_chains.registry
    listed = await provider_models.google_ids() if provider_models is not None else None
    models = []
    for model_id, spec in registry.models.items():
        reason = ""
        if not spec.enabled:
            reason = "disabled"
        elif not group_chains._has_credentials(spec.provider):
            reason = "no_credentials"
        elif not group_chains.can_serve(spec):
            reason = "no_adapter"
        elif spec.provider == "google" and provider_models is not None and (
            provider_models.is_retired(spec.provider_model) or (listed is not None and spec.provider_model not in listed)
        ):
            reason = RETIRED_REASON
        else:
            factory = getattr(group_chains, "_breaker_factory", None)
            if factory is not None:
                from app.providers.group_chain import ENTRY_PREFIX

                breaker = factory(f"{ENTRY_PREFIX[spec.provider]}:{spec.provider_model}")
                if breaker is not None and breaker.state == CircuitState.OPEN:
                    reason = "breaker_open"
        models.append({"id": model_id, "available": reason == "", "reason": reason})
    return {"models": models, "checked_at": datetime.now(UTC).isoformat()}
