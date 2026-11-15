"""Internal: model availability for billing-service. Service token required; never routed by the gateway."""

import hmac

from fastapi import APIRouter, Header, HTTPException

from app.api.dependencies import ContainerDep
from app.providers.availability import build_availability

router = APIRouter(tags=["models-internal"])


@router.get("/models/availability")
async def models_availability(container: ContainerDep, x_service_token: str | None = Header(default=None)) -> dict:
    secret = container.config.internal_service_token.get_secret_value()
    if not secret or not hmac.compare_digest(secret, x_service_token or ""):
        raise HTTPException(status_code=401, detail="Service token required")
    group_chains = getattr(container, "group_chains", None)
    if group_chains is None:
        raise HTTPException(status_code=503, detail="Model registry is not loaded")
    return build_availability(group_chains)
