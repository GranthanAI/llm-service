"""
Readiness check endpoints.
"""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.dependencies import ContainerDep

router = APIRouter(tags=["readiness"])


class ReadinessResponse(BaseModel):
    status: str
    ready: bool
    service: str
    version: str


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(container: ContainerDep) -> JSONResponse:
    """Readiness probe: verifies all startup tasks have completed and service is ready for traffic."""
    if not container.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "ready": False,
                "service": container.config.service_name,
                "version": container.config.service_version,
            },
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ready",
            "ready": True,
            "service": container.config.service_name,
            "version": container.config.service_version,
        },
    )


@router.get("/readyz", response_model=ReadinessResponse)
async def readyz_check(container: ContainerDep) -> JSONResponse:
    """Kubernetes readiness probe alias."""
    return await readiness_check(container)
