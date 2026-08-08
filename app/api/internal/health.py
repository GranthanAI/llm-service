"""
Health (liveness) check endpoints.
"""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.dependencies import ContainerDep

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse)
async def health_check(container: ContainerDep) -> JSONResponse:
    """Liveness probe: verifies process is alive and responding."""
    if not container.is_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": container.config.service_name,
                "version": container.config.service_version,
                "environment": container.config.environment,
            },
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "service": container.config.service_name,
            "version": container.config.service_version,
            "environment": container.config.environment,
        },
    )


@router.get("/healthz", response_model=HealthResponse)
async def healthz_check(container: ContainerDep) -> JSONResponse:
    """Kubernetes liveness probe alias."""
    return await health_check(container)
