"""
Internal monitoring routers package.
"""

from fastapi import APIRouter

from app.api.internal.health import router as health_router
from app.api.internal.metrics import router as metrics_router
from app.api.internal.readiness import router as readiness_router

internal_router = APIRouter()
internal_router.include_router(health_router)
internal_router.include_router(readiness_router)
internal_router.include_router(metrics_router)

__all__ = ["internal_router", "health_router", "readiness_router", "metrics_router"]
