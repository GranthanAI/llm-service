"""
API routers aggregation.
"""

from fastapi import APIRouter

from app.api.internal import internal_router
from app.api.tools import router as tools_router
from app.api.visualizations import internal_router as viz_internal_router
from app.api.visualizations import router as viz_router

api_router = APIRouter()
api_router.include_router(internal_router, prefix="/internal")
api_router.include_router(tools_router)
api_router.include_router(viz_router)
api_router.include_router(viz_internal_router)

__all__ = ["api_router"]
