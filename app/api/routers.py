"""
API routers aggregation.
"""

from fastapi import APIRouter

from app.api.internal import internal_router
from app.api.tools import router as tools_router

api_router = APIRouter()
api_router.include_router(internal_router, prefix="/internal")
api_router.include_router(tools_router)

__all__ = ["api_router"]
