"""
API routers aggregation.
"""

from fastapi import APIRouter

from app.api.internal import internal_router

api_router = APIRouter()
api_router.include_router(internal_router, prefix="/internal")

__all__ = ["api_router"]
