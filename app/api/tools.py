"""
Tool discovery for clients (routed through the API Gateway as GET /api/v1/tools).

Lists only tools (and response modes) that are registered/enabled right now, so the UI can never offer a tool the
pipeline would reject. Authentication is enforced by the gateway (JWT); this service is not exposed
publicly and the response contains no secrets.
"""

from fastapi import APIRouter

from app.api.dependencies import ContainerDep
from app.tools.selection import MAX_SELECTED_TOOLS, list_enabled_tools
from app.workflow_engine.modes import available_modes

router = APIRouter(prefix="/v1", tags=["tools"])


@router.get("/tools")
async def list_tools(container: ContainerDep) -> dict:
    registry = container.tool_registry
    tools = list_enabled_tools(registry) if registry is not None else []
    # `modes` are the selectable response modes; "Auto" (no mode) is implicit and is the default.
    return {"tools": tools, "max_selected": MAX_SELECTED_TOOLS, "modes": available_modes(container.config)}
