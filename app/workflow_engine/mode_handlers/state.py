"""
Mode Handler State Definition.
Implements LLD v2.0 Section 14.2.
Strictly plain Pydantic with NO LangGraph dependency.
"""

from pydantic import BaseModel, Field

from app.context.schemas import ContextBundle, Message
from app.models.execution_plan import ExecutionPlan
from app.models.tool import ToolResult


class ModeHandlerState(BaseModel):
    """
    Optional lightweight state container for Mode Handlers.
    Contains zero LangGraph reducers or graph artifacts.
    """

    conversation_id: str
    user_id: str
    request_id: str
    user_message: str
    plan: ExecutionPlan
    context_bundle: ContextBundle
    conversation_history: list[Message] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
