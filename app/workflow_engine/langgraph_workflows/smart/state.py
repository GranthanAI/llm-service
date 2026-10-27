"""
SmartGraph State and SubTask Schema.
Implements LLD v2.0 Section 13.1.2 and Section 14.4.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

from app.context.schemas import ContextBundle, Message
from app.models.execution_plan import ToolCall
from app.models.tool import ToolResult


class SubTask(BaseModel):
    """Planner-generated sub-goal during SmartGraph execution."""

    id: str
    description: str
    tool_name: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)   # arguments the LLM planner chose for the tool
    required: bool = False
    completed: bool = False


class SmartGraphState(TypedDict):
    """
    LangGraph State TypedDict for SmartGraph.
    Uses operator.add reducers for iterative accumulator fields.
    """

    # Identity
    conversation_id: str
    user_id: str
    request_id: str
    mode: Literal["smart"]

    # Inputs
    user_message: str
    context_bundle: ContextBundle
    conversation_history: list[Message]

    # Tool plan decided by the LLM Request Analyzer (which tool, with what arguments) and whether that plan is only the
    # safe default because the analyzer failed (then the deterministic freshness policy decides instead).
    planned_tools: list[ToolCall]
    plan_is_fallback: bool

    # Planner output & loop progression
    sub_tasks: list[SubTask]
    satisfied_sub_tasks: Annotated[list[str], operator.add]

    # Tool loop state
    tool_results: Annotated[list[ToolResult], operator.add]
    next_tool_call: ToolCall | None
    loop_iteration_count: int
    max_iterations: int

    # Terminal output
    draft_response: str | None
