"""
Workflow Result and Normalization Boundary Data Models.
Implements LLD v2.0 Section 11.2 and Section 14.2.
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from app.context.schemas import Message
from app.models.tool import ToolResult


class ModeHandlerOutput(BaseModel):
    """
    Standard output produced by any of the 5 Mode Handlers
    (Default, Tutor, Code, AskFiles, WebSearch).
    """

    mode: str
    tool_outputs: list[ToolResult] = Field(default_factory=list)
    conversation_history: list[Message] = Field(default_factory=list)
    user_message: str


@dataclass
class WorkflowResult:
    """
    Normalized execution output produced by either Mode Handlers or LangGraph.
    Fed directly into the shared PromptBuilder.
    """

    mode: str
    engine_type: str  # "mode_handler" | "langgraph"
    draft_content: str | None
    tool_outputs: list[ToolResult] = field(default_factory=list)
    conversation_history: list[Message] = field(default_factory=list)
    user_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mode_handler(cls, handler_output: ModeHandlerOutput) -> "WorkflowResult":
        """Normalize output from a deterministic Mode Handler."""
        return cls(
            mode=handler_output.mode,
            engine_type="mode_handler",
            draft_content=None,
            tool_outputs=handler_output.tool_outputs,
            conversation_history=handler_output.conversation_history,
            user_message=handler_output.user_message,
            metadata={},
        )

    @classmethod
    def from_graph_state(cls, final_state: dict[str, Any]) -> "WorkflowResult":
        """Normalize output from a compiled LangGraph graph (Smart or DeepResearch)."""
        mode_val = final_state.get("mode", "smart")
        draft = (
            final_state.get("draft_response")
            or final_state.get("structured_report")
            or final_state.get("synthesis")
        )
        tool_results = final_state.get("tool_results") or final_state.get("search_results") or []
        history = final_state.get("conversation_history", [])
        user_msg = final_state.get("user_message", "")
        loop_count = final_state.get("loop_iteration_count", 0)

        return cls(
            mode=str(mode_val),
            engine_type="langgraph",
            draft_content=draft,
            tool_outputs=list(tool_results),
            conversation_history=list(history),
            user_message=str(user_msg),
            metadata={"loop_iterations": loop_count},
        )
