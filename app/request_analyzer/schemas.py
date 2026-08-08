"""
Request Analyzer Schemas and Safe Default Factory.
Implements LLD v2.0 Section 10.4 and Section 10.5.
"""

from pydantic import BaseModel, Field

from app.models.execution_plan import (
    EngineType,
    ExecutionPlan,
    IntentCategory,
    ReasoningMode,
    Skill,
    ToolCall,
    UserMode,
)


class AnalysisPromptVariables(BaseModel):
    """Template variables passed to Request Analyzer system prompt."""

    formatted_memory: str = ""
    formatted_graph: str = ""
    formatted_retrieval: str = ""
    tools_json_schema: str = "[]"
    mode_hint: str = "none"
    user_message: str = ""


class SafeDefaultPlan(ExecutionPlan):
    """
    Deterministic fallback plan used when Groq is unavailable or fails to parse.
    Guaranteed to route to DefaultHandler (never a LangGraph graph).
    """

    intent: IntentCategory = IntentCategory.GENERAL_CHAT
    mode: UserMode = UserMode.DEFAULT
    skill: Skill = Skill.GENERAL_CHAT
    reasoning: ReasoningMode = ReasoningMode.DIRECT
    tools: list[ToolCall] = Field(default_factory=list)
    max_iterations: int = 1
    suggested_temperature: float = 0.7
    analysis_confidence: float = 0.0
    groq_model_used: str = "fallback_safe_default"
    analysis_latency_ms: float = 0.0


__all__ = [
    "ExecutionPlan",
    "IntentCategory",
    "UserMode",
    "Skill",
    "ReasoningMode",
    "EngineType",
    "ToolCall",
    "SafeDefaultPlan",
    "AnalysisPromptVariables",
]
