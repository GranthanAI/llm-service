"""
Execution Plan and Reasoning Enums and Data Models.
Implements LLD v2.0 Section 10.5 and HLD v2.0 Section 9.4.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class IntentCategory(StrEnum):
    """Classification of user intent by the Request Analyzer."""

    GENERAL_CHAT = "GENERAL_CHAT"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    CODE_GENERATION = "CODE_GENERATION"
    CODE_DEBUGGING = "CODE_DEBUGGING"
    CODE_EXPLANATION = "CODE_EXPLANATION"
    RESEARCH = "RESEARCH"
    WEB_SEARCH = "WEB_SEARCH"
    DOCUMENT_ANALYSIS = "DOCUMENT_ANALYSIS"
    TUTORING = "TUTORING"
    CREATIVE_WRITING = "CREATIVE_WRITING"
    REASONING = "REASONING"


class UserMode(StrEnum):
    """Authoritative user execution modes routed by ModeDispatcher."""

    DEFAULT = "default"
    TUTOR = "tutor"
    CODE = "code"
    ASK_FILES = "ask_files"
    WEB_SEARCH = "web_search"
    SMART = "smart"
    DEEP_RESEARCH = "deep_research"


class Skill(StrEnum):
    """Skill domain activated for system prompt styling."""

    GENERAL_CHAT = "general_chat"
    TUTOR = "tutor"
    CODING = "coding"
    RESEARCH = "research"
    WRITING = "writing"
    REASONING = "reasoning"


class ReasoningMode(StrEnum):
    """Reasoning strategy chosen for the request."""

    DIRECT = "DIRECT"
    CHAIN_OF_THOUGHT = "CHAIN_OF_THOUGHT"
    REACT = "REACT"


class EngineType(StrEnum):
    """Underlying execution engine category."""

    MODE_HANDLER = "mode_handler"
    LANGGRAPH = "langgraph"


class ToolCall(BaseModel):
    """Tool invocation definition formulated during planning or graph loops."""

    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    parallel: bool = True
    required: bool = False


class ExecutionPlan(BaseModel):
    """Complete execution plan produced by Request Analyzer (Groq - Call 1)."""

    intent: IntentCategory = IntentCategory.GENERAL_CHAT
    mode: UserMode = UserMode.DEFAULT
    skill: Skill = Skill.GENERAL_CHAT
    reasoning: ReasoningMode = ReasoningMode.DIRECT
    tools: list[ToolCall] = Field(default_factory=list)
    max_iterations: int = Field(default=1, ge=1)
    suggested_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    analysis_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    groq_model_used: str = "llama-3.3-70b-versatile"
    analysis_latency_ms: float = 0.0
