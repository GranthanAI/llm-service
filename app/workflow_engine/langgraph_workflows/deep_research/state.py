"""
DeepResearchGraph State and Finding Schema.
Implements LLD v2.0 Section 13.2.2 and Section 14.4.
"""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from app.context.schemas import ContextBundle, Message
from app.models.tool import ToolResult


class Finding(BaseModel):
    """Information unit extracted and synthesized from a research source."""

    source: str
    title: str = ""
    url: str | None = None
    snippet: str
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    corroborated: bool = False


class DeepResearchGraphState(TypedDict):
    """
    LangGraph State TypedDict for DeepResearchGraph.
    Uses operator.add reducers for iterative accumulator fields.
    """

    # Identity
    conversation_id: str
    user_id: str
    request_id: str
    mode: Literal["deep_research"]

    # Inputs
    user_message: str
    context_bundle: ContextBundle
    conversation_history: list[Message]

    # Research iterations
    queries_issued: Annotated[list[str], operator.add]
    search_results: Annotated[list[ToolResult], operator.add]
    findings: Annotated[list[Finding], operator.add]
    coverage_sufficient: bool
    loop_iteration_count: int
    max_iterations: int

    # Cross-referencing and Synthesis outputs
    cross_referenced_findings: list[Finding] | None
    synthesis: str | None
    structured_report: str | None
