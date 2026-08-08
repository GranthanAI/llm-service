"""
Safe Default Plan Factory.
Implements LLD v2.0 Section 10.4.
"""

from app.models.execution_plan import (
    IntentCategory,
    ReasoningMode,
    Skill,
    UserMode,
)
from app.request_analyzer.schemas import SafeDefaultPlan


class SafeDefaultFactory:
    """
    Produces deterministic fallback execution plans when Groq is unavailable,
    trips circuit breaker, or produces unparseable output.
    Guaranteed to route to DefaultHandler (never a LangGraph agentic loop).
    """

    @staticmethod
    def create(latency_ms: float = 0.0, reason: str = "groq_unavailable") -> SafeDefaultPlan:
        """Create canonical safe default plan."""
        return SafeDefaultPlan(
            intent=IntentCategory.GENERAL_CHAT,
            mode=UserMode.DEFAULT,
            skill=Skill.GENERAL_CHAT,
            reasoning=ReasoningMode.DIRECT,
            tools=[],
            max_iterations=1,
            suggested_temperature=0.7,
            analysis_confidence=0.0,
            groq_model_used="fallback_safe_default",
            analysis_latency_ms=latency_ms,
        )
