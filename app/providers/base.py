"""
Base Provider Adapter Abstract Interface.
Implements LLD v2.0 Section 20.1.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from app.providers.models import GenerationResponse


class BaseProviderAdapter(ABC):
    """
    Abstract contract for all LLM provider integrations (NVIDIA, Gemini, Groq).
    """

    def __init__(self, provider_name: str, model_name: str):
        self.provider_name = provider_name
        self.model_name = model_name
        # How this model thinks longer (set from the registry by the group chain builder; kind "none" = it cannot).
        self.think_spec: Any = None

    def wants_think(self, params: dict[str, Any]) -> bool:
        """True when this call asked to think longer AND this model has a reasoning control. Marks the call so the chain can
        report that reasoning was really used (chat.response.generated.think)."""
        spec = self.think_spec
        if not params.get("think") or spec is None or spec.kind == "none":
            return False
        params["_think_applied"] = True
        return True

    @abstractmethod
    async def execute(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        """
        Executes a complete (non-streaming) completion request.
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Yields tokens incrementally as an asynchronous stream.
        """
        pass


# Appended to a streamed answer that stopped because it hit the output limit, so a cut-off answer is never
# presented as if it were complete.
TRUNCATION_NOTICE = (
    "\n\n---\n*The response was cut off because it reached the length limit. "
    "Reply \u201ccontinue\u201d to get the rest.*"
)
