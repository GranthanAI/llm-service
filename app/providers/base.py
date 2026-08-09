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
