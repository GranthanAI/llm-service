"""
Provider Adapter and Generation Data Models.
Implements LLD v2.0 Section 19.3 and Section 20.
"""

from typing import Any

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration parameters for an LLM provider adapter."""

    name: str  # "nvidia" | "gemini" | "groq"
    model: str
    api_key: str = ""
    base_url: str | None = None
    timeout_ms: int = 15000
    max_retries: int = 2


class ProviderSelection(BaseModel):
    """Routing specification for generation."""

    primary: str = "nvidia"
    fallback: str = "gemini"


class GenerationResponse(BaseModel):
    """Complete non-streaming LLM generation response."""

    content: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_s: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
