"""
LLM Provider and Prompt Composition Data Models.
Implements LLD v2.0 Section 17.5, 18.1, 19.1, and 23.1.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ProviderType(StrEnum):
    """Supported LLM providers."""

    GROQ = "groq"
    NVIDIA = "nvidia"
    GEMINI = "gemini"


class CircuitState(StrEnum):
    """Circuit breaker state."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class ModelLimits(BaseModel):
    """Token limits and context window parameters for a model."""

    model: str
    context_window: int = 128000
    max_output_tokens: int = 4096
    effective_input: int = 120000


class ProviderSelection(BaseModel):
    """Routing decision made by GenerationRouter."""

    provider: ProviderType
    model: str
    is_fallback: bool = False
    reason: str | None = None


class GenerationConfig(BaseModel):
    """Hyperparameters for LLM generation."""

    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95
    stream: bool = True
    stop_sequences: list[str] = Field(default_factory=list)


class PromptSection(BaseModel):
    """Individual section of an assembled prompt with priority budgeting."""

    name: str
    content: str
    priority: int = Field(default=5, ge=1, le=10)  # 1-10; lower = trimmed first
    token_count: int = 0
    trimmable: bool = True


class ComposedPrompt(BaseModel):
    """Full assembled prompt before context window trimming."""

    sections: list[PromptSection] = Field(default_factory=list)
    total_tokens: int = 0
    messages: list[dict[str, Any]] = Field(default_factory=list)
    mode: str
    engine_type: str = "mode_handler"  # "mode_handler" | "langgraph"
    template_version: str = "2.0"


class TrimmedPrompt(BaseModel):
    """Final prompt trimmed to fit model context window."""

    sections: list[PromptSection] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    total_tokens: int = 0
    was_trimmed: bool = False
    trimmed_sections: list[str] = Field(default_factory=list)
