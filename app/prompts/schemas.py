"""
Prompt Engine Schemas and Data Models.
Implements LLD v2.0 Section 17.5.
"""

from typing import Any

from pydantic import BaseModel, Field


class PromptTemplateConfig(BaseModel):
    """Configuration schema for a loaded YAML prompt template."""

    name: str
    version: str = "1.0"
    mode: str | None = None
    graph: str | None = None
    node: str | None = None
    engine: str = "mode_handler"  # "mode_handler" | "langgraph"
    variables: list[str] = Field(default_factory=list)
    system: str
    user_template: str | None = None


class PromptSection(BaseModel):
    """
    Individual section of a composed prompt with priority for context trimming.
    Priority ranges from 1 (lowest priority, trimmed first) to 10 (highest priority, never trimmed).
    """

    name: str
    content: str
    priority: int = Field(default=5, ge=1, le=10)
    token_count: int = 0
    trimmable: bool = True


class ComposedPrompt(BaseModel):
    """
    Fully composed multi-section prompt ready for context window budget management
    and provider generation.
    """

    sections: list[PromptSection] = Field(default_factory=list)
    total_tokens: int = 0
    messages: list[dict[str, Any]] = Field(
        default_factory=list
    )  # OpenAI format: [{"role": "system", ...}, {"role": "user", ...}]
    mode: str
    engine_type: str  # "mode_handler" | "langgraph"
    template_version: str = "1.0"
    system_prompt: str = ""
    user_prompt: str = ""
