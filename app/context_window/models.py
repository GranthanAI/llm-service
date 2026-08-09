"""
Context Window Manager Data Models and Schemas.
Implements LLD v2.0 Section 18.1.
"""

from typing import Any

from pydantic import BaseModel, Field

from app.prompts.schemas import PromptSection


class ModelLimits(BaseModel):
    """Token boundaries and context window specifications for a target model."""

    model: str
    context_window: int = 131072
    max_output_tokens: int = 4096
    effective_input: int = Field(default=0)
    provider: str = "nvidia"  # "nvidia" | "gemini" | "groq"

    def model_post_init(self, __context: Any) -> None:
        if self.effective_input == 0:
            self.effective_input = int(self.context_window * 0.95) - self.max_output_tokens


class BudgetAllocation(BaseModel):
    """Detailed token budget allocated per prompt section."""

    effective_window: int
    reserved_output: int
    input_budget: int
    section_budgets: dict[str, int] = Field(default_factory=dict)


class TrimmedPrompt(BaseModel):
    """
    Optimized and budget-enforced prompt payload ready for provider generation.
    """

    sections: list[PromptSection] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)  # OpenAI format
    total_tokens: int = 0
    original_tokens: int = 0
    was_trimmed: bool = False
    trimmed_sections: list[str] = Field(default_factory=list)
    mode: str = "default"
    target_model: str = "meta/llama-3.1-405b-instruct"
