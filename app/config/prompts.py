"""
Prompt configuration paths and templates loader settings.
"""

from pathlib import Path

from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    prompts_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "prompts")
