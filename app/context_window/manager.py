"""
Context Window Manager Implementation.
Implements LLD v2.0 Section 18.1 and Section 18.2.
"""

from typing import Any

import structlog

from app.config.logging import get_logger
from app.context_window.budget import BudgetCalculator
from app.context_window.models import ModelLimits, TrimmedPrompt
from app.context_window.token_counter import TokenCounter
from app.context_window.trimmer import PriorityTrimmer
from app.prompts.schemas import ComposedPrompt
from app.utils.metrics import CONTEXT_TOKENS
from app.utils.tracing import get_tracer

logger = get_logger("context_window_manager")


class ContextWindowManager:
    """
    Orchestrates token counting, model-specific context window budget calculations,
    priority trimming, and OpenAI-compatible message generation.
    """

    DEFAULT_MODEL_LIMITS: dict[str, ModelLimits] = {
        "meta/llama-3.1-405b-instruct": ModelLimits(
            model="meta/llama-3.1-405b-instruct",
            context_window=131072,
            max_output_tokens=4096,
            provider="nvidia",
        ),
        "meta/llama-3.1-70b-instruct": ModelLimits(
            model="meta/llama-3.1-70b-instruct",
            context_window=131072,
            max_output_tokens=4096,
            provider="nvidia",
        ),
        "gemini-1.5-pro": ModelLimits(
            model="gemini-1.5-pro",
            context_window=2097152,
            max_output_tokens=8192,
            provider="gemini",
        ),
        "gemini-1.5-flash": ModelLimits(
            model="gemini-1.5-flash",
            context_window=1048576,
            max_output_tokens=8192,
            provider="gemini",
        ),
        "llama-3.3-70b-versatile": ModelLimits(
            model="llama-3.3-70b-versatile",
            context_window=131072,
            max_output_tokens=4096,
            provider="groq",
        ),
        "default": ModelLimits(
            model="default",
            context_window=131072,
            max_output_tokens=4096,
            provider="nvidia",
        ),
    }

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        trimmer: PriorityTrimmer | None = None,
        model_limits: dict[str, ModelLimits] | None = None,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        self.token_counter = token_counter or TokenCounter()
        self.trimmer = trimmer or PriorityTrimmer(token_counter=self.token_counter)
        self.provider_limits = model_limits or dict(self.DEFAULT_MODEL_LIMITS)
        self.logger = logger_instance or logger
        self.tracer = get_tracer()

    def register_model_limits(self, model: str, limits: ModelLimits) -> None:
        """Registers custom context limits for a specific model."""
        self.provider_limits[model] = limits

    def get_model_limits(self, target_model: str) -> ModelLimits:
        """Retrieves model limits or falls back to default."""
        return self.provider_limits.get(target_model, self.provider_limits["default"])

    def count_tokens(self, text: str, provider: str = "nvidia") -> int:
        """Counts calibrated tokens for given text."""
        return self.token_counter.count_text(text, provider)

    def count_messages(self, messages: list[dict[str, Any]], provider: str = "nvidia") -> int:
        """Counts calibrated tokens for an OpenAI message payload."""
        return self.token_counter.count_messages(messages, provider)

    def manage(
        self,
        prompt: ComposedPrompt,
        target_model: str = "meta/llama-3.1-405b-instruct",
        custom_input_budget: int | None = None,
    ) -> TrimmedPrompt:
        """
        Calculates token budget, executes priority trimming if required,
        and constructs the finalized TrimmedPrompt.
        """
        with self.tracer.start_as_current_span("context_window_manager.manage") as span:
            span.set_attribute("target_model", target_model)
            span.set_attribute("mode", prompt.mode)

            model_limits = self.get_model_limits(target_model)
            budget_allocation = BudgetCalculator.calculate(model_limits)
            effective_budget = custom_input_budget or budget_allocation.input_budget

            original_tokens = sum(s.token_count for s in prompt.sections)

            # Execute priority trimming if needed
            final_sections, trimmed_sections = self.trimmer.trim(
                sections=prompt.sections,
                total_budget=effective_budget,
                provider=model_limits.provider,
            )

            was_trimmed = len(trimmed_sections) > 0
            final_total_tokens = sum(s.token_count for s in final_sections)

            # Record Prometheus token distribution metrics
            for s in final_sections:
                try:
                    CONTEXT_TOKENS.labels(section=s.name, mode=prompt.mode).observe(s.token_count)
                except Exception:
                    pass

            # Reconstruct OpenAI-compatible message payload
            system_sections = [s.content for s in final_sections if s.name == "system"]
            system_prompt = "\n\n".join(system_sections)

            user_sections = [s.content for s in final_sections if s.name != "system"]
            user_prompt = "\n\n".join(user_sections)

            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})

            self.logger.info(
                "Context window management complete",
                target_model=target_model,
                mode=prompt.mode,
                original_tokens=original_tokens,
                final_tokens=final_total_tokens,
                was_trimmed=was_trimmed,
                trimmed_sections=trimmed_sections,
            )

            return TrimmedPrompt(
                sections=final_sections,
                messages=messages,
                total_tokens=final_total_tokens,
                original_tokens=original_tokens,
                was_trimmed=was_trimmed,
                trimmed_sections=trimmed_sections,
                mode=prompt.mode,
                target_model=target_model,
            )
