"""
Priority Trimmer for Prompt Sections.
Implements LLD v2.0 Section 18.4 and Section 18.5.
"""

import structlog

from app.config.logging import get_logger
from app.context_window.token_counter import TokenCounter
from app.exceptions.analysis import ContextOverflowError
from app.prompts.schemas import PromptSection

logger = get_logger("priority_trimmer")


class PriorityTrimmer:
    """
    Trims prompt sections in ascending priority order (lowest priority trimmed first)
    down to defined safety minimums to enforce model context budgets.
    """

    # Section minimum token floors from LLD Section 18.5
    SECTION_MINIMUMS: dict[str, int] = {
        "system": 0,  # Locked / Not trimmable
        "user_query": 0,  # Locked / Not trimmable
        "draft_content": 200,  # LangGraph synthesized output
        "conversation_history": 200,  # Last conversation turns
        "long_term_memory": 100,  # Top user facts
        "graph_context": 0,  # Can be fully removed if budget constrained
        "retrieval": 0,  # Can be fully removed if budget constrained
        "tool_results": 50,  # At least one search snippet
    }

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        logger_instance: structlog.stdlib.BoundLogger | None = None,
    ):
        self.token_counter = token_counter or TokenCounter()
        self.logger = logger_instance or logger

    def trim(
        self,
        sections: list[PromptSection],
        total_budget: int,
        provider: str = "nvidia",
    ) -> tuple[list[PromptSection], list[str]]:
        """
        Trims sections to satisfy total_budget.
        Returns (trimmed_sections_list, list_of_trimmed_section_names).
        Raises ContextOverflowError if budget cannot be met after maximum trimming.
        """
        current_total = sum(s.token_count for s in sections)
        if current_total <= total_budget:
            return [s.model_copy() for s in sections], []

        excess = current_total - total_budget
        self.logger.info(
            "Prompt exceeds target token budget, initiating priority trimming",
            current_tokens=current_total,
            budget=total_budget,
            excess=excess,
        )

        # Clone sections to prevent mutating caller state
        working_sections = [s.model_copy() for s in sections]
        trimmed_section_names: list[str] = []

        # Sort trimmable sections by priority ASCENDING (lowest priority trimmed first)
        trimmable_indices = [
            i for i, s in enumerate(working_sections) if s.trimmable and s.priority < 10
        ]
        trimmable_indices.sort(key=lambda idx: working_sections[idx].priority)

        for idx in trimmable_indices:
            if excess <= 0:
                break

            sec = working_sections[idx]
            min_allowed = self.SECTION_MINIMUMS.get(sec.name, 0)
            can_trim = max(0, sec.token_count - min_allowed)

            if can_trim <= 0:
                continue

            actual_trim = min(can_trim, excess)
            target_tokens = sec.token_count - actual_trim

            if target_tokens <= 0:
                sec.content = ""
                sec.token_count = 0
            else:
                sec.content = self.token_counter.truncate_text_to_tokens(
                    sec.content, target_tokens, provider
                )
                sec.token_count = self.token_counter.count_text(sec.content, provider)

            excess -= actual_trim
            if sec.name not in trimmed_section_names:
                trimmed_section_names.append(sec.name)

            self.logger.debug(
                "Trimmed prompt section",
                section=sec.name,
                priority=sec.priority,
                trimmed_tokens=actual_trim,
                remaining_section_tokens=sec.token_count,
            )

        # Re-filter any sections that became completely empty
        final_sections = [s for s in working_sections if s.content.strip() != ""]

        final_total = sum(s.token_count for s in final_sections)
        if final_total > total_budget:
            raise ContextOverflowError(
                f"Cannot fit prompt within token budget ({total_budget}) even after maximum trimming. "
                f"Locked sections require {final_total} tokens."
            )

        return final_sections, trimmed_section_names
