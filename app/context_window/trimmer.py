"""
Priority Trimmer for Prompt Sections.
Implements LLD v2.0 Section 18.4 and Section 18.5.
"""

import re

import structlog

from app.config.logging import get_logger
from app.context_window.token_counter import TokenCounter
from app.exceptions.analysis import ContextOverflowError
from app.prompts.schemas import PromptSection

logger = get_logger("priority_trimmer")

_HISTORY_ROLE_LINE = re.compile(r"^(User|Assistant|System|Tool): ")


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
                sec.items = None
                sec.token_count = 0
            else:
                self._shrink(sec, target_tokens, provider)
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

    # ─── section-aware shrinking ─────────────────────────────────────────────────────────────────────────────────────
    # Cutting every section from the END keeps the oldest turns of a conversation and drops the newest, and can stop in
    # the middle of a line or a fact. Both are wrong: history is trimmed from the OLDEST end at message boundaries, and
    # everything else at line (fact / passage) boundaries.

    def _shrink(self, sec: PromptSection, target_tokens: int, provider: str) -> None:
        if sec.name == "conversation_history":
            self._shrink_history(sec, target_tokens, provider)
        else:
            sec.content = self._keep_whole_lines(sec.content, target_tokens, provider)

    def _keep_whole_lines(self, text: str, target_tokens: int, provider: str) -> str:
        """The longest prefix made of WHOLE lines that fits; empty when not even the first line fits."""
        kept: list[str] = []
        used = 0
        for line in text.split("\n"):
            cost = self.token_counter.count_text(line, provider) + 1
            if used + cost > target_tokens:
                break
            kept.append(line)
            used += cost
        if kept:
            return "\n".join(kept).rstrip()
        # Not even the first line fits (running prose without line breaks): keep its start, cut at a word boundary.
        return self._cut_at_word(self.token_counter.truncate_text_to_tokens(text, target_tokens, provider), text)

    @staticmethod
    def _cut_at_word(cut: str, original: str) -> str:
        """`cut` is a prefix of `original`; step back to the last whitespace so no word is split."""
        if len(cut) >= len(original) or original[len(cut)].isspace():
            return cut.rstrip()
        space = max(cut.rfind(" "), cut.rfind("\n"), cut.rfind("\t"))
        return cut[:space].rstrip() if space > 0 else ""

    @staticmethod
    def _history_items(sec: PromptSection) -> tuple[str, list[dict[str, str]]]:
        """(header, messages). Structured items when the builder supplied them; otherwise parsed from the text
        (sections built by hand): a line starting with a role prefix begins a message, other lines continue it."""
        if sec.items is not None:
            return sec.header, [dict(i) for i in sec.items]
        header, items = "", []
        for line in sec.content.split("\n"):
            match = _HISTORY_ROLE_LINE.match(line)
            if match:
                items.append({"role": match.group(1).lower(), "content": line[match.end():]})
            elif items:
                items[-1]["content"] += "\n" + line
            elif not header:
                header = line
            else:
                header += "\n" + line
        return header, items

    @staticmethod
    def render_history(header: str, items: list[dict[str, str]]) -> str:
        lines = [f"{i['role'].capitalize()}: {i['content']}" for i in items]
        return "\n".join(([header] if header else []) + lines)

    def _shrink_history(self, sec: PromptSection, target_tokens: int, provider: str) -> None:
        header, items = self._history_items(sec)
        if not items:                                          # unstructured text: keep the NEWEST end, at a word boundary
            words = sec.content.split(" ")
            while words and self.token_counter.count_text(" ".join(words), provider) > target_tokens:
                words.pop(0)
            sec.content = " ".join(words).strip()
            return
        while items and self.token_counter.count_text(self.render_history(header, items), provider) > target_tokens:
            items.pop(0)                                       # the OLDEST message goes first; the newest stay whole
        if sec.items is not None:
            sec.items = items
        sec.content = self.render_history(header, items) if items else ""
