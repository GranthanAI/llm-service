"""
Budget Calculator for Context Window Allocations.
Implements LLD v2.0 Section 18.3.
"""

from app.context_window.models import BudgetAllocation, ModelLimits


class BudgetCalculator:
    """
    Calculates effective input token budgets and per-section maximums
    based on target model limits and safety buffers.
    """

    @staticmethod
    def calculate(model: ModelLimits, safety_margin: float = 0.95) -> BudgetAllocation:
        """
        Calculates input budget and prioritized section quotas.
        """
        effective_window = int(model.context_window * safety_margin)
        reserved_output = model.max_output_tokens
        input_budget = max(500, effective_window - reserved_output)

        # Proportional distribution per LLD Section 18.3
        section_budgets = {
            "system": int(input_budget * 0.10),
            "long_term_memory": int(input_budget * 0.10),
            "graph_context": int(input_budget * 0.10),
            "retrieval": int(input_budget * 0.15),
            "tool_results": int(input_budget * 0.15),
            "draft_content": int(input_budget * 0.10),
            "conversation_history": int(input_budget * 0.15),
            "user_query": max(100, int(input_budget * 0.05)),
        }

        return BudgetAllocation(
            effective_window=effective_window,
            reserved_output=reserved_output,
            input_budget=input_budget,
            section_budgets=section_budgets,
        )
