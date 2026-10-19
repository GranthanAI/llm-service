"""
Fits one request's prompt to its context budget: resolve the budget, decide whether to compact, compact the history,
trim optional context, and validate the FINAL serialized messages. An oversized request is never returned.

    estimate >= trigger_ratio * input_budget   -> compact   (also whenever estimate > input_budget)
    compaction target                          -> target_ratio * input_budget (a goal, not a reason to delete
                                                  protected content)
    after trimming, the serialized messages are recounted with the provider's calibration and must be <= input_budget,
    otherwise ContextOverflowError (a controlled, user-visible error) - never an oversized provider call.
"""

import time
from dataclasses import dataclass
from typing import Any

from app.config.settings import LLMServiceConfig
from app.context_window.budget_policy import ContextBudget, resolve_context_budget
from app.context_window.compaction import compact_history
from app.context_window.manager import ContextWindowManager
from app.context_window.models import TrimmedPrompt
from app.exceptions.analysis import ContextOverflowError
from app.utils import metrics


@dataclass
class FitReport:
    budget: ContextBudget
    tokens_before: int
    tokens_after: int
    compaction: str                 # skipped | compacted | no_effect | failed
    compaction_ms: float = 0.0
    messages_condensed: int = 0
    messages_omitted: int = 0
    trimmed_sections: tuple[str, ...] = ()


def _history_of(result: Any, ctx: Any) -> list:
    if getattr(result, "conversation_history", None):
        return list(result.conversation_history)
    bundle = getattr(ctx, "context_bundle", None)
    memory = getattr(bundle, "memory", None) if bundle else None
    return list(memory.short_term_messages) if memory and memory.short_term_messages else []


def _set_history(result: Any, ctx: Any, messages: list) -> None:
    if getattr(result, "conversation_history", None):
        result.conversation_history = list(messages)
    bundle = getattr(ctx, "context_bundle", None)
    memory = getattr(bundle, "memory", None) if bundle else None
    if memory is not None and memory.short_term_messages:
        memory.short_term_messages = list(messages)


def fit_prompt(
    *,
    config: LLMServiceConfig,
    manager: ContextWindowManager,
    prompt_builder: Any,
    result: Any,
    ctx: Any,
    composed: Any,
    provider: str,
    model: str,
    max_output_tokens: int | None,
    logger: Any,
) -> tuple[TrimmedPrompt, FitReport]:
    limits, assumed = manager.resolve_limits(model)
    budget = resolve_context_budget(config, provider, model, limits, window_assumed=assumed,
                                    requested_output_tokens=max_output_tokens)
    counter = manager.token_counter
    estimate = counter.count_messages(composed.messages, provider)
    before = estimate
    metrics.CONTEXT_BUDGET_TOKENS.labels(provider=_label(provider)).set(budget.input_budget)

    trigger_at = int(budget.input_budget * config.context_compaction_trigger_ratio)
    must = estimate > budget.input_budget
    outcome, condensed, omitted, took = "skipped", 0, 0, 0.0
    if config.context_compaction_enabled and (estimate >= trigger_at or must):
        started = time.perf_counter()
        try:
            history = _history_of(result, ctx)
            outcome = "no_effect"
            if history:
                history_tokens = sum(counter.count_text(m.content, provider) + 4 for m in history)
                target = int(budget.input_budget * config.context_compaction_target_ratio)
                allowance = max(0, target - (estimate - history_tokens))
                bundle = getattr(ctx, "context_bundle", None)
                memory = getattr(bundle, "memory", None) if bundle else None
                cr = compact_history(
                    history, token_budget=allowance, min_recent=config.context_compaction_min_recent_messages,
                    counter=counter, provider=provider, has_memory_summary=bool(memory and memory.summary.strip()))
                if cr.changed:
                    _set_history(result, ctx, cr.messages)
                    composed = prompt_builder.build(result, ctx)
                    estimate = counter.count_messages(composed.messages, provider)    # recount after compaction
                    outcome, condensed, omitted = "compacted", cr.condensed, cr.omitted
        except Exception as exc:                       # compaction must never break the request: trim instead
            outcome = "failed"
            logger.warning("context_compaction_failed", error_type=type(exc).__name__)
        took = (time.perf_counter() - started) * 1000.0
        metrics.CONTEXT_COMPACTION_SECONDS.observe(took / 1000.0)
    metrics.CONTEXT_COMPACTION_TOTAL.labels(outcome=outcome).inc()

    # Trim optional context to the budget (priority order, protected sections never trimmed).
    try:
        trimmed = manager.manage(composed, target_model=model, custom_input_budget=budget.input_budget,
                                 max_output_tokens=max_output_tokens)
        final = counter.count_messages(trimmed.messages, provider)
        if final > budget.input_budget:
            # The section counts under-estimated the serialized payload (role framing, provider calibration):
            # trim again with the measured difference removed from the budget.
            gap = final - trimmed.total_tokens
            trimmed = manager.manage(composed, target_model=model, custom_input_budget=max(1, budget.input_budget - gap),
                                     max_output_tokens=max_output_tokens)
            final = counter.count_messages(trimmed.messages, provider)
        if final > budget.input_budget:
            raise ContextOverflowError(
                f"The request needs {final} tokens but at most {budget.input_budget} fit the context budget "
                f"for this model. Shorten the message or start a new conversation.")
    except ContextOverflowError:
        metrics.CONTEXT_OVERFLOW_TOTAL.labels(provider=_label(provider)).inc()
        raise

    metrics.CONTEXT_INPUT_TOKENS.labels(stage="before", provider=_label(provider)).observe(before)
    metrics.CONTEXT_INPUT_TOKENS.labels(stage="after", provider=_label(provider)).observe(final)
    report = FitReport(budget=budget, tokens_before=before, tokens_after=final, compaction=outcome,
                       compaction_ms=round(took, 1), messages_condensed=condensed, messages_omitted=omitted,
                       trimmed_sections=tuple(trimmed.trimmed_sections))
    logger.info(
        "context_budget",
        provider=_label(provider), budget_source=budget.app_budget_source, app_budget=budget.app_budget,
        model_window=budget.model_window, window_assumed=budget.window_assumed, input_budget=budget.input_budget,
        output_reserve=budget.output_reserve, tokens_before=before, tokens_after=final, compaction=outcome,
        condensed=condensed, omitted=omitted, trimmed_sections=list(trimmed.trimmed_sections),
        legacy_cap=budget.legacy_cap_applied)
    return trimmed, report


def _label(provider: str) -> str:
    """Bounded metric label: a fixed set of provider names, never an arbitrary string."""
    p = provider.split(":", 1)[0].lower()
    return p if p in ("groq", "gemini", "nvidia") else "other"


def build_chain_fitter(config: LLMServiceConfig, manager: ContextWindowManager, logger: Any):
    """Returns fit(prompt, entry_name) -> TrimmedPrompt for ProviderChain: before a provider is tried, the prompt is
    re-validated against THAT provider/model's budget (its own application budget, its real window and its output
    limit). A larger window never re-expands a prompt that was already trimmed; a smaller budget trims further or, if
    the mandatory content cannot fit, raises ContextOverflowError so the provider is skipped instead of being sent an
    oversized request."""
    from app.context_window.budget_policy import model_of, provider_of
    from app.prompts.schemas import ComposedPrompt

    def fit(prompt: TrimmedPrompt, entry_name: str) -> TrimmedPrompt:
        provider = provider_of(entry_name)
        model = model_of(entry_name, {"gemini": config.gemini_model, "nvidia": config.nvidia_model}.get(provider, entry_name))
        max_out = {"nvidia": config.nvidia_max_tokens, "gemini": config.gemini_max_tokens}.get(provider, config.groq_max_tokens)
        limits, assumed = manager.resolve_limits(model)
        budget = resolve_context_budget(config, provider, model, limits, window_assumed=assumed, requested_output_tokens=max_out)
        counter = manager.token_counter
        current = counter.count_messages(prompt.messages, provider)
        if current <= budget.input_budget:
            return prompt
        composed = ComposedPrompt(sections=prompt.sections, total_tokens=prompt.total_tokens, messages=[], mode=prompt.mode,
                                  engine_type="mode_handler")
        trimmed = manager.manage(composed, target_model=model, custom_input_budget=budget.input_budget, max_output_tokens=max_out)
        final = counter.count_messages(trimmed.messages, provider)
        if final > budget.input_budget:
            gap = final - trimmed.total_tokens
            trimmed = manager.manage(composed, target_model=model, custom_input_budget=max(1, budget.input_budget - gap),
                                     max_output_tokens=max_out)
            final = counter.count_messages(trimmed.messages, provider)
        if final > budget.input_budget:
            metrics.CONTEXT_OVERFLOW_TOTAL.labels(provider=_label(provider)).inc()
            raise ContextOverflowError(f"does not fit {provider}'s context budget ({final} > {budget.input_budget} tokens)")
        logger.info("context_refit_for_fallback", provider=_label(provider), tokens_before=current, tokens_after=final,
                    input_budget=budget.input_budget)
        return trimmed

    return fit

