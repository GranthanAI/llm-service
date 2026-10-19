"""
Context-budget resolution: one pure, testable function that answers "how many INPUT tokens may this request use on this
provider/model?".

    total   = min(application budget, model window)            # both ceilings must hold
    input   = floor(total * (1 - safety_margin)) - output_reserve
    input   = min(input, legacy PROMPT_TOKEN_BUDGET)            # only when that rate-limit guard is configured

Application budget, in order: per-model override -> provider setting (groq / gemini) -> default. Unknown models never
inherit a provider's limit: their window is the configured conservative value and the result is flagged
`window_assumed`.

This is a CONTEXT budget. A provider's tokens-per-minute quota is a separate, rolling throughput limit and is handled
by the provider adapters (429 / Retry-After), never by this function.
"""

from dataclasses import dataclass

from app.config.settings import LLMServiceConfig
from app.context_window.models import ModelLimits

MIN_INPUT_TOKENS = 1_000
TPM_HEADROOM_TOKENS = 200       # estimate slack inside a tokens-per-minute limit


class ContextBudgetError(ValueError):
    """The configured budget cannot hold even a minimal request (misconfiguration: reserve too large, etc.)."""


@dataclass(frozen=True)
class ContextBudget:
    provider: str
    model: str
    app_budget: int
    app_budget_source: str          # "model_override" | "provider" | "default" | "disabled"
    model_window: int
    window_assumed: bool
    total: int
    safety_margin: float
    output_reserve: int
    input_budget: int
    legacy_cap_applied: bool
    tpm_cap_applied: bool = False


def provider_of(name: str) -> str:
    """'groq:openai/gpt-oss-120b' -> 'groq'; 'gemini' -> 'gemini'; anything else -> as given."""
    return name.split(":", 1)[0].strip().lower()


def model_of(name: str, default_model: str = "") -> str:
    return name.split(":", 1)[1].strip() if ":" in name else default_model


def resolve_context_budget(
    config: LLMServiceConfig,
    provider: str,
    model: str,
    limits: ModelLimits | None = None,
    *,
    window_assumed: bool = False,
    requested_output_tokens: int | None = None,
) -> ContextBudget:
    provider = provider_of(provider)

    # 1. application budget
    if not config.context_budget_enabled:
        app_budget, source = 0, "disabled"
    else:
        overrides = config.context_budget_model_overrides
        if f"{provider}:{model}" in overrides:
            app_budget, source = overrides[f"{provider}:{model}"], "model_override"
        elif model in overrides:
            app_budget, source = overrides[model], "model_override"
        elif provider == "groq" and config.context_budget_groq_tokens:
            app_budget, source = config.context_budget_groq_tokens, "provider"
        elif provider == "gemini" and config.context_budget_gemini_tokens:
            app_budget, source = config.context_budget_gemini_tokens, "provider"
        else:
            app_budget, source = config.context_budget_default_tokens, "default"

    # 2. model window (second, independent ceiling)
    window = limits.context_window if limits is not None else config.context_unknown_model_window_tokens
    assumed = window_assumed or limits is None
    total = min(app_budget, window) if app_budget else window

    # 3. output reserve and safety margin
    reserve = requested_output_tokens or config.context_output_token_reserve
    if limits is not None and not window_assumed:
        reserve = min(reserve, limits.max_output_tokens) if requested_output_tokens is None else reserve
    margin = config.context_token_safety_margin
    input_budget = int(total * (1 - margin)) - reserve

    # Rate-limit-derived cap (NOT the context window): Groq counts prompt + max_tokens against the account's
    # tokens-per-minute limit per request, so on a limited tier the input must leave room for the whole answer or the
    # answer is cut short (or rejected). GROQ_TPM_LIMIT=0 (paid tiers) disables it.
    tpm_capped = False
    if provider == "groq" and config.groq_tpm_limit > 0:
        tpm_input = config.groq_tpm_limit - reserve - TPM_HEADROOM_TOKENS
        if tpm_input < input_budget:
            input_budget, tpm_capped = tpm_input, True

    legacy = False
    if config.prompt_token_budget and config.prompt_token_budget < input_budget:
        input_budget, legacy = config.prompt_token_budget, True
    if input_budget < MIN_INPUT_TOKENS:
        raise ContextBudgetError(
            f"effective input budget for {provider}:{model} is {input_budget} tokens "
            f"(total {total}, reserve {reserve}, margin {margin}); raise the budget or lower the output reserve"
        )
    return ContextBudget(provider=provider, model=model, app_budget=app_budget, app_budget_source=source,
                         model_window=window, window_assumed=assumed, total=total, safety_margin=margin,
                         output_reserve=reserve, input_budget=input_budget, legacy_cap_applied=legacy,
                         tpm_cap_applied=tpm_capped)
