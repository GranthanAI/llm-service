"""
How an adapter hands the provider-reported token usage of one call to the request's usage collector.

The collector (app/usage/collector.py) is the request's ContextVar (set by the pipeline) or, for a direct call,
`params["_usage"]`. Adapters are shared between requests, so usage is never stored on the adapter itself. A call made
with no collector (tests, tools) reports nothing.
"""

from typing import Any

from app.usage.collector import current_collector


def report_usage(
    params: dict[str, Any] | None,
    *,
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_input_tokens: int | None = 0,
    messages: list[dict[str, Any]] | None = None,
    output_text: str | None = None,
) -> None:
    """Report what the PROVIDER said this call used. None for a count the provider did not send; the collector then
    estimates it from `messages` / `output_text` (and flags usage_estimated)."""
    sink = (params or {}).get("_usage") or current_collector()
    if sink is None:
        return
    sink.record_provider_usage(
        provider=provider, model=model, input_tokens=input_tokens, output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens, messages=messages, output_text=output_text,
    )
