"""
How an adapter hands the provider-reported token usage of one call to the request's usage collector.

The collector (app/usage/collector.py) travels in the per-request `params` dict under "_usage". Adapters are shared
between requests, so usage is never stored on the adapter itself. A call made without a collector (tests, tools)
simply reports nothing.
"""

from typing import Any


def report_usage(
    params: dict[str, Any] | None,
    *,
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_input_tokens: int | None = 0,
) -> None:
    """Report what the PROVIDER said this call used. None for a count the provider did not send; the collector
    then estimates it (and flags usage_estimated)."""
    sink = (params or {}).get("_usage")
    if sink is None:
        return
    sink.record_provider_usage(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
    )
