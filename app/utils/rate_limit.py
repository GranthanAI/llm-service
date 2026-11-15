"""What a provider's 429 tells us about when to try again, and the longest we are willing to wait for it."""

import re
from collections.abc import Mapping
from typing import Any

# One call never waits longer than this for a rate limit to clear; a longer wait fails the call at once, so a request is
# never held for minutes behind a provider's limit (the user sees an error and can retry).
MAX_RATE_LIMIT_WAIT_SECONDS = 10.0

_UNIT = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def parse_duration_seconds(value: str | None) -> float | None:
    """Retry-After seconds ("3", "1.5") or Groq's x-ratelimit-reset-* durations ("7.66s", "2m59.56s", "1h2m", "250ms")."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    parts = _UNIT.findall(text)
    if not parts or "".join(n + u for n, u in parts) != text.replace(" ", ""):
        return None
    return sum(float(n) * _SECONDS[u] for n, u in parts)


def retry_after_from_headers(headers: Mapping[str, Any] | None) -> float | None:
    """Retry-After when present, else the sooner of Groq's x-ratelimit-reset-requests / -tokens."""
    if not headers:
        return None
    get = {str(k).lower(): v for k, v in dict(headers).items()}
    explicit = parse_duration_seconds(get.get("retry-after"))
    if explicit is not None:
        return explicit
    resets = [
        d
        for d in (
            parse_duration_seconds(get.get("x-ratelimit-reset-requests")),
            parse_duration_seconds(get.get("x-ratelimit-reset-tokens")),
        )
        if d is not None
    ]
    return min(resets) if resets else None


def retry_after_of(exc: BaseException) -> float | None:
    """The wait a provider SDK exception asks for (from its HTTP response headers), or None."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    try:
        return retry_after_from_headers(headers)
    except Exception:  # noqa: BLE001 - a header we cannot read must never hide the 429 itself
        return None
