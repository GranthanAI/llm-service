"""G7: a provider 429 stays a 429 (it used to become a TypeError), is waited out for at most 10 s, is counted, and the
request still ends in exactly one terminal error event."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import groq
import httpx
import pytest

from app.exceptions.provider import ProviderRateLimitError
from app.request_analyzer.groq_client import GroqAnalysisClient
from app.utils import metrics
from app.utils.rate_limit import MAX_RATE_LIMIT_WAIT_SECONDS, parse_duration_seconds, retry_after_from_headers, retry_after_of
from app.utils.retry import RetryManager, RetryPolicy
from tests.unit.test_request_pipeline import make_event
from tests.unit.test_terminal_events import TERMINAL_TOPICS, FakeProducer, real_pipeline


def rate_limited(headers=None) -> groq.RateLimitError:
    response = httpx.Response(429, headers=headers or {}, request=httpx.Request("POST", "https://api.groq.com/x"))
    return groq.RateLimitError("rate limit", response=response, body=None)


def counted(provider: str, action: str = "raised") -> float:
    return metrics.PROVIDER_RATE_LIMITED_TOTAL.labels(provider=provider, action=action)._value.get()


# --- the TypeError ----------------------------------------------------------------------------------------------------

async def test_the_analysis_client_raises_a_429_not_a_type_error():
    client = GroqAnalysisClient.__new__(GroqAnalysisClient)
    client.model, client.timeout_ms, client.logger = "openai/gpt-oss-20b", 1000, MagicMock()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_k: (_ for _ in ()).throw(rate_limited({"retry-after": "2"})))))

    async def boom(**_k):
        raise rate_limited({"retry-after": "2"})

    client._client.chat.completions.create = boom
    before = counted("groq")
    with pytest.raises(ProviderRateLimitError) as e:
        await client.complete(messages=[{"role": "user", "content": "hi"}])
    assert e.value.status_code == 429 and e.value.provider == "groq" and e.value.retry_after_seconds == 2.0
    assert counted("groq") == before + 1


def test_every_raise_site_builds_the_error_without_a_status_code_argument():
    import pathlib
    import re

    bad = []
    for path in pathlib.Path("app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"ProviderRateLimitError\((?:[^()]|\([^()]*\))*\)", text):
            if "status_code" in m.group(0):
                bad.append(str(path))
    assert bad == []
    err = ProviderRateLimitError("x", provider="gemini")
    assert (err.status_code, err.retry_after_seconds) == (429, None)


# --- what the headers say -----------------------------------------------------------------------------------------------

def test_header_parsing():
    assert parse_duration_seconds("7.66s") == 7.66 and parse_duration_seconds("2m59.56s") == pytest.approx(179.56)
    assert parse_duration_seconds("250ms") == 0.25 and parse_duration_seconds("3") == 3.0 and parse_duration_seconds("1h2m") == 3720
    assert parse_duration_seconds("soon") is None and parse_duration_seconds("") is None and parse_duration_seconds(None) is None
    assert retry_after_from_headers({"Retry-After": "4", "x-ratelimit-reset-tokens": "1s"}) == 4.0  # Retry-After wins
    assert retry_after_from_headers({"x-ratelimit-reset-requests": "6.5s", "x-ratelimit-reset-tokens": "2s"}) == 2.0
    assert retry_after_from_headers({}) is None and retry_after_from_headers(None) is None
    assert retry_after_of(rate_limited({"x-ratelimit-reset-tokens": "3s"})) == 3.0
    assert retry_after_of(ValueError("no response")) is None


# --- the bounded wait -----------------------------------------------------------------------------------------------------

@pytest.fixture
def sleeps(monkeypatch):
    waited = []

    async def fake_sleep(s):
        waited.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return waited


async def test_the_provider_asked_wait_is_honoured(sleeps):
    calls = []

    async def op():
        calls.append(1)
        if len(calls) < 3:
            raise ProviderRateLimitError("429", provider="groq", retry_after_seconds=3.5)
        return "ok"

    assert await RetryManager(RetryPolicy(max_attempts=3, jitter=False)).execute_with_retry(op, "answer", "groq") == "ok"
    assert sleeps == [3.5, 3.5]


async def test_a_wait_longer_than_ten_seconds_fails_at_once(sleeps):
    async def op():
        raise ProviderRateLimitError("429", provider="groq", retry_after_seconds=MAX_RATE_LIMIT_WAIT_SECONDS + 0.1)

    with pytest.raises(ProviderRateLimitError):
        await RetryManager(RetryPolicy(max_attempts=3)).execute_with_retry(op, "answer", "groq")
    assert sleeps == []  # it did not wait at all, and did not retry


async def test_exactly_ten_seconds_is_waited_and_no_hint_uses_the_normal_backoff(sleeps):
    n = []

    async def op():
        n.append(1)
        if len(n) == 1:
            raise ProviderRateLimitError("429", provider="groq", retry_after_seconds=10.0)
        if len(n) == 2:
            raise ProviderRateLimitError("429", provider="groq")
        return "done"

    assert await RetryManager(RetryPolicy(max_attempts=3, jitter=False, initial_delay_ms=100)).execute_with_retry(op) == "done"
    assert sleeps[0] == 10.0 and sleeps[1] == pytest.approx(0.2)  # attempt 2's normal exponential delay


# --- the request still ends cleanly -----------------------------------------------------------------------------------------

async def test_a_rate_limited_answer_ends_in_exactly_one_terminal_error_event():
    prod = FakeProducer()
    pipeline, mocks = real_pipeline(prod)

    async def limited(*_a, **_k):
        raise ProviderRateLimitError("Groq Rate Limit: 429", provider="groq", retry_after_seconds=30)
        yield  # pragma: no cover

    mocks["generation_router"].generate_stream = MagicMock(side_effect=lambda *a, **k: limited())
    await pipeline.run(make_event())
    (ev,) = prod.of(*TERMINAL_TOPICS)
    assert ev["status"] == "error" and ev["error_code"]
    assert "calls" in ev  # the calls made so far are reported (none are billable here)
    assert prod.of("memory.update.requested") == []
