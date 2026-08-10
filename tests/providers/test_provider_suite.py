"""
Provider Suite Tests (Phase 20).
Extended coverage for NVIDIA, Gemini, Groq adapters and the GenerationRouter.
Tests: streaming chunk aggregation, retry triggering, rate limit / timeout
error translation, circuit breaker interaction, provider selection metrics.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.context_window.models import PromptSection, TrimmedPrompt
from app.exceptions.provider import (
    AllProvidersFailedError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.providers.gemini import GeminiAdapter
from app.providers.groq import GroqAdapter
from app.providers.nvidia import NVIDIAAdapter
from app.providers.router import GenerationRouter
from app.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockStreamAdapter:
    """Stub adapter that streams fixed tokens or fails."""

    def __init__(self, provider_name: str, tokens: list[str], should_fail: bool = False):
        self.provider_name = provider_name
        self.tokens = tokens
        self.should_fail = should_fail
        self.call_count = 0

    async def stream(self, messages, params=None, **kwargs):
        self.call_count += 1
        if self.should_fail:
            raise ProviderRateLimitError(f"{self.provider_name} rate limited")
        for t in self.tokens:
            yield t

    async def execute(self, messages, params=None, **kwargs):
        self.call_count += 1
        if self.should_fail:
            raise ProviderTimeoutError(f"{self.provider_name} timed out")
        mock = MagicMock()
        mock.content = "".join(self.tokens)
        mock.provider = self.provider_name
        mock.prompt_tokens = 10
        mock.completion_tokens = 5
        mock.total_tokens = 15
        return mock


def _make_prompt(content: str = "Explain attention") -> TrimmedPrompt:
    return TrimmedPrompt(
        sections=[
            PromptSection(
                name="user_query",
                content=content,
                priority=10,
                token_count=10,
                trimmable=False,
            )
        ],
        messages=[{"role": "user", "content": content}],
        total_tokens=10,
        mode="default",
        target_model="meta/llama-3.3-70b-instruct",
    )


def _make_router(
    primary_tokens: list[str] = None,
    primary_fail: bool = False,
    fallback_tokens: list[str] = None,
    fallback_fail: bool = False,
) -> GenerationRouter:
    cb_nvidia = CircuitBreaker("nvidia", CircuitBreakerConfig(failure_threshold=3))
    cb_gemini = CircuitBreaker("gemini", CircuitBreakerConfig(failure_threshold=3))
    return GenerationRouter(
        nvidia_adapter=_MockStreamAdapter("nvidia", primary_tokens or [], should_fail=primary_fail),
        gemini_adapter=_MockStreamAdapter(
            "gemini", fallback_tokens or [], should_fail=fallback_fail
        ),
        circuit_breakers={"nvidia": cb_nvidia, "gemini": cb_gemini},
    )


# ---------------------------------------------------------------------------
# NVIDIA Adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nvidia_streaming_aggregates_all_chunks():
    """Streaming all tokens from NVIDIAAdapter produces complete response."""
    mock_client = MagicMock()

    async def _stream():
        for piece in ["The", " quick", " brown", " fox"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock(delta=MagicMock(content=piece))]
            yield chunk

    mock_client.chat.completions.create = AsyncMock(return_value=_stream())
    adapter = NVIDIAAdapter(api_key="test_nvidia", client=mock_client)
    tokens = [t async for t in adapter.stream([{"role": "user", "content": "Hello"}])]
    assert "".join(tokens) == "The quick brown fox"


@pytest.mark.asyncio
async def test_nvidia_rate_limit_error_maps_correctly():
    """HTTP 429 from NVIDIA maps to ProviderRateLimitError."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception("HTTP status 429 Too Many Requests")
    )
    adapter = NVIDIAAdapter(api_key="test_nvidia", client=mock_client)
    with pytest.raises(ProviderRateLimitError):
        await adapter.execute([{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_nvidia_timeout_error_maps_correctly():
    """Timeout from NVIDIA maps to ProviderTimeoutError."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception("Request timed out after 30s")
    )
    adapter = NVIDIAAdapter(api_key="test_nvidia", client=mock_client)
    with pytest.raises(ProviderTimeoutError):
        await adapter.execute([{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_nvidia_adapter_execute_returns_usage_metrics():
    """NVIDIAAdapter.execute() returns prompt and completion token counts."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Done"
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 42
    mock_usage.completion_tokens = 18
    mock_usage.total_tokens = 60
    mock_resp = MagicMock(choices=[mock_choice], usage=mock_usage)
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    adapter = NVIDIAAdapter(api_key="key", client=mock_client)
    result = await adapter.execute([{"role": "user", "content": "Test"}])

    assert result.prompt_tokens == 42
    assert result.completion_tokens == 18
    assert result.total_tokens == 60


# ---------------------------------------------------------------------------
# Gemini Adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_streaming_aggregates_all_chunks():
    """Streaming all tokens from GeminiAdapter produces complete response."""
    mock_client = MagicMock()

    async def _gemini_stream():
        for piece in ["Gemini", " is", " fast"]:
            chunk = MagicMock()
            chunk.text = piece
            yield chunk

    mock_client.aio.models.generate_content_stream = AsyncMock(return_value=_gemini_stream())
    adapter = GeminiAdapter(api_key="test_gemini", client=mock_client)
    tokens = [t async for t in adapter.stream([{"role": "user", "content": "Hello"}])]
    assert "".join(tokens) == "Gemini is fast"


@pytest.mark.asyncio
async def test_gemini_execute_system_prompt_separation():
    """GeminiAdapter correctly handles system + user message separation."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "OK"
    mock_resp.usage_metadata.prompt_token_count = 20
    mock_resp.usage_metadata.candidates_token_count = 5
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    adapter = GeminiAdapter(api_key="test_gemini", client=mock_client)
    result = await adapter.execute(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
    )
    assert result.content == "OK"
    assert result.provider == "gemini"


# ---------------------------------------------------------------------------
# Groq Adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_groq_adapter_json_completions_parse_correctly():
    """GroqAdapter complete() returns structured JSON for analysis tasks."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"mode": "deep_research", "need_web_search": true}'
    mock_resp = MagicMock(choices=[mock_choice])
    mock_resp.usage.prompt_tokens = 25
    mock_resp.usage.completion_tokens = 12
    mock_resp.usage.total_tokens = 37
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    adapter = GroqAdapter(api_key="key", client=mock_client)
    result = await adapter.execute(
        messages=[{"role": "user", "content": "Classify this"}],
        params={"response_format": {"type": "json_object"}},
    )
    assert '"mode": "deep_research"' in result.content
    assert result.total_tokens == 37


# ---------------------------------------------------------------------------
# GenerationRouter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_streams_primary_nvidia_by_default():
    """GenerationRouter streams from NVIDIA when healthy."""
    router = _make_router(primary_tokens=["Hello", " World"])
    tokens = [t async for t in router.generate_stream(_make_prompt())]
    assert "".join(tokens) == "Hello World"


@pytest.mark.asyncio
async def test_router_falls_back_to_gemini_on_nvidia_failure():
    """GenerationRouter falls back to Gemini when NVIDIA raises an error."""
    router = _make_router(primary_fail=True, fallback_tokens=["Fallback", " OK"])
    tokens = [t async for t in router.generate_stream(_make_prompt())]
    assert "".join(tokens) == "Fallback OK"


@pytest.mark.asyncio
async def test_router_raises_when_all_providers_fail():
    """GenerationRouter raises AllProvidersFailedError when both adapters fail."""
    router = _make_router(primary_fail=True, fallback_fail=True)
    with pytest.raises(AllProvidersFailedError):
        async for _ in router.generate_stream(_make_prompt()):
            pass


@pytest.mark.asyncio
async def test_router_uses_gemini_when_nvidia_circuit_open():
    """GenerationRouter skips NVIDIA if its circuit breaker is open."""
    cb_nvidia = CircuitBreaker("nvidia", CircuitBreakerConfig(failure_threshold=1))
    cb_nvidia.trip()  # force OPEN
    cb_gemini = CircuitBreaker("gemini", CircuitBreakerConfig(failure_threshold=3))

    nvidia_adapter = _MockStreamAdapter("nvidia", ["Should not appear"])
    gemini_adapter = _MockStreamAdapter("gemini", ["Gemini", " direct"])

    router = GenerationRouter(
        nvidia_adapter=nvidia_adapter,
        gemini_adapter=gemini_adapter,
        circuit_breakers={"nvidia": cb_nvidia, "gemini": cb_gemini},
    )
    tokens = [t async for t in router.generate_stream(_make_prompt())]
    assert "".join(tokens) == "Gemini direct"
    assert nvidia_adapter.call_count == 0
