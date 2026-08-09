"""
Unit Tests for Provider Adapters (NVIDIAAdapter, GeminiAdapter, GroqAdapter) and GenerationRouter.
Implements test coverage for Phase 15 as per LLD v2.0 Section 19 and Section 20.
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.context_window.models import TrimmedPrompt
from app.exceptions.provider import (
    AllProvidersFailedError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.prompts.schemas import PromptSection
from app.providers.base import BaseProviderAdapter
from app.providers.gemini import GeminiAdapter
from app.providers.groq import GroqAdapter
from app.providers.models import GenerationResponse
from app.providers.nvidia import NVIDIAAdapter
from app.providers.router import GenerationRouter
from app.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# --- Helper Mock Adapters ---


class MockStreamAdapter(BaseProviderAdapter):
    def __init__(self, provider_name: str, tokens: list[str], should_fail: bool = False):
        super().__init__(provider_name=provider_name, model_name=f"{provider_name}-mock")
        self.tokens = tokens
        self.should_fail = should_fail
        self.call_count = 0

    async def execute(self, messages, params=None) -> GenerationResponse:
        self.call_count += 1
        if self.should_fail:
            raise ProviderError(f"{self.provider_name} failed", provider=self.provider_name)
        return GenerationResponse(
            content="".join(self.tokens),
            provider=self.provider_name,
            model=self.model_name,
            total_tokens=len(self.tokens),
        )

    async def stream(self, messages, params=None) -> AsyncIterator[str]:
        self.call_count += 1
        if self.should_fail:
            raise ProviderError(f"{self.provider_name} failed", provider=self.provider_name)
        for token in self.tokens:
            yield token


# --- 1. NVIDIAAdapter Tests ---


@pytest.mark.asyncio
async def test_nvidia_adapter_execute_and_stream():
    """Verify NVIDIAAdapter executes and streams via AsyncOpenAI client."""
    mock_client = MagicMock()

    # Setup mock non-streaming response
    mock_choice = MagicMock()
    mock_choice.message.content = "NVIDIA generated response"
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 50
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 70
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    adapter = NVIDIAAdapter(api_key="test_key", client=mock_client)
    res = await adapter.execute([{"role": "user", "content": "Hello"}])

    assert res.content == "NVIDIA generated response"
    assert res.provider == "nvidia"
    assert res.total_tokens == 70

    # Setup mock streaming response
    async def mock_stream_gen():
        for piece in ["Hello", " from", " NVIDIA"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock(delta=MagicMock(content=piece))]
            yield chunk

    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream_gen())
    collected = []
    async for token in adapter.stream([{"role": "user", "content": "Hello"}]):
        collected.append(token)

    assert "".join(collected) == "Hello from NVIDIA"


@pytest.mark.asyncio
async def test_nvidia_adapter_error_translation():
    """Verify NVIDIAAdapter translates errors to ProviderError types."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception("HTTP 429 Rate limit exceeded")
    )
    adapter = NVIDIAAdapter(api_key="test_key", client=mock_client)

    with pytest.raises(ProviderRateLimitError):
        await adapter.execute([{"role": "user", "content": "Hello"}])


# --- 2. GeminiAdapter Tests ---


@pytest.mark.asyncio
async def test_gemini_adapter_execute_and_stream():
    """Verify GeminiAdapter executes and streams via Google GenAI SDK client."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Gemini generated response"
    mock_resp.usage_metadata.prompt_token_count = 40
    mock_resp.usage_metadata.candidates_token_count = 15
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    adapter = GeminiAdapter(api_key="test_key", client=mock_client)
    res = await adapter.execute(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Explain transformers."},
        ]
    )

    assert res.content == "Gemini generated response"
    assert res.provider == "gemini"
    assert res.total_tokens == 55

    # Streaming test
    async def mock_gemini_stream():
        for piece in ["Transformers", " use", " self-attention."]:
            chunk = MagicMock()
            chunk.text = piece
            yield chunk

    mock_client.aio.models.generate_content_stream = AsyncMock(return_value=mock_gemini_stream())
    collected = []
    async for token in adapter.stream([{"role": "user", "content": "Hello"}]):
        collected.append(token)

    assert "".join(collected) == "Transformers use self-attention."


@pytest.mark.asyncio
async def test_gemini_adapter_error_translation():
    """Verify GeminiAdapter translates timeout errors to ProviderTimeoutError."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=Exception("Deadline exceeded during call timeout")
    )
    adapter = GeminiAdapter(api_key="test_key", client=mock_client)

    with pytest.raises(ProviderTimeoutError):
        await adapter.execute([{"role": "user", "content": "Hello"}])


# --- 3. GroqAdapter Tests ---


@pytest.mark.asyncio
async def test_groq_adapter_execute():
    """Verify GroqAdapter executes JSON completions."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"mode": "smart", "need_web_search": true}'
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 30
    mock_resp.usage.completion_tokens = 10
    mock_resp.usage.total_tokens = 40
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    adapter = GroqAdapter(api_key="test_key", client=mock_client)
    res = await adapter.execute(
        messages=[{"role": "user", "content": "Analyze request"}],
        params={"response_format": {"type": "json_object"}},
    )

    assert '{"mode": "smart"' in res.content
    assert res.provider == "groq"


# --- 4. GenerationRouter Tests ---


@pytest.mark.asyncio
async def test_generation_router_primary_route():
    """Verify GenerationRouter routes to primary NVIDIA adapter when healthy."""
    primary = MockStreamAdapter(provider_name="nvidia", tokens=["Primary", " response"])
    fallback = MockStreamAdapter(provider_name="gemini", tokens=["Fallback", " response"])
    cb_nvidia = CircuitBreaker("nvidia", CircuitBreakerConfig(failure_threshold=2))
    cb_gemini = CircuitBreaker("gemini", CircuitBreakerConfig(failure_threshold=2))

    router = GenerationRouter(
        nvidia_adapter=primary,
        gemini_adapter=fallback,
        circuit_breakers={"nvidia": cb_nvidia, "gemini": cb_gemini},
    )

    prompt = TrimmedPrompt(
        sections=[
            PromptSection(
                name="user_query", content="Hello", priority=10, token_count=5, trimmable=False
            )
        ],
        messages=[{"role": "user", "content": "Hello"}],
        total_tokens=5,
        mode="default",
        target_model="meta/llama-3.1-70b-instruct",
    )

    tokens = []
    async for t in router.generate_stream(prompt):
        tokens.append(t)

    assert "".join(tokens) == "Primary response"
    assert primary.call_count == 1
    assert fallback.call_count == 0


@pytest.mark.asyncio
async def test_generation_router_fallback_on_primary_failure():
    """Verify GenerationRouter falls back to Gemini when NVIDIA fails."""
    primary = MockStreamAdapter(provider_name="nvidia", tokens=[], should_fail=True)
    fallback = MockStreamAdapter(provider_name="gemini", tokens=["Fallback", " successful"])
    cb_nvidia = CircuitBreaker("nvidia", CircuitBreakerConfig(failure_threshold=2))
    cb_gemini = CircuitBreaker("gemini", CircuitBreakerConfig(failure_threshold=2))

    router = GenerationRouter(
        nvidia_adapter=primary,
        gemini_adapter=fallback,
        circuit_breakers={"nvidia": cb_nvidia, "gemini": cb_gemini},
    )

    prompt = TrimmedPrompt(
        sections=[],
        messages=[{"role": "user", "content": "Hello"}],
        total_tokens=5,
        mode="default",
    )

    tokens = []
    async for t in router.generate_stream(prompt):
        tokens.append(t)

    assert "".join(tokens) == "Fallback successful"
    assert primary.call_count == 1
    assert fallback.call_count == 1


@pytest.mark.asyncio
async def test_generation_router_all_providers_failed():
    """Verify GenerationRouter raises AllProvidersFailedError when both fail."""
    primary = MockStreamAdapter(provider_name="nvidia", tokens=[], should_fail=True)
    fallback = MockStreamAdapter(provider_name="gemini", tokens=[], should_fail=True)
    cb_nvidia = CircuitBreaker("nvidia", CircuitBreakerConfig(failure_threshold=2))
    cb_gemini = CircuitBreaker("gemini", CircuitBreakerConfig(failure_threshold=2))

    router = GenerationRouter(
        nvidia_adapter=primary,
        gemini_adapter=fallback,
        circuit_breakers={"nvidia": cb_nvidia, "gemini": cb_gemini},
    )

    prompt = TrimmedPrompt(
        sections=[],
        messages=[{"role": "user", "content": "Hello"}],
        total_tokens=5,
    )

    with pytest.raises(AllProvidersFailedError):
        async for _ in router.generate_stream(prompt):
            pass
