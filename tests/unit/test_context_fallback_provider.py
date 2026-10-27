"""When the first provider's (small, rate-limit-derived) budget cannot take the prompt, the next provider in the chain is used."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.config.settings import LLMServiceConfig
from app.exceptions.analysis import ContextOverflowError
from app.pipeline.request_pipeline import RequestPipeline


def _pipeline(entry_names):
    p = RequestPipeline.__new__(RequestPipeline)          # only the fit step is under test
    p.config = LLMServiceConfig()
    p.logger = MagicMock()
    p.context_window_manager = p.prompt_builder = MagicMock()
    p.generation_router = SimpleNamespace(entries=[SimpleNamespace(name=n) for n in entry_names])
    return p


def _fit(p, fake):
    with patch("app.pipeline.request_pipeline.fit_prompt", side_effect=fake):
        return p._fit_to_first_fitting_provider(result=MagicMock(), ctx=MagicMock(), composed=MagicMock())


def test_falls_through_to_gemini_when_groq_budget_overflows():
    calls = []

    def fake(**kw):
        calls.append(kw["provider"])
        if kw["provider"] == "groq":
            raise ContextOverflowError("needs 5753, 5752 fit")
        return "TRIMMED", SimpleNamespace(tokens_after=9000)

    provider, model, _max, trimmed, fit = _fit(_pipeline(["groq:openai/gpt-oss-120b", "gemini:g"]), fake)
    assert calls == ["groq", "gemini"] and provider == "gemini" and trimmed == "TRIMMED" and fit.tokens_after == 9000


def test_first_provider_is_used_when_it_fits():
    calls = []
    provider, *_ = _fit(_pipeline(["groq:m", "gemini:g"]), lambda **kw: (calls.append(kw["provider"]) or "T", SimpleNamespace(tokens_after=1)))
    assert provider == "groq" and calls == ["groq"]


def test_raises_only_when_no_provider_can_take_the_prompt():
    def fake(**kw):
        raise ContextOverflowError("too big")

    with pytest.raises(ContextOverflowError):
        _fit(_pipeline(["groq:m", "gemini:g"]), fake)
