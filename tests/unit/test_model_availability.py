"""M2: GET /internal/models/availability: enabled AND credentials AND adapter AND breaker closed."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.internal.models import router
from app.providers.availability import build_availability
from app.providers.registry import ModelRegistry
from app.utils.circuit_breaker import CircuitState

REGISTRY = ModelRegistry.load(Path(__file__).resolve().parents[2] / "config" / "models.yaml")


def chains(keys=("groq", "google"), adapters=True, open_breakers=()):
    def breaker(name):
        return SimpleNamespace(state=CircuitState.OPEN if name in open_breakers else CircuitState.CLOSED)

    return SimpleNamespace(registry=REGISTRY, _has_credentials=lambda p: p in keys,
                           can_serve=lambda spec: adapters, _breaker_factory=breaker)


def by_id(payload):
    return {m["id"]: m for m in payload["models"]}


def test_models_without_keys_are_unavailable_with_a_reason():
    m = by_id(asyncio.run(build_availability(chains())))
    assert m["openai/gpt-oss-120b"] == {"id": "openai/gpt-oss-120b", "available": True, "reason": ""}
    assert m["gemini-3.8-flash"]["available"] is True
    for model in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "gpt-6-luna", "gpt-6.1-sol"):
        assert m[model] == {"id": model, "available": False, "reason": "no_credentials"}


def test_all_keys_present_makes_every_enabled_model_available_and_disabled_stays_out():
    m = by_id(asyncio.run(build_availability(chains(keys=("groq", "google", "anthropic", "openai")))))
    assert m["claude-haiku-5-5"]["available"] and m["gpt-6-luna"]["available"]
    assert m["gemini-3.1-pro"] == {"id": "gemini-3.1-pro", "available": False, "reason": "disabled"}
    assert m["gpt-6-astra"]["reason"] == "disabled"


def test_no_adapter_and_open_breaker():
    assert by_id(asyncio.run(build_availability(chains(adapters=False))))["openai/gpt-oss-20b"]["reason"] == "no_adapter"
    open_ = by_id(asyncio.run(build_availability(chains(open_breakers=("groq:openai/gpt-oss-120b",)))))
    assert open_["openai/gpt-oss-120b"] == {"id": "openai/gpt-oss-120b", "available": False, "reason": "breaker_open"}
    assert open_["openai/gpt-oss-20b"]["available"] is True


def app(token="s3cret", group_chains=None):
    a = FastAPI()
    a.include_router(router, prefix="/internal")
    container = SimpleNamespace(config=SimpleNamespace(internal_service_token=SecretStr(token), gemini_api_key=SecretStr("")), group_chains=group_chains or chains())
    from app.api.dependencies import get_container
    a.dependency_overrides[get_container] = lambda: container
    return TestClient(a)


def test_endpoint_needs_the_service_token_and_returns_the_shape():
    c = app()
    assert c.get("/internal/models/availability").status_code == 401
    assert c.get("/internal/models/availability", headers={"X-Service-Token": "nope"}).status_code == 401
    r = c.get("/internal/models/availability", headers={"X-Service-Token": "s3cret"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"models", "checked_at"} and {"id", "available", "reason"} == set(body["models"][0])


def test_an_unset_token_refuses_everyone():
    assert app(token="").get("/internal/models/availability", headers={"X-Service-Token": ""}).status_code == 401


# --- C2: the provider must still list the id --------------------------------------------------------------------------

from app.providers.provider_models import ProviderModels  # noqa: E402


def with_provider(listed=None, retired=()):
    async def fetch(key):
        if listed is None:
            raise ConnectionError("models.list down")
        return set(listed)

    pm = ProviderModels("key", fetch)
    for r in retired:
        pm.mark_retired(r)
    return pm


def avail(pm, keys=("groq", "google")):
    return by_id(asyncio.run(build_availability(chains(keys=keys), pm)))


def test_a_google_id_the_provider_no_longer_lists_is_unavailable_as_retired():
    m = avail(with_provider(listed={"gemini-3.1-flash-lite", "gemini-3.8-flash"}))
    assert m["gemini-3.1-flash-lite"]["available"] is True
    assert m["gemini-3.5-flash-lite"] == {"id": "gemini-3.5-flash-lite", "available": False, "reason": "retired by provider"}
    assert m["openai/gpt-oss-120b"]["available"] is True          # Groq ids are not checked against Google's list


def test_a_404_seen_at_runtime_retires_an_id_even_while_it_is_still_listed():
    m = avail(with_provider(listed={"gemini-3.1-flash-lite", "gemini-3.8-flash"}, retired=["gemini-3.8-flash"]))
    assert m["gemini-3.8-flash"]["reason"] == "retired by provider"


def test_when_the_list_cannot_be_read_it_fails_open():
    m = avail(with_provider(listed=None))
    assert m["gemini-3.1-flash-lite"]["available"] is True and m["gemini-3.5-flash-lite"]["available"] is True


def test_the_gemini_adapter_marks_a_404_as_retired():
    from app.providers.gemini import GeminiAdapter
    from app.providers.provider_models import provider_models
    from app.exceptions.provider import ProviderError

    adapter = GeminiAdapter(api_key="k", model="gemini-2.5-flash-lite")
    with pytest.raises(ProviderError):
        adapter._handle_exception(Exception("404 NOT_FOUND. This model is no longer available to new users"))
    assert provider_models.is_retired("gemini-2.5-flash-lite")
    provider_models._retired.clear()
