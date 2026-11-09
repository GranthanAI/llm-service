"""P5a: billing model registry and group routing (Best only tries Best, never escalate, legacy untouched, context cap)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.config.settings import LLMServiceConfig
from app.context_window.fit import build_group_fitter
from app.context_window.models import PromptSection, TrimmedPrompt
from app.exceptions.provider import AllProvidersFailedError
from app.models.billing import BillingGrant
from app.models.request import ChatMessageCreatedEvent
from app.pipeline.request_pipeline import RequestPipeline
from app.providers.group_chain import GroupChainBuilder
from app.providers.registry import DEFAULT_GROUP_ORDER, ModelRegistry, RegistryError

REGISTRY_FILE = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


def spec(provider, group, model=None, enabled=True, window=200_000, out=8192):
    return {"provider": provider, "provider_model": model or "pm", "group": group,
            "context_window": window, "max_output_tokens": out, "enabled": enabled}


def registry(all_enabled=True):
    raw = ModelRegistryFile()
    if all_enabled:
        for m in raw["models"].values():
            m["enabled"] = True
    return ModelRegistry.from_dict(raw)


def ModelRegistryFile():
    import yaml
    return yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))


ALL = lambda provider: True            # noqa: E731  every provider has credentials


def ids(chain):
    return [m.id for m in chain]


# --- registry file --------------------------------------------------------------------------------------------

def test_the_shipped_registry_loads_and_its_group_order_is_the_default():
    reg = ModelRegistry.load(REGISTRY_FILE)
    assert {g: [m.id for m in reg.group_members(g)] for g in DEFAULT_GROUP_ORDER} == DEFAULT_GROUP_ORDER


def test_unknown_provider_is_a_startup_error():
    with pytest.raises(RegistryError, match="unknown provider"):
        ModelRegistry.from_dict({"models": {"m": spec("mistral", "fast")}})


def test_unknown_group_is_a_startup_error():
    with pytest.raises(RegistryError, match="unknown group"):
        ModelRegistry.from_dict({"models": {"m": spec("groq", "turbo")}})


def test_duplicate_id_is_a_startup_error(tmp_path):
    f = tmp_path / "models.yaml"
    f.write_text("models:\n  a: {provider: groq, provider_model: x, group: fast, context_window: 8000, max_output_tokens: 10}\n"
                 "  a: {provider: groq, provider_model: y, group: fast, context_window: 8000, max_output_tokens: 10}\n")
    # PyYAML keeps the last duplicate key silently; the loader must still not accept two entries for one provider model
    raw = {"models": {"a": spec("groq", "fast", "same"), "b": spec("groq", "fast", "same")}}
    with pytest.raises(RegistryError, match="duplicate"):
        ModelRegistry.from_dict(raw)


def test_group_order_naming_a_foreign_or_unknown_model_is_a_startup_error():
    raw = {"models": {"a": spec("groq", "fast", "a")}, "group_order": {"fast": ["nope"]}}
    with pytest.raises(RegistryError, match="not in `models`"):
        ModelRegistry.from_dict(raw)
    raw = {"models": {"a": spec("groq", "fast", "a"), "b": spec("groq", "smart", "b")}, "group_order": {"fast": ["b"]}}
    with pytest.raises(RegistryError, match="whose group"):
        ModelRegistry.from_dict(raw)


def test_missing_file_is_a_startup_error(tmp_path):
    with pytest.raises(RegistryError):
        ModelRegistry.load(tmp_path / "absent.yaml")


def test_unpriced_models_in_a_group_warn_and_are_never_routed(caplog):
    raw = {"models": {"qwen/qwen3.8-27b": spec("groq", "smart", "qwen/qwen3.8-27b"),
                      "ok": spec("groq", "smart", "ok")}}
    reg = ModelRegistry.from_dict(raw)
    assert ids(reg.chain_for(BillingGrant(granted_group="smart"), ALL)) == ["ok"]


# --- chain building -------------------------------------------------------------------------------------------

def test_best_only_tries_best_models_in_group_order():
    chain = registry().chain_for(BillingGrant(granted_group="best"), ALL)
    assert ids(chain) == DEFAULT_GROUP_ORDER["best"]
    assert {m.group for m in chain} == {"best"}


def test_ultra_is_never_reached_from_best_and_fast_never_escalates():
    for group, order in DEFAULT_GROUP_ORDER.items():
        chain = registry().chain_for(BillingGrant(granted_group=group, granted_model=order[-1]), ALL)
        assert {m.group for m in chain} == {group} and set(ids(chain)) == set(order)
    assert not set(DEFAULT_GROUP_ORDER["ultra"]) & set(ids(registry().chain_for(BillingGrant(granted_group="best"), ALL)))


def test_explicit_model_goes_first_then_the_rest_of_its_group():
    chain = registry().chain_for(BillingGrant(granted_group="smart", granted_model="gemini-2.5-flash"), ALL)
    assert ids(chain) == ["gemini-2.5-flash", "openai/gpt-oss-120b", "claude-haiku-5-5", "gpt-6-luna", "gemini-3.1-flash-lite"]


def test_auto_uses_the_group_order():
    chain = registry().chain_for(BillingGrant(granted_group="smart", granted_model=None), ALL)
    assert ids(chain) == DEFAULT_GROUP_ORDER["smart"]


def test_a_granted_model_outside_the_granted_group_is_ignored_not_escalated():
    chain = registry().chain_for(BillingGrant(granted_group="best", granted_model="claude-opus-5-5"), ALL)
    assert ids(chain) == DEFAULT_GROUP_ORDER["best"]


def test_disabled_models_and_providers_without_credentials_are_skipped():
    reg = ModelRegistry.load(REGISTRY_FILE)  # gpt-6-luna and gemini-3.1-flash-lite ship disabled until verified
    chain = reg.chain_for(BillingGrant(granted_group="smart"), lambda p: p != "anthropic")
    assert "gpt-6-luna" not in ids(chain) and "claude-haiku-5-5" not in ids(chain)
    assert ids(chain) == ["openai/gpt-oss-120b", "gemini-2.5-flash"]


def test_a_group_with_no_usable_model_is_an_empty_chain():
    assert registry().chain_for(BillingGrant(granted_group="ultra"), lambda p: False) == []


def test_no_grant_or_null_group_means_the_legacy_chain():
    reg = registry()
    assert reg.chain_for(None, ALL) is None
    assert reg.chain_for(BillingGrant(granted_group=None, hold_id="h"), ALL) is None


# --- the chain actually tried ------------------------------------------------------------------------------------

class FakeAdapter:
    def __init__(self, name, fail=False, tokens=("hi",)):
        self.name, self.fail, self.tokens, self.calls = name, fail, tokens, 0
        self.provider_name, self.model_name = name, name

    async def stream(self, messages, params=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"{self.name} down")
        for t in self.tokens:
            yield t

    async def execute(self, messages, params=None):
        raise NotImplementedError


def prompt(n=1):
    return TrimmedPrompt(
        sections=[PromptSection(name="q", content="x", priority=1, token_count=1, trimmable=False)],
        messages=[{"role": "user", "content": "x"}], total_tokens=n, mode="default", target_model="m")


def builder(adapters, reg=None, config=None, manager=None):
    config = config or LLMServiceConfig()
    manager = manager or MagicMock()
    # the fitter is not under test here: let every prompt through
    return GroupChainBuilder(reg or registry(), config, manager, lambda s: adapters.get(s.id), ALL)


async def drain(chain):
    return [t async for t in chain.generate_stream(prompt(), {})]


async def test_best_falls_back_inside_best_only_and_reports_the_model_that_answered(monkeypatch):
    adapters = {"claude-sonnet-5-5": FakeAdapter("sonnet", fail=True), "gpt-6.1-sol": FakeAdapter("sol"),
                "gemini-3.1-pro": FakeAdapter("g31pro"), "claude-opus-5-5": FakeAdapter("opus")}
    monkeypatch.setattr("app.providers.group_chain.build_group_fitter", lambda *a, **k: (lambda p, n, e: p))
    chain = builder(adapters).build(BillingGrant(granted_group="best"))
    params: dict = {}
    assert [t async for t in chain.generate_stream(prompt(), params)] == ["hi"]
    assert params["_route"]["model_id"] == "gpt-6.1-sol" and params["_route"]["fallback_used"] is True
    assert adapters["claude-opus-5-5"].calls == 0          # never escalated


async def test_explicit_model_falls_back_inside_its_group_only(monkeypatch):
    adapters = {i: FakeAdapter(i, fail=True) for i in ("openai/gpt-oss-120b", "claude-haiku-5-5", "gemini-2.5-flash",
                                                      "gpt-6-luna", "gemini-3.1-flash-lite")}
    adapters.update({"claude-sonnet-5-5": FakeAdapter("sonnet"), "openai/gpt-oss-20b": FakeAdapter("fast20b")})
    monkeypatch.setattr("app.providers.group_chain.build_group_fitter", lambda *a, **k: (lambda p, n, e: p))
    chain = builder(adapters).build(BillingGrant(granted_group="smart", granted_model="gemini-2.5-flash"))
    with pytest.raises(AllProvidersFailedError):
        await drain(chain)
    assert adapters["gemini-2.5-flash"].calls == 1                       # the explicit model was tried first
    assert adapters["claude-sonnet-5-5"].calls == 0 and adapters["openai/gpt-oss-20b"].calls == 0   # neither up nor down


async def test_whole_group_failure_raises_after_trying_every_group_model_once(monkeypatch):
    adapters = {i: FakeAdapter(i, fail=True) for i in DEFAULT_GROUP_ORDER["best"]}
    adapters["claude-opus-5-5"] = FakeAdapter("opus")
    monkeypatch.setattr("app.providers.group_chain.build_group_fitter", lambda *a, **k: (lambda p, n, e: p))
    chain = builder(adapters).build(BillingGrant(granted_group="best"))
    with pytest.raises(AllProvidersFailedError):
        await drain(chain)
    assert [a.calls for k, a in adapters.items() if k in DEFAULT_GROUP_ORDER["best"]] == [1, 1, 1]
    assert adapters["claude-opus-5-5"].calls == 0


def test_a_group_with_no_adapters_raises_instead_of_leaving_the_group():
    with pytest.raises(AllProvidersFailedError, match="granted group"):
        builder({}).build(BillingGrant(granted_group="best"))


def test_no_billing_block_builds_no_group_chain():
    assert builder({}).build(None) is None
    assert builder({}).build(BillingGrant(hold_id="h")) is None


# --- event contract ---------------------------------------------------------------------------------------------

BASE = {"message_id": "m", "response_message_id": "r", "conversation_id": "c", "user_id": "u", "content": "hi"}


def test_schema_2_3_without_a_billing_block_and_2_4_with_one_both_parse():
    old = ChatMessageCreatedEvent.model_validate({**BASE, "schema_version": "2.3"})
    assert old.billing is None
    new = ChatMessageCreatedEvent.model_validate({**BASE, "schema_version": "2.4", "future_field": 1, "billing": {
        "hold_id": "0193", "requested_model": None, "granted_group": "best", "granted_model": "claude-sonnet-5-5",
        "max_context_tokens": 128000, "funded_by": "plan", "downgraded": False, "enforcement": "enforce", "extra": 1}})
    assert new.billing.granted_group == "best" and new.billing.max_context_tokens == 128000 and new.billing.hold_id == "0193"


# --- pipeline wiring ----------------------------------------------------------------------------------------------

def pipeline(group_chains):
    p = RequestPipeline.__new__(RequestPipeline)
    p.group_chains = group_chains
    return p


def test_legacy_requests_never_touch_group_routing():
    ctx = SimpleNamespace(billing=None)
    assert pipeline(None)._group_chain(ctx) is None
    assert pipeline(MagicMock())._group_chain(SimpleNamespace(billing=BillingGrant(granted_group=None))) is None


def test_a_grant_is_never_answered_by_the_legacy_chain_when_group_routing_is_not_configured():
    with pytest.raises(AllProvidersFailedError, match="not configured"):
        pipeline(None)._group_chain(SimpleNamespace(billing=BillingGrant(granted_group="best")))


# --- context cap ---------------------------------------------------------------------------------------------------

def test_effective_budget_is_the_smallest_of_grant_cap_model_window_and_policy():
    from app.context_window.budget_policy import resolve_context_budget
    from app.context_window.models import ModelLimits
    cfg = LLMServiceConfig()
    limits = ModelLimits(model="m", context_window=200_000, max_output_tokens=8192, provider="anthropic")
    uncapped = resolve_context_budget(cfg, "anthropic", "m", limits, requested_output_tokens=8192)
    capped = resolve_context_budget(cfg, "anthropic", "m", limits, requested_output_tokens=8192, input_cap=16_000)
    assert capped.input_budget == 16_000 < uncapped.input_budget
    assert resolve_context_budget(cfg, "anthropic", "m", limits, requested_output_tokens=8192,
                                  input_cap=10_000_000).input_budget == uncapped.input_budget  # a cap is never a floor
    small = ModelLimits(model="m", context_window=12_000, max_output_tokens=2048, provider="anthropic")
    assert resolve_context_budget(cfg, "anthropic", "m", small, requested_output_tokens=2048,
                                  input_cap=50_000).input_budget < 12_000


def test_group_fitter_trims_through_the_context_manager_to_the_grant_cap():
    cfg = LLMServiceConfig()
    manager = MagicMock()
    manager.token_counter.count_messages.side_effect = lambda msgs, provider: msgs[0]["tokens"]
    calls = []

    def manage(composed, target_model, custom_input_budget, max_output_tokens):
        calls.append(custom_input_budget)
        return SimpleNamespace(messages=[{"tokens": custom_input_budget}], total_tokens=custom_input_budget,
                               trimmed_sections=["graph_context"], sections=[], mode="default")

    manager.manage.side_effect = manage
    fit = build_group_fitter(cfg, manager, MagicMock())
    entry = SimpleNamespace(spec=SimpleNamespace(id="claude-sonnet-5-5", provider="anthropic", provider_model="claude-sonnet-5-5",
                                                 context_window=200_000, max_output_tokens=8192),
                            input_cap=20_000)
    big = SimpleNamespace(messages=[{"tokens": 60_000}], sections=[], total_tokens=60_000, mode="default")
    out = fit(big, "anthropic:claude-sonnet-5-5", entry)
    assert calls and calls[0] == 20_000 and out.messages[0]["tokens"] <= 20_000
    small = SimpleNamespace(messages=[{"tokens": 1_000}], sections=[], total_tokens=1_000, mode="default")
    assert fit(small, "anthropic:claude-sonnet-5-5", entry) is small            # already within the cap: untouched
