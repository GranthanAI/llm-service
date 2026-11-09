"""P5f: models.yaml group membership matches billing's catalog snapshot; the periodic catalog check."""

import json
from pathlib import Path

from app.providers.catalog_check import CATALOG_MODEL_MISSING, CatalogChecker, compare
from app.providers.registry import DEFAULT_GROUP_ORDER, ModelRegistry

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = json.loads((ROOT / "tests" / "fixtures" / "billing_models.json").read_text(encoding="utf-8"))["models"]
REGISTRY = ModelRegistry.load(ROOT / "config" / "models.yaml")
ALL = lambda provider: True       # noqa: E731


def test_models_yaml_group_membership_equals_billings_catalog_snapshot():
    """Fails when config/models.yaml and billing's plans.yaml model_groups drift apart (rebuild the fixture from
    billing-service/config/plans.yaml or GET /billing/v1/models when billing's catalog changes on purpose)."""
    catalog = {m["id"]: m["group"] for m in SNAPSHOT}
    registry = {m.id: m.group for m in REGISTRY.models.values()}
    assert registry == catalog


def test_group_order_covers_exactly_the_catalog_and_the_default_order_is_the_documented_one():
    for group, ids in DEFAULT_GROUP_ORDER.items():
        assert sorted(ids) == sorted(m["id"] for m in SNAPSHOT if m["group"] == group)
        assert [m.id for m in REGISTRY.group_members(group)] == ids


def test_providers_in_the_registry_match_billings_provider_names():
    assert {m.id: m.provider for m in REGISTRY.models.values()} == {m["id"]: m["provider"] for m in SNAPSHOT}


def test_compare_reports_each_kind_of_mismatch():
    catalog = [dict(m) for m in SNAPSHOT]
    problems = compare(catalog, REGISTRY, lambda p: p != "anthropic", lambda spec: spec.id != "gpt-6-luna")
    assert set(problems["no_credentials"]) == {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"}
    assert problems["disabled"] == ["gemini-3.1-pro"]                  # shipped disabled: gemini-3.1-pro-preview unverified
    assert problems["no_adapter"] == ["gpt-6-luna"]
    catalog.append({"id": "brand-new", "group": "fast", "provider": "groq", "label": "x", "sort_order": 99})
    catalog[0]["group"] = "ultra"
    problems = compare(catalog, REGISTRY, ALL, lambda s: True)
    assert problems["not_in_registry"] == ["brand-new"] and problems["group_mismatch"] == [catalog[0]["id"]]
    assert "not_in_catalog" not in problems
    assert compare(catalog[2:], REGISTRY, ALL, lambda s: True)["not_in_catalog"] == [catalog[0]["id"], catalog[1]["id"]]


async def test_check_once_sets_and_clears_the_gauge_and_survives_billing_being_down():
    state = {"catalog": [dict(m) for m in SNAPSHOT], "down": False}

    async def fetch(url):
        if state["down"]:
            raise ConnectionError("billing down")
        return state["catalog"]

    checker = CatalogChecker("http://billing", REGISTRY, lambda p: p != "openai", lambda s: True, fetch=fetch)
    problems = await checker.check_once()
    assert set(problems["no_credentials"]) == {"gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra"}
    assert CATALOG_MODEL_MISSING.labels(model="gpt-6-astra", reason="no_credentials")._value.get() == 1

    state["down"] = True
    assert await checker.check_once() is None                          # logged, not raised; gauge unchanged
    assert CATALOG_MODEL_MISSING.labels(model="gpt-6-astra", reason="no_credentials")._value.get() == 1

    state["down"] = False
    checker.has_credentials = lambda p: True                           # the key was added: next cycle clears it
    await checker.check_once()
    assert CATALOG_MODEL_MISSING.labels(model="gpt-6-astra", reason="no_credentials")._value.get() == 0


async def test_the_loop_never_blocks_start_and_stop_is_clean():
    async def fetch(url):
        raise ConnectionError("billing down at startup")

    checker = CatalogChecker("http://billing", REGISTRY, ALL, lambda s: True, fetch=fetch)
    checker.start()
    await checker.stop()
