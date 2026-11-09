"""
Model registry (config/models.yaml) and billing-group routing.

The registry is keyed by billing's catalog ids. When billing grants a request a model group, the request may only
be answered by models of that group: the chain is built here and NEVER contains a model outside the granted group
(no escalation to a higher group, no fallback to a cheaper or unpriced one). Without a grant the legacy
GENERATION_PROVIDER_ORDER chain is used, unchanged.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.config.logging import get_logger
from app.models.billing import BillingGrant

logger = get_logger("model_registry")

PROVIDERS = frozenset({"groq", "google", "anthropic", "openai"})
GROUPS = ("fast", "smart", "best", "ultra")

# Models with no price in billing's catalog: legacy chain only, never group routing.
UNPRICED_MODELS = frozenset({"qwen/qwen3.8-27b", "nvidia/nemotron-3.5-lightning-30b-a3b"})

DEFAULT_GROUP_ORDER: dict[str, list[str]] = {
    "fast": ["openai/gpt-oss-20b", "gemini-2.5-flash-lite"],
    "smart": ["openai/gpt-oss-120b", "claude-haiku-5-5", "gpt-6-luna", "gemini-3.1-flash-lite", "gemini-2.5-flash"],
    "best": ["claude-sonnet-5-5", "gpt-6.1-sol", "gemini-3.1-pro"],
    "ultra": ["claude-opus-5-5", "gpt-6-astra"],
}


class RegistryError(ValueError):
    """The model registry file is invalid (a startup error)."""


@dataclass(frozen=True)
class ModelSpec:
    id: str                 # billing catalog id
    provider: str           # groq | google | anthropic | openai
    provider_model: str     # the string the SDK needs
    group: str
    context_window: int
    max_output_tokens: int
    enabled: bool = True


class ModelRegistry:
    def __init__(self, models: dict[str, ModelSpec], group_order: dict[str, list[str]] | None = None):
        self.models = models
        if group_order is None:  # the defaults, restricted to the models this registry actually has
            group_order = {g: [i for i in ids if i in models and models[i].group == g]
                           for g, ids in DEFAULT_GROUP_ORDER.items()}
        self.group_order = {g: list(ids) for g, ids in group_order.items()}
        self._warned: set[str] = set()
        self._validate()

    # --- loading ---------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "ModelRegistry":
        try:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except OSError as exc:
            raise RegistryError(f"cannot read model registry {path}: {exc}") from exc
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelRegistry":
        entries = raw.get("models")
        if not isinstance(entries, dict) or not entries:
            raise RegistryError("model registry needs a non-empty `models` mapping")
        models: dict[str, ModelSpec] = {}
        for model_id, body in entries.items():
            if model_id in models:
                raise RegistryError(f"duplicate model id {model_id!r}")
            if not isinstance(body, dict):
                raise RegistryError(f"model {model_id!r}: expected a mapping")
            try:
                spec = ModelSpec(
                    id=str(model_id), provider=str(body["provider"]), provider_model=str(body["provider_model"]),
                    group=str(body["group"]), context_window=int(body["context_window"]),
                    max_output_tokens=int(body["max_output_tokens"]), enabled=bool(body.get("enabled", True)),
                )
            except KeyError as exc:
                raise RegistryError(f"model {model_id!r}: missing field {exc.args[0]!r}") from exc
            models[spec.id] = spec
        order = raw.get("group_order")
        return cls(models, order if isinstance(order, dict) else None)

    def _validate(self) -> None:
        seen_provider_models: set[tuple[str, str]] = set()
        for spec in self.models.values():
            if spec.provider not in PROVIDERS:
                raise RegistryError(f"model {spec.id!r}: unknown provider {spec.provider!r} (one of {sorted(PROVIDERS)})")
            if spec.group not in GROUPS:
                raise RegistryError(f"model {spec.id!r}: unknown group {spec.group!r} (one of {list(GROUPS)})")
            if spec.context_window < 1000 or spec.max_output_tokens < 1:
                raise RegistryError(f"model {spec.id!r}: context_window / max_output_tokens out of range")
            key = (spec.provider, spec.provider_model)
            if key in seen_provider_models:
                raise RegistryError(f"duplicate provider model {spec.provider}:{spec.provider_model}")
            seen_provider_models.add(key)
            if spec.id in UNPRICED_MODELS or spec.provider_model in UNPRICED_MODELS:
                logger.warning("unpriced_model_in_registry_group", model=spec.id, group=spec.group,
                               note="has no price in billing's catalog; it is legacy-chain only and never group-routed")
        for group, ids in self.group_order.items():
            if group not in GROUPS:
                raise RegistryError(f"group_order names an unknown group {group!r}")
            for model_id in ids:
                spec = self.models.get(model_id)
                if spec is None:
                    raise RegistryError(f"group_order[{group}] names {model_id!r}, which is not in `models`")
                if spec.group != group:
                    raise RegistryError(f"group_order[{group}] lists {model_id!r}, whose group is {spec.group!r}")

    # --- routing ---------------------------------------------------------------------------------------------
    def group_members(self, group: str) -> list[ModelSpec]:
        """The group's models in routing order; any member missing from group_order follows it."""
        ordered = [self.models[i] for i in self.group_order.get(group, []) if i in self.models]
        rest = [m for m in self.models.values() if m.group == group and m not in ordered]
        return ordered + rest

    def _usable(self, spec: ModelSpec, has_credentials: Callable[[str], bool]) -> bool:
        if spec.id in UNPRICED_MODELS or spec.provider_model in UNPRICED_MODELS:
            return False
        if not spec.enabled:
            self._warn_once(spec.id, "disabled")
            return False
        if not has_credentials(spec.provider):
            self._warn_once(spec.id, f"no credentials for provider {spec.provider}")
            return False
        return True

    def _warn_once(self, model_id: str, why: str) -> None:
        if model_id not in self._warned:
            self._warned.add(model_id)
            logger.warning("model_skipped_for_routing", model=model_id, reason=why)

    def chain_for(self, grant: BillingGrant | None, has_credentials: Callable[[str], bool]) -> list[ModelSpec] | None:
        """The models a granted request may try, in order, or None when this request uses the legacy chain (no grant,
        or granted_group null). An empty list means the group has no usable model right now.

        granted_model set -> it first, then the rest of its group in group_order; Auto -> the group's group_order.
        Disabled models and providers without credentials are skipped. Never anything outside the granted group."""
        if grant is None or not grant.granted_group:
            return None
        group = grant.granted_group.strip().lower()
        members = self.group_members(group)
        if grant.granted_model:
            first = self.models.get(grant.granted_model)
            if first is None or first.group != group:
                logger.warning("granted_model_not_in_group", granted_model=grant.granted_model, group=group)
            else:
                members = [first] + [m for m in members if m.id != first.id]
        return [m for m in members if self._usable(m, has_credentials)]

    def cap_outputs(self, caps: dict[str, int]) -> None:
        """Lower each model's output limit to the service's own per-provider limit (GROQ_MAX_TOKENS, GEMINI_MAX_TOKENS, ...).
        Without it a Groq free-tier request asks for 8192 output tokens against an 8000-token-per-request cap and the context
        budget comes out negative."""
        from dataclasses import replace

        for model_id, spec in list(self.models.items()):
            cap = caps.get(spec.provider)
            if cap and cap < spec.max_output_tokens:
                self.models[model_id] = replace(spec, max_output_tokens=cap)
        self.__dict__.pop("_by_provider_model", None)

    def catalog_id(self, provider: str, provider_model: str) -> str | None:
        """The billing catalog id for a provider's model string (None for a legacy-only model with no catalog id)."""
        if not hasattr(self, "_by_provider_model"):
            self._by_provider_model = {(m.provider, m.provider_model): m.id for m in self.models.values()}
        return self._by_provider_model.get((provider, provider_model))

    def context_window_of(self, model_id: str) -> int | None:
        spec = self.models.get(model_id)
        return spec.context_window if spec else None
