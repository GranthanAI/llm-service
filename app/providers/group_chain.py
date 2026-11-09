"""
Builds the per-request ProviderChain for a billing grant (see registry.ModelRegistry.chain_for).

The chain contains only models of the granted group, each with its own adapter, breaker and deadlines. Adapters are
created once per catalog model and reused (an adapter holds an HTTP client). A model whose provider has no adapter in
this deployment is skipped, like one without credentials, and logged once.
"""

from collections.abc import Callable
from typing import Any

from app.config.logging import get_logger
from app.context_window.fit import build_group_fitter
from app.exceptions.provider import AllProvidersFailedError
from app.models.billing import BillingGrant
from app.providers.base import BaseProviderAdapter
from app.providers.chain import ChainEntry, ProviderChain
from app.providers.registry import ModelRegistry, ModelSpec

logger = get_logger("group_chain")

# entry-name prefix per provider family: keeps names compatible with the context-budget code ("gemini", "groq", ...)
ENTRY_PREFIX = {"groq": "groq", "google": "gemini", "anthropic": "anthropic", "openai": "openai"}


class GroupChainBuilder:
    """
    adapter_factory(spec) -> BaseProviderAdapter | None   (None = no adapter / no credentials for that model)
    breaker_factory(entry_name) -> CircuitBreaker | None
    """

    def __init__(
        self,
        registry: ModelRegistry,
        config: Any,
        context_window_manager: Any,
        adapter_factory: Callable[[ModelSpec], BaseProviderAdapter | None],
        has_credentials: Callable[[str], bool],
        breaker_factory: Callable[[str], Any] | None = None,
    ):
        self.registry = registry
        self.config = config
        self.manager = context_window_manager
        self._adapter_factory = adapter_factory
        self._has_credentials = has_credentials
        self._breaker_factory = breaker_factory
        self._adapters: dict[str, BaseProviderAdapter | None] = {}
        self._missing_logged: set[str] = set()

    def _adapter(self, spec: ModelSpec) -> BaseProviderAdapter | None:
        if spec.id not in self._adapters:
            self._adapters[spec.id] = self._adapter_factory(spec)
        adapter = self._adapters[spec.id]
        if adapter is None and spec.id not in self._missing_logged:
            self._missing_logged.add(spec.id)
            logger.warning("model_skipped_for_routing", model=spec.id, reason=f"no adapter for provider {spec.provider}")
        return adapter

    def build(self, grant: BillingGrant | None) -> ProviderChain | None:
        """None -> legacy routing. Otherwise the group chain; raises AllProvidersFailedError when the granted group has
        no usable model at all (nothing is ever tried outside the group)."""
        specs = self.registry.chain_for(grant, self._has_credentials)
        if specs is None:
            return None
        assert grant is not None
        entries: list[ChainEntry] = []
        for spec in specs:
            adapter = self._adapter(spec)
            if adapter is None:
                continue
            name = f"{ENTRY_PREFIX[spec.provider]}:{spec.provider_model}"
            entries.append(
                ChainEntry(
                    name=name,
                    adapter=adapter,
                    breaker=self._breaker_factory(name) if self._breaker_factory else None,
                    first_token_timeout_s=self._first_token_timeout(spec),
                    spec=spec,
                    input_cap=grant.max_context_tokens,
                )
            )
        if not entries:
            raise AllProvidersFailedError(
                f"No model of the granted group {grant.granted_group!r} is available (disabled, no credentials or no adapter)"
            )
        return ProviderChain(
            entries,
            idle_timeout_s=self.config.generation_idle_timeout_ms / 1000.0 or None,
            fitter=build_group_fitter(self.config, self.manager, logger),
        )

    def _first_token_timeout(self, spec: ModelSpec) -> float | None:
        c = self.config
        ms = {"groq": c.groq_first_token_timeout_ms, "google": c.gemini_timeout_ms}.get(
            spec.provider, c.generation_first_token_timeout_ms)
        return ms / 1000.0 or None


def build_group_chains(config: Any, context_window_manager: Any, breaker_registry: Any) -> GroupChainBuilder:
    """Loads config/models.yaml (MODEL_REGISTRY_FILE) and wires one adapter per catalog model. Raises RegistryError on
    an invalid or missing registry: a service that cannot honour a billing grant must not start."""
    from app.providers.gemini import GeminiAdapter
    from app.providers.groq import GroqAdapter

    registry = ModelRegistry.load(config.model_registry_file)
    keys = {
        "groq": config.groq_api_key.get_secret_value(),
        "google": config.gemini_api_key.get_secret_value(),
        "anthropic": config.anthropic_api_key.get_secret_value(),
        "openai": config.openai_api_key.get_secret_value(),
    }

    def has_credentials(provider: str) -> bool:
        return bool(keys.get(provider))

    def make_adapter(spec: ModelSpec) -> BaseProviderAdapter | None:
        if spec.provider == "groq":
            return GroqAdapter(api_key=keys["groq"], model=spec.provider_model,
                               timeout_s=config.groq_timeout_ms / 1000.0, tpm_limit=config.groq_tpm_limit)
        if spec.provider == "google":
            return GeminiAdapter(api_key=keys["google"], model=spec.provider_model)
        return None  # anthropic / openai adapters arrive in P5b

    builder = GroupChainBuilder(
        registry, config, context_window_manager, make_adapter, has_credentials,
        breaker_factory=lambda name: breaker_registry.get_or_create(name),
    )
    logger.info("model_registry_loaded", models=len(registry.models),
                groups={g: [m.id for m in registry.group_members(g)] for g in ("fast", "smart", "best", "ultra")})
    return builder
