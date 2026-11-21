"""S3: the context manager takes each model's window and output limit from models.yaml, not 'unknown_model_limits, assumed 32768'."""

from app.config.settings import LLMServiceConfig
from app.context_window.manager import ContextWindowManager
from app.providers.registry import ModelRegistry


def test_every_registry_model_has_its_yaml_window_under_both_names():
    cfg = LLMServiceConfig(_env_file=None)
    registry = ModelRegistry.load(cfg.model_registry_file)
    mgr = ContextWindowManager()
    assert mgr.apply_registry(registry) == len(registry.models) > 0
    for spec in registry.models.values():
        for name in {spec.id, spec.provider_model}:
            lim = mgr.provider_limits[name]
            assert (lim.context_window, lim.max_output_tokens) == (spec.context_window, spec.max_output_tokens)
            assert name not in mgr._warned_unknown
    assert "qwen/qwen3.8-27b" in mgr.provider_limits       # the analysis helper model is known too


def test_an_operator_declared_window_is_not_overridden():
    registry = ModelRegistry.load(LLMServiceConfig(_env_file=None).model_registry_file)
    spec = next(iter(registry.models.values()))
    mgr = ContextWindowManager()
    mgr._operator_models.add(spec.id)
    before = mgr.provider_limits.get(spec.id)
    mgr.apply_registry(registry)
    assert mgr.provider_limits.get(spec.id) is before
