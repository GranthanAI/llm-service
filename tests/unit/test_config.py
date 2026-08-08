"""
Unit tests for settings and configuration management.
"""

import pytest
from pydantic import ValidationError

from app.config.settings import LLMServiceConfig


def test_config_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
    monkeypatch.setenv("NVIDIA_API_KEY", "test_nvidia_key")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")

    config = LLMServiceConfig()
    assert config.service_name == "llm-service"
    assert config.service_version == "2.0.0"
    assert config.port == 8000
    assert config.groq_api_key.get_secret_value() == "test_groq_key"
    assert config.nvidia_api_key.get_secret_value() == "test_nvidia_key"
    assert config.gemini_api_key.get_secret_value() == "test_gemini_key"
    assert config.enable_smart_mode is True


def test_config_nvidia_key_alias(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_KEY", "nvapi-custom-alias")

    config = LLMServiceConfig()
    assert config.nvidia_api_key.get_secret_value() == "nvapi-custom-alias"


def test_config_invalid_log_level(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(ValidationError):
        LLMServiceConfig(log_level="INVALID_LEVEL")  # type: ignore[arg-type]


def test_config_loop_iterations_validation():
    with pytest.raises(ValidationError):
        LLMServiceConfig(langgraph_max_loop_iterations_smart=15)

    with pytest.raises(ValidationError):
        LLMServiceConfig(langgraph_max_loop_iterations_deep_research=0)
