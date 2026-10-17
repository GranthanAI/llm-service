"""
Configuration management for LLM Service using Pydantic Settings.
Implements LLD v2.0 Section 3.3 and Section 25.
"""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMServiceConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # General Service Settings
    environment: Literal["development", "staging", "production", "test"] = "development"
    service_name: str = "llm-service"
    service_version: str = "2.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "llm-service-group"
    kafka_input_topic: str = "chat.message.created"
    kafka_output_topic: str = "chat.response.generated"
    kafka_chunk_topic: str = "chat.response.chunk"
    kafka_dlq_topic: str = "chat.message.dlq"
    kafka_max_poll_interval_ms: int = 300000
    kafka_username: str | None = None
    kafka_password: SecretStr | None = None

    # gRPC Services (Context Collector baseline providers)
    memory_service_host: str = "localhost"
    memory_service_port: int = 50051
    graph_service_host: str = "localhost"
    graph_service_port: int = 50052
    retrieval_service_host: str = "localhost"
    retrieval_service_port: int = 50060
    grpc_deadline_ms: int = Field(default=2000, ge=500, le=30000)
    grpc_max_connections: int = 20

    # Provider API Keys & Hyperparameters
    groq_api_key: SecretStr = Field(default=SecretStr(""))
    groq_model: str = "openai/gpt-oss-20b"
    groq_timeout_ms: int = 5000
    groq_temperature: float = 0.0
    groq_max_tokens: int = 2048
    # Groq enforces tokens-per-minute per request as (prompt tokens + max_tokens). The free/on-demand tier of
    # gpt-oss-20b allows 8000, so a request must fit in it or it is rejected (HTTP 429) before generating a
    # single token. 0 disables the cap (paid tiers).
    groq_tpm_limit: int = 8000

    nvidia_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices(
            "NVIDIA_API_KEY", "NVIDIA_KEY", "nvidia_api_key", "nvidia_key"
        ),
    )
    nvidia_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_timeout_ms: int = 30000
    nvidia_temperature: float = 0.2
    nvidia_top_p: float = 0.7
    nvidia_max_tokens: int = 8192

    gemini_api_key: SecretStr = Field(default=SecretStr(""))
    gemini_model: str = "gemini-2.5-flash"
    gemini_timeout_ms: int = 15000
    gemini_temperature: float = 0.7
    gemini_top_p: float = 0.95
    gemini_max_tokens: int = 8192

    tavily_api_key: SecretStr = Field(default=SecretStr(""))

    # Tool Timeouts
    web_search_timeout_ms: int = 5000

    # Workflow Engine & LangGraph Loop Limits
    langgraph_max_loop_iterations_smart: int = Field(default=6, ge=1, le=10)
    langgraph_max_loop_iterations_deep_research: int = Field(default=4, ge=1, le=10)

    # Observability
    otel_endpoint: str = "http://localhost:4317"
    prometheus_port: int = 9090

    # Generation provider toggle: when true, NVIDIA NIM is the primary
    # generation provider (Gemini fallback) as per the HLD default. When
    # false, Groq is used as the generation provider instead (e.g. when
    # NVIDIA's configured model is unavailable / Gemini's key is revoked).
    enable_nvidia: bool = Field(
        default=True,
        validation_alias=AliasChoices("ENABLE_NVIDIA", "enable_nvidia"),
    )

    # Which provider answers when the primary generation provider (NVIDIA) fails:
    #   gemini (default, HLD) | groq | none  (none: fail the turn immediately).
    # Only that provider is ever contacted as the fallback.
    generation_fallback_provider: Literal["gemini", "groq", "none"] = Field(
        default="gemini",
        validation_alias=AliasChoices("GENERATION_FALLBACK_PROVIDER", "generation_fallback_provider"),
    )

    # Generation latency guards. NVIDIA's httpx timeout is a per-read timeout, so a model
    # that trickles reasoning tokens never trips it (observed: 60-180 s for "hi"). These are
    # deadlines on the *visible* token stream: if the primary produces no content token within
    # `generation_first_token_timeout_ms`, the turn is handed to the fallback provider (only
    # while nothing has been emitted, so text is never duplicated). `idle` bounds the gap
    # between tokens once streaming has started.
    generation_first_token_timeout_ms: int = Field(
        default=8000,
        validation_alias=AliasChoices("GENERATION_FIRST_TOKEN_TIMEOUT_MS", "generation_first_token_timeout_ms"),
    )
    generation_idle_timeout_ms: int = Field(
        default=20000,
        validation_alias=AliasChoices("GENERATION_IDLE_TIMEOUT_MS", "generation_idle_timeout_ms"),
    )
    generation_fallback_first_token_timeout_ms: int = Field(
        default=30000,
        validation_alias=AliasChoices(
            "GENERATION_FALLBACK_FIRST_TOKEN_TIMEOUT_MS", "generation_fallback_first_token_timeout_ms"
        ),
    )

    # Deterministic fast path for obvious greetings / small talk (skips Memory+Graph
    # collection and the Groq analysis call; generation is unchanged). See
    # app/request_analyzer/fast_path.py for the conservative matching rules.
    enable_fast_path: bool = Field(
        default=True,
        validation_alias=AliasChoices("ENABLE_FAST_PATH", "enable_fast_path"),
    )

    # Feature Flags (LLD Section 25.3)
    feature_smart_mode: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "FEATURE_SMART_MODE", "feature_smart_mode", "enable_smart_mode"
        ),
    )
    feature_deep_research_mode: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "FEATURE_DEEP_RESEARCH_MODE",
            "feature_deep_research_mode",
            "enable_deep_research_mode",
        ),
    )
    feature_web_search: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "FEATURE_WEB_SEARCH", "feature_web_search", "enable_web_search"
        ),
    )
    feature_vision: bool = Field(
        default=False,
        validation_alias=AliasChoices("FEATURE_VISION", "feature_vision"),
    )
    feature_grpc_keepalive: bool = Field(
        default=True,
        validation_alias=AliasChoices("FEATURE_GRPC_KEEPALIVE", "feature_grpc_keepalive"),
    )

    @field_validator("kafka_bootstrap_servers")
    @classmethod
    def validate_kafka_servers(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("kafka_bootstrap_servers cannot be empty")
        return v.strip()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"Invalid log_level: {v}. Must be one of {valid_levels}")
        return upper

    @property
    def enable_smart_mode(self) -> bool:
        return self.feature_smart_mode

    @property
    def enable_deep_research_mode(self) -> bool:
        return self.feature_deep_research_mode

    @property
    def enable_web_search(self) -> bool:
        return self.feature_web_search


@lru_cache(maxsize=1)
def get_settings() -> LLMServiceConfig:
    """Returns cached instance of LLMServiceConfig."""
    return LLMServiceConfig()
