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
    retrieval_service_port: int = 50053
    grpc_deadline_ms: int = Field(default=2000, ge=500, le=30000)
    grpc_max_connections: int = 20

    # Provider API Keys & Hyperparameters
    groq_api_key: SecretStr = Field(default=SecretStr(""))
    groq_model: str = "llama-3.3-70b-versatile"
    groq_timeout_ms: int = 5000
    groq_temperature: float = 0.0
    groq_max_tokens: int = 2048

    nvidia_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices(
            "NVIDIA_API_KEY", "NVIDIA_KEY", "nvidia_api_key", "nvidia_key"
        ),
    )
    nvidia_model: str = "meta/llama-3.3-70b-instruct"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_timeout_ms: int = 30000
    nvidia_temperature: float = 0.2
    nvidia_top_p: float = 0.7
    nvidia_max_tokens: int = 4096

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

    # Feature Flags
    enable_smart_mode: bool = True
    enable_deep_research_mode: bool = True
    enable_web_search: bool = True
    feature_vision: bool = False
    feature_grpc_keepalive: bool = True

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"Invalid log_level: {v}. Must be one of {valid_levels}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> LLMServiceConfig:
    """Returns cached instance of LLMServiceConfig."""
    return LLMServiceConfig()
