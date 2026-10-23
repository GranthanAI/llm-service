"""
Configuration management for LLM Service using Pydantic Settings.
Implements LLD v2.0 Section 3.3 and Section 25.
"""

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
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
    # Maximum INPUT tokens sent per request; lower-priority context (graph, memory, old history, search
    # results) is trimmed to fit. Sized for the free Groq tier (8000 tokens/min per request, 7000 input
    # tokens/min on qwen): the prompt AND the answer must both fit. 0 = no cap (model's full window).
    # LEGACY / rate-limit guard, OFF by default (0): when > 0 it is an extra hard cap on the
    # INPUT tokens of one request (useful on rate-limited tiers). It is NOT the context budget (see CONTEXT_BUDGET_*
    # below) and is only ever applied as a minimum. Set to 0 on accounts whose rate limit is not the constraint.
    prompt_token_budget: int = 0

    # ---- Visualizations (Mermaid diagrams, Manim animations) ----
    # Conservative defaults: Mermaid on (rendered in the browser, no server work), Manim OFF until a worker is deployed.
    visualization_enabled: bool = True
    mermaid_enabled: bool = True
    manim_enabled: bool = False
    # Modes whose prompt may offer visualizations (comma separated mode ids). Grounded modes (ask_files, web_search,
    # deep_research) are off by default: a diagram there must reflect evidence, so enable them deliberately.
    visualization_modes: str = "default,tutor,code,smart"
    # Total attempts per visualization block: attempt 1 = the original, attempts 2..N = repairs (default 3 total).
    visualization_max_attempts: int = Field(default=3, ge=1, le=5)
    visualization_max_blocks_per_response: int = Field(default=5, ge=1, le=20)
    manim_max_jobs_per_response: int = Field(default=1, ge=1, le=5)
    mermaid_max_source_chars: int = Field(default=6000, ge=200, le=50_000)
    manim_max_spec_chars: int = Field(default=4000, ge=200, le=50_000)
    visualization_repair_enabled: bool = True
    # Manim worker / rendering
    manim_render_timeout_seconds: int = Field(default=90, ge=5, le=600)
    manim_max_output_mb: int = Field(default=20, ge=1, le=200)
    manim_max_concurrent_jobs: int = Field(default=1, ge=1, le=16)       # per worker process
    manim_max_active_jobs_per_user: int = Field(default=2, ge=1, le=20)
    visualization_jobs_per_user_per_hour: int = Field(default=20, ge=1, le=1000)
    manim_job_ttl_seconds: int = Field(default=3600, ge=60)                # status records
    manim_media_retention_hours: int = Field(default=24, ge=1)             # rendered files
    # Scratch copies of finished renders, served once to conversation-service which stores the verified MP4 in object
    # storage. Deliberately OUTSIDE the source tree (system temp) so a render can never be written into the repository.
    manim_media_dir: str = str(Path(tempfile.gettempdir()) / "granthan-viz")
    manim_python: str = ""                  # interpreter that has manim installed; empty = this interpreter
    manim_quality: Literal["l", "m", "h"] = "l"
    # True: renders run as subprocesses of this API process (single-node/dev; jobs are in memory). False: jobs go through
    # Redis to separate `python -m app.visualization.worker` processes (the isolated deployment).
    manim_inline_worker: bool = True
    visualization_redis_url: str = "redis://localhost:6379/0"
    # Shared secret the worker presents when it calls the internal repair endpoint (empty = endpoint trusts the network).
    visualization_internal_token: SecretStr = Field(default=SecretStr(""))

    # ---- Context budget (application policy; independent of provider rate limits) ----
    # Max tokens one request's effective context may use (system + history + memory + retrieval + tools + output
    # reserve + safety margin). The model's real window is a second, independent ceiling: both must hold.
    context_budget_enabled: bool = True          # False: only the model's window (and the legacy cap) limit a request
    context_budget_default_tokens: int = Field(default=150_000, ge=1_000)
    context_budget_groq_tokens: int | None = Field(default=150_000, ge=1_000)    # None -> default
    context_budget_gemini_tokens: int | None = Field(default=150_000, ge=1_000)  # None -> default
    # Optional per-model overrides, JSON: {"groq:openai/gpt-oss-120b": 100000, "gemini-1.5-flash": 200000}
    # (key = "<provider>:<model>" or just "<model>").
    context_budget_model_overrides: dict[str, int] = Field(default_factory=dict)
    # Real context windows of models that are NOT in the built-in table, JSON: {"qwen/qwen3.8-27b": 131072}. Verify
    # against the provider's documentation: an unregistered model uses the conservative window below (with a warning).
    context_model_windows: dict[str, int] = Field(default_factory=dict)
    context_unknown_model_window_tokens: int = Field(default=32_768, ge=2_048)

    # ---- Automatic compaction ----
    context_compaction_enabled: bool = True
    # Compact when the estimated input reaches this fraction of the effective input budget (or exceeds it).
    context_compaction_trigger_ratio: float = 0.85
    # ...and aim for this fraction afterwards.
    context_compaction_target_ratio: float = 0.65
    # Most recent messages that are kept verbatim whenever feasible.
    context_compaction_min_recent_messages: int = Field(default=6, ge=1, le=100)

    # ---- Token safety ----
    # Fraction of the budget held back because token counts are estimates (no provider tokenizer locally).
    context_token_safety_margin: float = 0.05
    # Output tokens reserved when the request does not state its own max_tokens.
    context_output_token_reserve: int = Field(default=8192, ge=256)
    # Order in which answer generation tries providers; the first to produce a token wins and the rest
    # are fallbacks. Entries: `groq:<model>` (same Groq key; each model has its own rate limits), `nvidia`,
    # `gemini`. Providers without credentials are skipped. NVIDIA is last because it can take 1-2 minutes.
    # The same Groq models also back up request planning and search-query rewriting. Empty = legacy
    # NVIDIA-primary behaviour (ENABLE_NVIDIA / GENERATION_FALLBACK_PROVIDER).
    generation_provider_order: str = "groq:openai/gpt-oss-120b,groq:qwen/qwen3.8-27b,gemini,groq:openai/gpt-oss-20b,nvidia"
    groq_first_token_timeout_ms: int = 12000
    # Upper bound for the last-resort NVIDIA call during request planning (it can take a minute).
    analysis_fallback_timeout_ms: int = 8000

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

    @property
    def generation_order_list(self) -> list[str]:
        seen: list[str] = []
        for item in self.generation_provider_order.split(","):
            item = item.strip()
            if item and item not in seen:
                seen.append(item)
        return seen

    @property
    def groq_order_models(self) -> list[str]:
        """Groq model ids named in the order, in order."""
        return [i.split(":", 1)[1] for i in self.generation_order_list if i.startswith("groq:")]

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
    def visualization_mode_list(self) -> list[str]:
        return [m.strip().lower() for m in self.visualization_modes.split(",") if m.strip()]

    @field_validator("context_budget_model_overrides", "context_model_windows")
    @classmethod
    def _positive_token_maps(cls, v: dict[str, int]) -> dict[str, int]:
        for key, tokens in v.items():
            if not key.strip() or tokens < 1_000:
                raise ValueError(f"token budgets/windows must be >= 1000 and keyed by model (got {key!r}: {tokens})")
        return v

    @field_validator("context_token_safety_margin")
    @classmethod
    def _safety_margin(cls, v: float) -> float:
        if not 0.0 <= v < 0.5:
            raise ValueError("context_token_safety_margin must be in [0, 0.5)")
        return v

    @model_validator(mode="after")
    def _context_policy_is_consistent(self):
        """Fail at startup, never silently fall back to unsafe values."""
        if not 0.0 < self.context_compaction_trigger_ratio <= 1.0:
            raise ValueError("context_compaction_trigger_ratio must satisfy 0 < ratio <= 1")
        if not 0.0 < self.context_compaction_target_ratio < self.context_compaction_trigger_ratio:
            raise ValueError("context_compaction_target_ratio must satisfy 0 < target < trigger")
        if self.prompt_token_budget < 0:
            raise ValueError("prompt_token_budget must be >= 0")
        smallest = min(
            [self.context_budget_default_tokens]
            + [t for t in (self.context_budget_groq_tokens, self.context_budget_gemini_tokens) if t]
        )
        if self.context_output_token_reserve >= smallest * (1 - self.context_token_safety_margin):
            raise ValueError("context_output_token_reserve leaves no room for input within the smallest context budget")
        return self

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
