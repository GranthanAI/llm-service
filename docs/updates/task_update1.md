# GraphGPT LLM Service v2.0 — Task Update Report (Part 1)

## Executive Summary & Engineering Overview

This engineering report provides an exhaustive technical record of the architectural design, implementation details, testing verifications, and design pattern compliances achieved across **Phase 1 (Project Bootstrap)**, **Phase 2 (Core Models & Schemas)**, and **Phase 3 (Kafka Messaging Infrastructure)** for the GraphGPT `llm-service` codebase.

All implementations strictly adhere to the authoritative specifications defined in:
- `docs/hld2.md` — High Level Design Document v2.0
- `docs/lld2.md` — Low Level Design Document v2.0
- Principles of **SOLID Design**, **Dependency Injection (DI)**, and **Controlled Singletons**.

---

## 1. Phase 1: Project Bootstrap & Base Infrastructure

### 1.1 Scope and Objectives
The objective of Phase 1 was to establish an enterprise-grade, production-ready Python service framework configured for asynchronous high-concurrency execution, comprehensive telemetry, structural validation, and architectural boundary governance.

### 1.2 Delivered Artifacts and Architecture
The following structural foundation was created:

| Component | Target File | Description |
|---|---|---|
| **Dependency Manifest** | `pyproject.toml` | Configured Python `>=3.12`, FastAPI, Pydantic v2, Pydantic Settings, Structlog, OpenTelemetry SDK, Prometheus Client, `aiokafka`, `grpcio`, `httpx`, `tenacity`, `groq`, `openai`, `google-genai`, `langgraph`, `ruff`, and `pytest`. |
| **Settings Management** | `app/config/settings.py` | Centralized `LLMServiceConfig` implementing `pydantic-settings.BaseSettings` with environment variable overrides, secret string masking, and validation. |
| **Structured Logging** | `app/config/logging.py` | JSON structured logging via `structlog` with context variables for `trace_id`, `span_id`, `request_id`, `conversation_id`, and ISO-8601 UTC timestamps. |
| **Observability & Tracing** | `app/utils/tracing.py` | Distributed OpenTelemetry tracer provider with OTLP gRPC exporter and `@trace_span` decorator. |
| **Prometheus Telemetry** | `app/utils/metrics.py` | Standardized Prometheus metrics (`REQUESTS_TOTAL`, `REQUEST_DURATION`, `WORKFLOW_DISPATCH_TOTAL`, `TTFT`, `CIRCUIT_BREAKER_STATE`, `KAFKA_LAG`, `CONTEXT_FETCH_DURATION`, `CONTEXT_DEGRADED_TOTAL`). |
| **DI Container** | `app/api/dependencies.py` | Application-level `Container` holding initialized service dependencies without global mutable state in business logic. |
| **Internal Endpoints** | `app/api/internal/` | Kubernetes `/health`, `/healthz`, `/ready`, `/readyz`, and Prometheus `/metrics` scrapers. |
| **Application Lifespan** | `app/main.py` | FastAPI application composition root orchestrating orderly asynchronous initialization and graceful shutdown. |
| **Automated Tooling** | `Makefile` | Make targets for dependency sync, linting, formatting, tests, proto compilation, and docker lifecycle. |
| **Boundary Checker** | `scripts/check_import_boundaries.py` | AST-based architectural import rule verifier enforcing LLD2 Section 2.2 restrictions. |

### 1.3 Architectural Import Boundary Governance
The boundary checker script parses Python AST across modules to enforce strict architectural rules:
1. `workflow_engine/mode_handlers/` must **NEVER** import `langgraph` or `workflow_engine/langgraph_workflows/`.
2. `tools/` must **NEVER** import `context/`, `grpc/clients/`, or `prompts/`.
3. `context/` must **NEVER** import `tools/`.
4. `prompts/` must **NEVER** import `tools/`.
5. `grpc/clients/` must **NEVER** import `workflow_engine/` or `request_analyzer/`.

### 1.4 Structured Logging Architecture
The structured logging subsystem initializes in `app/config/logging.py` and binds context variables across async task executions:

```python
# Context variables for distributed request correlation
correlation_request_id: ContextVar[str] = ContextVar("correlation_request_id", default="")
correlation_trace_id: ContextVar[str] = ContextVar("correlation_trace_id", default="")
correlation_conversation_id: ContextVar[str] = ContextVar("correlation_conversation_id", default="")
```

Each log line is emitted in standard JSON format containing:
- `timestamp`: ISO-8601 UTC timestamp (e.g. `2026-08-08T18:00:00.000000Z`)
- `level`: Log level (`info`, `warning`, `error`, `debug`)
- `service`: `llm-service`
- `version`: `2.0.0`
- `request_id`: ContextVar request ID
- `trace_id`: Distributed W3C trace ID
- `conversation_id`: Active chat conversation ID
- `event`: Event message and arbitrary key-value metadata.

### 1.5 Container & Dependency Injection Pattern
The composition root pattern is enforced via `Container` in `app/api/dependencies.py`:
- Pure constructor injection without global variables.
- Instantiated exactly once inside `lifespan` in `app/main.py` and mounted to `app.state.container`.
- Dependencies requested in FastAPI route handlers using typed `Depends(get_container)`.

```python
class Container:
    def __init__(self, config: LLMServiceConfig):
        self.config: LLMServiceConfig = config
        self.is_ready: bool = False
        self.is_healthy: bool = True
        self.kafka_producer: KafkaPublisher | None = None
        self.kafka_consumer: KafkaConsumerEngine | None = None
        self.memory_client: MemoryServiceClient | None = None
        self.graph_client: GraphServiceClient | None = None
        self.retrieval_client: RetrievalServiceClient | None = None
        self.context_collector: ContextCollector | None = None
        self.request_analyzer: RequestAnalyzer | None = None
```

---

## 2. Phase 2: Core Models, Schemas & Enums

### 2.1 State Model Partitioning Architecture
To maintain clear separation of concerns, the state models were partitioned into three non-overlapping tiers matching LLD2 Section 14:
1. **Pre-Dispatch Context (`PipelineContext`)**: Shared Pydantic model tracking request correlation, raw message, baseline context bundle, and execution plan.
2. **Deterministic Mode State (`ModeHandlerState` & `ModeHandlerOutput`)**: Lightweight Pydantic containers with zero LangGraph imports for sub-millisecond execution.
3. **Agentic Graph States (`SmartGraphState` & `DeepResearchGraphState`)**: LangGraph `TypedDict` structures with `operator.add` reducers for iterative accumulator fields.
4. **Normalized Output Bridge (`WorkflowResult`)**: Normalized interface bridging mode handler outputs and LangGraph states into `PromptBuilder`.

```
+-------------------------------------------------------------------------------+
|                       State Model Tiering (LLD2 Sec 14)                       |
+-------------------------------------------------------------------------------+
|                                                                               |
|  1. Pre-Dispatch Tier:                                                        |
|     +----------------------------------------------------------------------+  |
|     | PipelineContext (Pydantic BaseModel)                                 |  |
|     | - request_id, conversation_id, user_id, message_id                   |  |
|     | - context_bundle: ContextBundle                                      |  |
|     | - plan: ExecutionPlan                                                |  |
|     | - trace_id, span_id, started_at                                      |  |
|     +----------------------------------------------------------------------+  |
|                                     |                                         |
|                                     v                                         |
|  2. Execution Engine Dispatch:                                                |
|     +--------------------------------+  +----------------------------------+  |
|     | Mode Handler State             |  | LangGraph State                  |  |
|     | (Deterministic Modes)          |  | (Agentic Loop Modes)             |  |
|     |                                |  |                                  |  |
|     | ModeHandlerState               |  | SmartGraphState                  |  |
|     | - Plain Pydantic BaseModel     |  | - TypedDict                      |  |
|     | - ZERO LangGraph dependencies  |  | - Annotated[list, operator.add]  |  |
|     |                                |  |                                  |  |
|     | Output: ModeHandlerOutput      |  | DeepResearchGraphState           |  |
|     | - draft_response: None         |  | - TypedDict                      |  |
|     | - tool_outputs: list           |  | - Annotated[list, operator.add]  |  |
|     +--------------------------------+  +----------------------------------+  |
|                     \                                    /                    |
|                      \                                  /                     |
|                       v                                v                      |
|  3. Normalized Bridge Tier:                                                   |
|     +----------------------------------------------------------------------+  |
|     | WorkflowResult (Pydantic BaseModel)                                  |  |
|     | - draft_content: str | None                                          |  |
|     | - tool_outputs: list[ToolResult]                                     |  |
|     | - engine_type: "mode_handler" | "langgraph"                          |  |
|     | - conversation_history: list[Message]                                |  |
|     +----------------------------------------------------------------------+  |
|                                                                               |
+-------------------------------------------------------------------------------+
```

### 2.2 Enumerations and Execution Planning
Implemented in `app/models/execution_plan.py` utilizing Python 3.12 `enum.StrEnum`:

```python
class IntentCategory(StrEnum):
    GENERAL_CHAT = "GENERAL_CHAT"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    CODE_GENERATION = "CODE_GENERATION"
    CODE_DEBUGGING = "CODE_DEBUGGING"
    CODE_EXPLANATION = "CODE_EXPLANATION"
    RESEARCH = "RESEARCH"
    WEB_SEARCH = "WEB_SEARCH"
    DOCUMENT_ANALYSIS = "DOCUMENT_ANALYSIS"
    TUTORING = "TUTORING"
    CREATIVE_WRITING = "CREATIVE_WRITING"
    REASONING = "REASONING"

class UserMode(StrEnum):
    DEFAULT = "default"
    TUTOR = "tutor"
    CODE = "code"
    ASK_FILES = "ask_files"
    WEB_SEARCH = "web_search"
    SMART = "smart"
    DEEP_RESEARCH = "deep_research"

class Skill(StrEnum):
    GENERAL_CHAT = "general_chat"
    TUTOR = "tutor"
    CODING = "coding"
    RESEARCH = "research"
    WRITING = "writing"
    REASONING = "reasoning"

class ReasoningMode(StrEnum):
    DIRECT = "DIRECT"
    CHAIN_OF_THOUGHT = "CHAIN_OF_THOUGHT"
    REACT = "REACT"

class EngineType(StrEnum):
    MODE_HANDLER = "mode_handler"
    LANGGRAPH = "langgraph"
```

### 2.3 ExecutionPlan Schema
```python
class ToolCall(BaseModel):
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    parallel: bool = True
    required: bool = False

class ExecutionPlan(BaseModel):
    intent: IntentCategory = IntentCategory.GENERAL_CHAT
    mode: UserMode = UserMode.DEFAULT
    skill: Skill = Skill.GENERAL_CHAT
    reasoning: ReasoningMode = ReasoningMode.DIRECT
    tools: list[ToolCall] = Field(default_factory=list)
    max_iterations: int = Field(default=1, ge=1)
    suggested_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    analysis_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    groq_model_used: str = "llama-3.3-70b-versatile"
    analysis_latency_ms: float = 0.0
```

### 2.4 Kafka Ingestion Event Schema (`chat.message.created`)
Canonical JSON format matching HLD2 Section 26.2 and consumed by `KafkaConsumerEngine`:

```json
{
  "event_type": "chat.message.created",
  "schema_version": "2.0",
  "message_id": "msg_abc123456",
  "conversation_id": "conv_xyz78910",
  "user_id": "user_def456789",
  "content": "Explain quantum entanglement in simple terms",
  "mode_hint": "tutor",
  "file_ids": ["file_001", "file_002"],
  "metadata": {
    "client_timestamp": "2026-08-08T14:47:51Z",
    "client_version": "3.2.1"
  },
  "trace_context": {
    "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    "tracestate": ""
  },
  "timestamp": "2026-08-08T14:47:51.234Z"
}
```

### 2.5 Kafka Egress Event Schema (`chat.response.generated`)
Canonical final generated event produced by `KafkaPublisher` per HLD2 Section 26.3:

```json
{
  "event_type": "chat.response.generated",
  "schema_version": "2.0",
  "response_id": "resp_ghi789012",
  "conversation_id": "conv_xyz78910",
  "user_id": "user_def456789",
  "request_message_id": "msg_abc123456",
  "full_content": "Quantum entanglement is a phenomenon where...",
  "provider": "nvidia",
  "generation_fallback_used": false,
  "mode": "tutor",
  "skill": "tutor",
  "engine_type": "mode_handler",
  "usage": {
    "prompt_tokens": 1200,
    "completion_tokens": 320,
    "total_tokens": 1520
  },
  "cost_usd": 0.0076,
  "latency_ms": 1840.0,
  "ttft_ms": 520.0,
  "finish_reason": "stop",
  "tools_used": [],
  "context_sources": ["memory", "graph", "retrieval"],
  "trace_context": {
    "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
  },
  "status": "success",
  "timestamp": "2026-08-08T14:47:53.074Z"
}
```

### 2.6 Domain Exception Hierarchies
Implemented in `app/exceptions/`:
- **Provider Exceptions (`app/exceptions/provider.py`)**: `ProviderError`, `AllProvidersFailedError`, `ProviderTimeoutError`, `ProviderRateLimitError`, `CircuitOpenError`.
- **Tool Exceptions (`app/exceptions/tool.py`)**: `ToolError`, `RequiredToolFailedError`, `ToolTimeoutError`, `UnknownToolError`, `ToolValidationError`.
- **gRPC Exceptions (`app/exceptions/grpc.py`)**: `GRPCError`, `GRPCUnavailableError`, `GRPCTimeoutError`.
- **Analysis Exceptions (`app/exceptions/analysis.py`)**: `AnalysisError`, `PlanParseError`, `CriticalAnalysisError`, `UnknownModeError`, `ContextOverflowError`.

---

## 3. Phase 3: Kafka Messaging Infrastructure

### 3.1 Ingestion & Egress Architecture (LLD2 Sec 6 & 7)

```
+-----------------------------------------------------------------------------------+
|                              Kafka Ingestion & Egress                             |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ Kafka Topic: chat.message.created ]                                            |
|                  |                                                                |
|                  v                                                                |
|        KafkaConsumerEngine (AIOKafkaConsumer, group: llm-service-group)           |
|                  |                                                                |
|                  +---> Deserialization Check ---> (Invalid) ---> Publish to DLQ   |
|                  |                                               & Commit Offset  |
|                  v                                                                |
|        TTLCache Idempotency Check (10,000 max, 300s TTL)                          |
|                  |                                                                |
|                  +---> Duplicate? ---> (Yes) ---> Log & Commit Offset             |
|                  v                                                                |
|        asyncio.Semaphore(max_concurrent=50)                                       |
|                  |                                                                |
|                  v                                                                |
|        Trace Header Extraction (W3C traceparent / tracestate)                     |
|                  |                                                                |
|                  v                                                                |
|        ChatConsumer.handle(event)                                                 |
|                  |                                                                |
|                  v                                                                |
|        Pipeline Execution (Retried up to 3x with Exponential Backoff + Jitter)    |
|                  |                                                                |
|                  +---> Success ---> Publish Egress Events ---> Commit Offset      |
|                  |                                                                |
|                  +---> Failure ---> Exhausted Retries ---> Publish to DLQ         |
|                                                            & Commit Offset        |
+-----------------------------------------------------------------------------------+
```

### 3.2 Kafka Publisher (`app/producers/kafka_producer.py`)
Key design characteristics:
- **Client Configuration**: `AIOKafkaProducer` with `acks="all"`, `compression_type="lz4"`, `max_request_size=2097152` (2MB), `linger_ms=5`, and `enable_idempotence=True`.
- **Partitioning Key Strategy**: All outgoing messages are keyed with `conversation_id.encode("utf-8")`, guaranteeing in-order per-conversation delivery and partition affinity.
- **Trace Context Propagation**: Automatically injects W3C headers:
  - `traceparent`
  - `tracestate`
  - `correlation_id`
- **Publishing Methods**:
  - `publish_chunk(event: ChatResponseChunkEvent)` -> `chat.response.chunk`
  - `publish_response(ctx: PipelineContext, full_response: str)` -> `chat.response.generated`
  - `publish_memory_update(ctx: PipelineContext, full_response: str)` -> `memory.update.requested`
  - `publish_cancellation(ctx: PipelineContext, reason: str)` -> `chat.response.cancelled`
  - `publish_dlq(event: ChatMessageDLQEvent)` -> `chat.message.dlq`

### 3.3 Kafka Consumer Engine (`app/consumers/kafka_consumer.py`)
Key design characteristics:
- **Manual Offset Commit**: `enable_auto_commit=False`. Offsets are committed via `_commit_offset(tp, offset + 1)` *only* after processing succeeds or a DLQ record is produced.
- **Task Concurrency Limiting**: Bounded by `asyncio.Semaphore(50)` preventing memory exhaustion under traffic surges.
- **Poison Pill Protection**: Malformed JSON payloads bypass the pipeline, trigger a `ChatMessageDLQEvent` to `chat.message.dlq`, and commit the offset to prevent queue starvation.
- **Idempotency Guard**: Integrated `TTLCache` (max size 10,000, 300s TTL) tracking `message_id`. Re-delivered duplicate messages acknowledge the offset without invoking downstream compute.

### 3.4 In-Memory TTL Cache (`app/utils/cache.py`)
Implemented `TTLCache` with combined lazy TTL expiration, explicit background cleanup, and LRU eviction:
- Thread-safe and async-compatible.
- Tracks `last_accessed` timestamp for O(1) eviction of least-recently-used keys when capacity is reached.

```python
class TTLCache:
    def __init__(self, max_size: int = 10000, default_ttl_seconds: int = 300):
        self._store: dict[str, CacheEntry] = {}
        self._max_size: int = max_size
        self._default_ttl: int = default_ttl_seconds
        self._lock: asyncio.Lock = asyncio.Lock()
```

### 3.5 Consumer & Operation Retry Manager (`app/utils/retry.py`)
- **Exponential Backoff with Jitter**:
  $$\text{delay} = \min(\text{initial} \times \text{backoff}^{\text{attempt}-1}, \text{max\_delay}) \pm 10\% \text{ jitter}$$
- **Transient Failure Predicate (`is_retriable`)**: Identifies retryable errors (`ProviderRateLimitError`, `ProviderTimeoutError`, `GRPCUnavailableError`, `TimeoutError`, socket disconnects) while rejecting permanent schema/validation failures.
- **Streaming Invariant**: Retries **never cross the streaming boundary** once the first token has been emitted.

---

## 4. Verification & Testing Matrix (Phases 1-3)

| Test Suite | Focus Area | Assertions & Scenarios | Status |
|---|---|---|---|
| `tests/unit/test_config.py` | Settings & Config | Default parameters, NVIDIA key aliases, invalid log level rejection, loop iterations validation | PASSED |
| `tests/unit/test_health.py` | Health Probes | `/health`, `/healthz`, `/ready`, `/readyz`, `/metrics`, root route, failure simulation | PASSED |
| `tests/unit/test_models.py` | Core Models & Events | Canonical Kafka event serialization/deserialization, PipelineContext, WorkflowResult normalization | PASSED |
| `tests/unit/test_exceptions.py` | Exception Hierarchies | Exception inheritance, status codes, ModeHandlerState initialization | PASSED |
| `tests/kafka/test_kafka_infrastructure.py` | Kafka Messaging | Publisher headers, DLQ, TTLCache idempotency deduplication, consumer retry loop, offset commits | PASSED |
| `scripts/check_import_boundaries.py` | Architectural Boundary | AST-based import check across all modules (0 violations) | PASSED |

---

## 5. Summary of Git History (Phases 1-3)

- **Commit `7820f3a`**: `feat(bootstrap): initialize Phase 1 project bootstrap`
- **Commit `6e33889`**: `feat(models): implement Core Models, Schemas, Enums, and Exception hierarchies (Phase 2)`
- **Commit `591c88d`**: `feat(kafka): implement Kafka Consumer, Producer, Manual Offset Commits, Idempotency, DLQ, and Retries (Phase 3)`
