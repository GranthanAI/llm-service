# GraphGPT LLM Service

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg)](https://fastapi.tiangolo.com)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

The **LLM Service** is the central orchestration and inference engine for GraphGPT, responsible for:
- Scatter-gather baseline context collection (Memory, Knowledge Graph, Retrieval) via gRPC.
- Intent analysis, skill detection, and execution planning via Groq.
- Dispatching execution to deterministic **Mode Handlers** (Tutor, Code, Default, Web Search, Ask Files) or iterative **LangGraph** workflows (Smart Mode, Deep Research).
- Token-budget-aware prompt assembly and priority-based trimming.
- Streaming response generation (NVIDIA NIM primary, Gemini fallback) published token-by-token over Kafka.

## Architecture

See detailed specifications in [docs/lld2.md](docs/lld2.md) and [docs/hld2.md](docs/hld2.md).

## Quick Start

### 1. Prerequisites
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) for package management
- Docker (for GraphGPT local infrastructure: Kafka, Redis, Neo4j, Cassandra, Milvus)

### 2. Start GraphGPT Docker Infrastructure
To start the existing local docker containers (`graphgpt-kafka`, `graphgpt-redis`, `graphgpt-neo4j`, `graphgpt-cassandra`, `milvus`):
```bash
make infra-up
```
Check status:
```bash
make infra-status
```

### 3. Setup Python Environment
```bash
# Sync dependencies
make sync
# or
uv sync --all-extras
```

Copy `.env.example` to `.env` and fill in your API keys (pre-populated with your keys):
```bash
cp .env.example .env
```

### 4. Run Locally
```bash
make dev
```

The service will start at `http://localhost:8000`.

### 5. Health & Metrics Endpoints
- **Liveness Probe**: `GET http://localhost:8000/health` (or `/healthz`)
- **Readiness Probe**: `GET http://localhost:8000/ready` (or `/readyz`)
- **Prometheus Metrics**: `GET http://localhost:8000/metrics`
- **Internal API Group**: `http://localhost:8000/api/internal/*`

### 6. Running Tests & Quality Checks
```bash
make test
make lint
make check-imports
```

## Billing integration (model groups, usage, Manim)

### Routing
`chat.message.created` 2.3 or 2.4 may carry a `billing` block (`hold_id`, `requested_model`, `granted_group`, `granted_model`,
`max_context_tokens`, `funded_by`, `downgraded`, `enforcement`). **No block, or `granted_group` null: the legacy
`GENERATION_PROVIDER_ORDER` chain, exactly as before.** With a granted group the request is answered only by models of that group:

- `granted_model` set: that model first, then the rest of its group in `group_order`; Auto (`granted_model` null): the group's `group_order`.
- Models that are disabled, whose provider has no key, or that have no adapter are skipped (logged once per model).
- **Never** a fallback outside the granted group and never a higher group. If the whole group fails the request ends in one
  `chat.response.generated` with `status: "error"` and the `calls` made so far.
- A grant is never answered by the legacy chain: if `config/models.yaml` cannot be loaded the service does not start.
- `qwen/qwen3.8-27b` and the NVIDIA model have no price in billing's catalog: legacy chain only, never group-routed.
- Input budget = `min(billing.max_context_tokens, model window, existing budget)` through the existing context-window manager.

### `config/models.yaml` (`MODEL_REGISTRY_FILE`)
Keyed by billing's catalog ids (exactly `model_groups` in billing's `plans.yaml`; a test compares against `tests/fixtures/billing_models.json`).
Per model: `provider` (`groq` | `google` | `anthropic` | `openai`), `provider_model` (the SDK string), `group`
(`fast|smart|best|ultra`), `context_window`, `max_output_tokens` (what we ask for), `enabled`. `group_order` sets the Auto order.
Unknown provider/group, a duplicate id or a `group_order` naming a foreign model is a startup error. Set `enabled: false` for any id
you cannot verify against the provider's docs. At startup and every 10 minutes the service compares the registry with
`GET {BILLING_SERVICE_URL}/billing/v1/models` (gauge `llm_catalog_model_missing{model,reason}`); it never blocks startup.

### Usage (`calls`)
Every model call of a request is a `CallUsage` (`provider`, `model` = catalog id where there is one, `purpose`, `input_tokens`,
`cached_input_tokens`, `output_tokens`, `usage_estimated`), collected per request and published on `chat.response.generated` 2.2
and `chat.response.cancelled`. Purposes: `answer`, `request_analysis`, `query_rewrite`, `workflow`, `research`, `manim_plan`.
Tokens are what the **provider reported**; when it reports none they are estimated with the token counter, `usage_estimated` is
true and `llm_usage_estimated_total{provider,model}` increments. Characters are never counted as tokens. Each request publishes
exactly one terminal event (success, error or cancelled) per process; across a restart delivery is at-least-once with the same
`request_message_id`. `memory.update.requested` carries `request_message_id`. Web search providers are free; a paid one must emit
`usage.metered` (plan 4.4.5).

### Manim
`BILLING_RESERVE_MANIM=off|shadow|enforce` (default `off`). A new job calls `CheckAndReserve(MANIM_VIDEO, "{job_id}:{attempt}")`
(150 ms deadline, breaker 5 failures / 30 s; `authorization: Bearer BILLING_SERVICE_TOKEN`, `x-correlation-id`). `enforce`: a deny
returns plan 4.6's error (429/403/402) and nothing is queued. Billing unreachable: the render proceeds with `hold_id` null. When a
job ends, `artifact.manim.rendered` 1.0 is published once per billing attempt (a failed job submitted again is attempt + 1); a job
that dies before rendering releases its hold. In the isolated worker deployment the worker reports the end of a job and its repair
calls to the API process (`/api/internal/visualizations/final`, `/repair`), which owns Kafka and the billing client.

### New environment variables
`MODEL_REGISTRY_FILE`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `BILLING_RESERVE_MANIM`, `BILLING_GRPC_TARGET`, `BILLING_SERVICE_TOKEN`,
`BILLING_SERVICE_URL`. On Windows use `KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092`, never `localhost`.
