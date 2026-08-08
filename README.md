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
