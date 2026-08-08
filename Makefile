.PHONY: help install sync dev run test test-cov lint format check-imports infra-up infra-down infra-status infra-logs-kafka docker-build docker-up docker-down clean

help:
	@echo "Available make commands:"
	@echo "  --- Infrastructure (Existing Docker Containers) ---"
	@echo "  infra-up          - Start existing GraphGPT docker containers (Kafka, Redis, Neo4j, Cassandra, Milvus)"
	@echo "  infra-down        - Stop GraphGPT docker containers"
	@echo "  infra-status      - Check status of GraphGPT containers"
	@echo "  infra-logs-kafka  - Follow Kafka container logs"
	@echo ""
	@echo "  --- Application & Environment ---"
	@echo "  install           - Install dependencies with uv"
	@echo "  sync              - Sync environment with uv sync"
	@echo "  dev               - Run FastAPI local development server (with reload)"
	@echo "  run               - Run FastAPI production server"
	@echo "  test              - Run pytest test suite"
	@echo "  test-cov          - Run pytest with coverage report"
	@echo "  lint              - Run ruff linter and mypy type checks"
	@echo "  format            - Format codebase with ruff"
	@echo "  check-imports     - Run architectural import boundary check"
	@echo ""
	@echo "  --- LLM Service Docker ---"
	@echo "  docker-build      - Build Docker image for llm-service"
	@echo "  docker-up         - Start llm-service via docker compose"
	@echo "  docker-down       - Stop llm-service via docker compose"
	@echo "  clean             - Clean python cache files and temp artifacts"

# Start existing docker containers for local GraphGPT stack
infra-up:
	-docker start graphgpt-kafka graphgpt-redis graphgpt-neo4j graphgpt-cassandra milvus-minio milvus-etcd milvus-standalone

infra-down:
	-docker stop graphgpt-kafka graphgpt-redis graphgpt-neo4j graphgpt-cassandra milvus-standalone milvus-etcd milvus-minio

infra-status:
	docker ps -a --filter "name=graphgpt" --filter "name=milvus"

infra-logs-kafka:
	docker logs -f graphgpt-kafka

install:
	uv pip install -e ".[dev]"

sync:
	uv sync --all-extras

dev:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	uv run pytest tests -v

test-cov:
	uv run pytest tests --cov=app --cov-report=term-missing

proto:
	uv run python -m grpc_tools.protoc -Iproto --python_out=app/grpc/proto --grpc_python_out=app/grpc/proto proto/memory.proto proto/graph.proto proto/retrieval.proto proto/llm_service.proto
	uv run python -c "import glob, re; [open(p, 'w', encoding='utf-8').write(re.sub(r'import (\w+_pb2) as (\w+__pb2)', r'from app.grpc.proto import \1 as \2', open(p, 'r', encoding='utf-8').read())) for p in glob.glob('app/grpc/proto/*_pb2_grpc.py')]"

lint:
	uv run ruff check app tests
	uv run mypy app

format:
	uv run ruff format app tests
	uv run ruff check --fix app tests

check-imports:
	uv run python scripts/check_import_boundaries.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache .mypy_cache
