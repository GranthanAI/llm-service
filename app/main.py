"""
Main FastAPI Application Bootstrap.
Implements LLD v2.0 Section 3.1 and Section 29.1.
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.dependencies import Container, init_container
from app.api.internal.health import router as health_router
from app.api.internal.metrics import router as metrics_router
from app.api.internal.readiness import router as readiness_router
from app.api.routers import api_router
from app.config.logging import (
    correlation_request_id,
    get_logger,
    setup_logging,
)
from app.config.settings import LLMServiceConfig, get_settings
from app.consumers.chat_consumer import ChatConsumer
from app.consumers.kafka_consumer import KafkaConsumerEngine
from app.grpc.clients import (
    GraphServiceClient,
    MemoryServiceClient,
    RetrievalServiceClient,
)
from app.producers.kafka_producer import KafkaPublisher
from app.utils.helpers import generate_request_id
from app.utils.metrics import REQUEST_DURATION, REQUESTS_TOTAL
from app.utils.tracing import setup_tracing


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Service lifecycle manager.
    Initializes dependencies in order per LLD v2.0 Section 3.1.
    """
    config: LLMServiceConfig = get_settings()

    # 1. Initialize logging
    setup_logging(
        log_level=config.log_level, service_name=config.service_name, version=config.service_version
    )
    log = get_logger("bootstrap")
    log.info(
        "Service starting",
        service=config.service_name,
        version=config.service_version,
        environment=config.environment,
    )

    # 2. Initialize OTel tracer
    setup_tracing(service_name=config.service_name, endpoint=config.otel_endpoint)

    # 3. Initialize DI Container
    container: Container = init_container(config=config)
    app.state.container = container

    # 4. Initialize Baseline gRPC Clients (Context Collector providers only)
    container.memory_client = MemoryServiceClient(
        host=config.memory_service_host,
        port=config.memory_service_port,
        pool_size=config.grpc_max_connections,
        deadline_ms=config.grpc_deadline_ms,
        keepalive_enabled=config.feature_grpc_keepalive,
    )
    container.graph_client = GraphServiceClient(
        host=config.graph_service_host,
        port=config.graph_service_port,
        pool_size=config.grpc_max_connections,
        deadline_ms=config.grpc_deadline_ms,
        keepalive_enabled=config.feature_grpc_keepalive,
    )
    container.retrieval_client = RetrievalServiceClient(
        host=config.retrieval_service_host,
        port=config.retrieval_service_port,
        pool_size=config.grpc_max_connections,
        deadline_ms=config.grpc_deadline_ms,
        keepalive_enabled=config.feature_grpc_keepalive,
    )

    # 5. Initialize Kafka Producer & Consumer
    publisher = KafkaPublisher(config=config)
    chat_consumer = ChatConsumer()
    consumer_engine = KafkaConsumerEngine(
        config=config,
        publisher=publisher,
        event_handler=chat_consumer.handle,
    )
    container.kafka_producer = publisher
    container.kafka_consumer = consumer_engine

    if config.environment != "test":
        try:
            await publisher.start()
            await consumer_engine.start()
            log.info("Kafka Producer and Consumer started")
        except Exception as exc:
            log.warning("Kafka broker connection deferred or offline", error=str(exc))

    # Mark service ready to accept traffic
    container.mark_ready(True)
    log.info("Service boot completed successfully — ready to accept traffic")

    try:
        yield
    finally:
        log.info("Service shutting down — cleaning up resources")
        container.mark_ready(False)
        if container.kafka_consumer:
            try:
                await container.kafka_consumer.stop()
            except Exception:
                pass
        if container.kafka_producer:
            try:
                await container.kafka_producer.stop()
            except Exception:
                pass
        # Close gRPC connection pools
        if container.memory_client:
            await container.memory_client.close()
        if container.graph_client:
            await container.graph_client.close()
        if container.retrieval_client:
            await container.retrieval_client.close()
        log.info("Service shutdown completed")


def create_app() -> FastAPI:
    """Create and configure FastAPI instance."""
    config: LLMServiceConfig = get_settings()

    app = FastAPI(
        title="GraphGPT LLM Service",
        description="LLM Orchestration and Generation Engine for GraphGPT",
        version=config.service_version,
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request correlation and logging middleware
    @app.middleware("http")
    async def request_correlation_middleware(request: Request, call_next):
        req_id = request.headers.get("x-request-id") or generate_request_id()
        correlation_request_id.set(req_id)
        start_time = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start_time
        response.headers["x-request-id"] = req_id

        # Update metrics for HTTP requests
        if request.url.path not in ("/metrics", "/health", "/healthz", "/ready", "/readyz"):
            REQUESTS_TOTAL.labels(
                mode="http",
                skill="none",
                engine_type="api",
                provider="none",
                status=str(response.status_code),
            ).inc()
            REQUEST_DURATION.labels(
                mode="http",
                skill="none",
                engine_type="api",
                provider="none",
            ).observe(duration)

        return response

    # Mount internal routes at root level for Kubernetes probes / Prometheus scrapers
    app.include_router(health_router)
    app.include_router(readiness_router)
    app.include_router(metrics_router)

    # Mount API router
    app.include_router(api_router, prefix="/api")

    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse(
            {
                "service": config.service_name,
                "version": config.service_version,
                "status": "online",
            }
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "app.main:app", host=cfg.host, port=cfg.port, reload=(cfg.environment == "development")
    )
