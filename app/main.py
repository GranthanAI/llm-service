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
from app.context.collector import ContextCollector
from app.context.merger import ContextMerger
from app.context_window import ContextWindowManager
from app.grpc.clients import (
    GraphServiceClient,
    MemoryServiceClient,
    RetrievalServiceClient,
)
from app.producers.kafka_producer import KafkaPublisher
from app.prompts import PromptBuilder, PromptLoader, PromptRegistry
from app.providers import (
    GeminiAdapter,
    GenerationRouter,
    GroqAdapter,
    NVIDIAAdapter,
)
from app.request_analyzer.analyzer import RequestAnalyzer
from app.request_analyzer.groq_client import GroqAnalysisClient
from app.request_analyzer.prompt_template import AnalysisPromptBuilder
from app.services.streaming_service import StreamingEngine
from app.tools.dispatcher import ToolDispatcher
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool
from app.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
)
from app.utils.error_handler import ErrorHandler
from app.utils.helpers import generate_request_id
from app.utils.metrics import REQUEST_DURATION, REQUESTS_TOTAL
from app.utils.retry import RetryManager, RetryPolicy
from app.utils.tracing import setup_tracing
from app.workflow_engine.engine import WorkflowEngine
from app.workflow_engine.langgraph_workflows import (
    DeepResearchGraphBuilder,
    SmartGraphBuilder,
)
from app.workflow_engine.mode_dispatcher import ModeDispatcher
from app.workflow_engine.mode_handlers import (
    AskFilesHandler,
    CodeHandler,
    DefaultHandler,
    TutorHandler,
    WebSearchHandler,
)


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

    # 5. Initialize Context Collector (Phase 5)
    container.context_collector = ContextCollector(
        memory_client=container.memory_client,
        graph_client=container.graph_client,
        retrieval_client=container.retrieval_client,
        merger=ContextMerger(),
    )

    # 6. Initialize Tool Framework (Phase 9 & 10: Multi-source Web Search)
    tool_registry = ToolRegistry()
    web_search_tool = WebSearchTool(
        timeout_ms=config.web_search_timeout_ms,
    )
    tool_registry.register(web_search_tool, enabled=config.enable_web_search)
    tool_dispatcher = ToolDispatcher(registry=tool_registry, executor=ToolExecutor())
    container.tool_registry = tool_registry
    container.tool_dispatcher = tool_dispatcher

    # 7. Initialize Request Analyzer (Phase 6)
    groq_client = GroqAnalysisClient(
        api_key=config.groq_api_key.get_secret_value() if config.groq_api_key else "",
        model=config.groq_model,
        timeout_ms=config.groq_timeout_ms,
    )
    container.request_analyzer = RequestAnalyzer(
        groq_client=groq_client,
        prompt_builder=AnalysisPromptBuilder(),
        circuit_breaker=CircuitBreaker(
            name="groq",
            config=CircuitBreakerConfig(failure_threshold=5, recovery_timeout_seconds=30),
        ),
        retry_manager=RetryManager(
            policy=RetryPolicy(max_attempts=2, initial_delay_ms=50, max_delay_ms=200)
        ),
        config=config,
    )

    # 8. Initialize Prompt Engine (Phase 13) & Context Window Manager (Phase 14)
    prompt_loader = PromptLoader()
    prompt_registry = PromptRegistry(loader=prompt_loader)
    prompt_builder = PromptBuilder(registry=prompt_registry)
    context_window_manager = ContextWindowManager()
    container.prompt_registry = prompt_registry
    container.prompt_builder = prompt_builder
    container.context_window_manager = context_window_manager

    # 9. Initialize Deterministic Mode Handlers (Phase 8), SmartGraph (Phase 11), & Mode Dispatcher (Phase 7)
    handlers = {
        "default": DefaultHandler(tool_dispatcher=tool_dispatcher, prompt_registry=prompt_registry),
        "tutor": TutorHandler(tool_dispatcher=tool_dispatcher, prompt_registry=prompt_registry),
        "code": CodeHandler(tool_dispatcher=tool_dispatcher, prompt_registry=prompt_registry),
        "ask_files": AskFilesHandler(),
        "web_search": WebSearchHandler(tool_dispatcher=tool_dispatcher),
    }
    smart_graph = SmartGraphBuilder(
        tool_dispatcher=tool_dispatcher,
        config=config,
    ).build()
    deep_research_graph = DeepResearchGraphBuilder(
        tool_dispatcher=tool_dispatcher,
        prompt_registry=prompt_registry,
        config=config,
    ).build()
    graphs = {
        "smart": smart_graph,
        "deep_research": deep_research_graph,
    }
    mode_dispatcher = ModeDispatcher(handlers=handlers, graphs=graphs)
    workflow_engine = WorkflowEngine(mode_dispatcher=mode_dispatcher)
    container.mode_dispatcher = mode_dispatcher
    container.workflow_engine = workflow_engine

    # 10. Initialize Provider Adapters & Generation Router (Phase 15)
    nvidia_adapter = NVIDIAAdapter(
        api_key=config.nvidia_api_key.get_secret_value(),
        model=config.nvidia_model,
        base_url=config.nvidia_base_url,
        timeout_s=config.nvidia_timeout_ms / 1000.0,
        temperature=config.nvidia_temperature,
        top_p=config.nvidia_top_p,
        max_tokens=config.nvidia_max_tokens,
    )
    gemini_adapter = GeminiAdapter(
        api_key=config.gemini_api_key.get_secret_value(),
        model=config.gemini_model,
    )
    groq_adapter = GroqAdapter(
        api_key=config.groq_api_key.get_secret_value(),
        model=config.groq_model,
        timeout_s=config.groq_timeout_ms / 1000.0,
    )
    # 10. Initialize Provider Adapters & Generation Router (Phase 15)
    cb_registry = CircuitBreakerRegistry(
        default_config=CircuitBreakerConfig(failure_threshold=3, recovery_timeout_seconds=30)
    )
    cb_nvidia = cb_registry.get_or_create("nvidia")
    cb_gemini = cb_registry.get_or_create("gemini")
    cb_groq = cb_registry.get_or_create("groq")
    cb_registry.get_or_create("memory_service")
    cb_registry.get_or_create("graph_service")
    cb_registry.get_or_create("retrieval_service")
    cb_registry.get_or_create("web_search")

    retry_manager = RetryManager(
        policy=RetryPolicy(max_attempts=3, initial_delay_ms=100, max_delay_ms=2000)
    )
    error_handler = ErrorHandler()

    container.circuit_breaker_registry = cb_registry
    container.retry_manager = retry_manager
    container.error_handler = error_handler

    generation_router = GenerationRouter(
        nvidia_adapter=nvidia_adapter,
        gemini_adapter=gemini_adapter,
        circuit_breakers={"nvidia": cb_nvidia, "gemini": cb_gemini},
    )
    container.nvidia_adapter = nvidia_adapter
    container.gemini_adapter = gemini_adapter
    container.groq_adapter = groq_adapter
    container.generation_router = generation_router

    # 11. Initialize Kafka Producer & Streaming Engine (Phase 16)
    publisher = KafkaPublisher(config=config)
    streaming_engine = StreamingEngine(publisher=publisher)
    container.kafka_producer = publisher
    container.streaming_engine = streaming_engine

    # 12. Initialize Kafka Consumer
    chat_consumer = ChatConsumer()
    consumer_engine = KafkaConsumerEngine(
        config=config,
        publisher=publisher,
        event_handler=chat_consumer.handle,
    )
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
