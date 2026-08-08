"""
Dependency Injection and Application Container.
Implements LLD v2.0 Section 4.1.
"""

from typing import Annotated, Any

from fastapi import Depends, Request

from app.config.settings import LLMServiceConfig, get_settings


class Container:
    """
    Application Dependency Injection Container.
    Holds initialized singletons, clients, registries, and runtime state.
    """

    def __init__(self, config: LLMServiceConfig):
        self.config: LLMServiceConfig = config
        self.is_ready: bool = False
        self.is_healthy: bool = True

        # Placeholders for phase 2+ infrastructure components
        self.memory_client: Any | None = None
        self.graph_client: Any | None = None
        self.retrieval_client: Any | None = None
        self.tool_registry: Any | None = None
        self.prompt_registry: Any | None = None
        self.workflow_engine: Any | None = None
        self.kafka_producer: Any | None = None
        self.kafka_consumer: Any | None = None
        self.generation_router: Any | None = None

    def mark_ready(self, ready: bool = True) -> None:
        """Update readiness state for Kubernetes / readiness probes."""
        self.is_ready = ready

    def mark_healthy(self, healthy: bool = True) -> None:
        """Update health state for Kubernetes / liveness probes."""
        self.is_healthy = healthy


_CONTAINER: Container | None = None


def init_container(config: LLMServiceConfig | None = None) -> Container:
    """Initialize global Container instance."""
    global _CONTAINER
    if config is None:
        config = get_settings()
    _CONTAINER = Container(config=config)
    return _CONTAINER


def get_container(request: Request) -> Container:
    """FastAPI dependency to retrieve the application Container."""
    global _CONTAINER
    if hasattr(request.app.state, "container"):
        return request.app.state.container  # type: ignore[no-any-return]
    if _CONTAINER is None:
        _CONTAINER = Container(config=get_settings())
    return _CONTAINER


def get_config() -> LLMServiceConfig:
    """FastAPI dependency to retrieve current settings."""
    return get_settings()


ContainerDep = Annotated[Container, Depends(get_container)]
ConfigDep = Annotated[LLMServiceConfig, Depends(get_config)]
