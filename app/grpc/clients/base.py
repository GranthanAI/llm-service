"""
Base gRPC Client with Channel Pool and Metadata Propagation.
Implements LLD v2.0 Section 8.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import grpc
import structlog
from grpc.aio import Channel

from app.config.logging import get_logger
from app.exceptions.grpc import GRPCError, GRPCTimeoutError, GRPCUnavailableError


class BaseGRPCClient[StubType](ABC):
    """
    Abstract Base gRPC Client with round-robin Channel Pool, keepalive options,
    deadline propagation, and metadata injection.
    """

    def __init__(
        self,
        host: str,
        port: int,
        pool_size: int = 20,
        deadline_ms: int = 2000,
        keepalive_enabled: bool = True,
        service_config_json: str | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ):
        self.host: str = host
        self.port: int = port
        self.pool_size: int = pool_size
        self.deadline_ms: int = deadline_ms
        self.keepalive_enabled: bool = keepalive_enabled
        self.service_config_json: str | None = service_config_json
        self.logger = logger or get_logger(self.__class__.__name__)

        self._channels: list[Channel] = []
        self._stubs: list[StubType] = []
        self._index: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()
        self._initialized: bool = False

    @property
    def target(self) -> str:
        return f"{self.host}:{self.port}"

    @abstractmethod
    def _create_stub(self, channel: Channel) -> StubType:
        """Instantiate specific gRPC service stub for a channel."""
        pass

    def _get_channel_options(self) -> list[tuple[str, Any]]:
        """Standard production HTTP/2 channel options per LLD v2.0 Section 8.3."""
        options: list[tuple[str, Any]] = [
            ("grpc.max_receive_message_length", 10485760),  # 10MB
            ("grpc.max_send_message_length", 2097152),  # 2MB
        ]
        if self.keepalive_enabled:
            options.extend(
                [
                    ("grpc.keepalive_time_ms", 30000),  # 30s
                    ("grpc.keepalive_timeout_ms", 5000),  # 5s
                    ("grpc.keepalive_permit_without_calls", 1),
                    ("grpc.http2.max_pings_without_data", 0),
                ]
            )
        if self.service_config_json:
            options.append(("grpc.service_config", self.service_config_json))
        return options

    async def initialize(self) -> None:
        """Initialize the connection pool channels and stubs."""
        if self._initialized:
            return

        target = self.target
        options = self._get_channel_options()

        self._channels = [
            grpc.aio.insecure_channel(target, options=options) for _ in range(self.pool_size)
        ]
        self._stubs = [self._create_stub(channel) for channel in self._channels]
        self._initialized = True
        self.logger.info(
            "Initialized gRPC connection pool", target=target, pool_size=self.pool_size
        )

    async def get_stub(self) -> StubType:
        """Round-robin channel selection from connection pool (LLD v2.0 Section 8.2)."""
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            idx = self._index % self.pool_size
            self._index += 1
            return self._stubs[idx]

    def build_metadata(
        self,
        trace_id: str | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[tuple[str, str]]:
        """Build standard service identity and correlation metadata per LLD Section 8.6."""
        meta: list[tuple[str, str]] = [("x-service-name", "llm-service")]
        if trace_id:
            meta.append(("x-trace-id", trace_id))
        if user_id:
            meta.append(("x-user-id", user_id))
        if conversation_id:
            meta.append(("x-conversation-id", conversation_id))
        return meta

    def handle_rpc_error(self, exc: Exception, service_name: str) -> None:
        """Translate gRPC AioRpcError into domain exceptions."""
        if isinstance(exc, grpc.aio.AioRpcError):
            code = exc.code()
            if code == grpc.StatusCode.UNAVAILABLE:
                raise GRPCUnavailableError(
                    f"{service_name} unavailable: {exc.details()}",
                    service=service_name,
                    code=code.value[0],
                ) from exc
            if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise GRPCTimeoutError(
                    f"{service_name} call timed out after {self.deadline_ms}ms",
                    service=service_name,
                    code=code.value[0],
                ) from exc
            raise GRPCError(
                f"{service_name} RPC failed: {exc.details()}",
                service=service_name,
                code=code.value[0],
            ) from exc
        raise exc

    async def close(self) -> None:
        """Gracefully close all channels in pool."""
        if not self._initialized:
            return
        close_tasks = [channel.close() for channel in self._channels]
        await asyncio.gather(*close_tasks, return_exceptions=True)
        self._channels.clear()
        self._stubs.clear()
        self._initialized = False
        self.logger.info("Closed gRPC connection pool", target=self.target)
