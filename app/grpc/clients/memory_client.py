"""
Memory Service gRPC Client.
Implements LLD v2.0 Section 8 and HLD v2.0 Section 25.2.
"""

from datetime import UTC, datetime

from grpc.aio import Channel

from app.context.schemas import Fact, MemoryContext, Message, Role
from app.grpc.clients.base import BaseGRPCClient
from app.grpc.proto import memory_pb2, memory_pb2_grpc


class MemoryServiceClient(BaseGRPCClient[memory_pb2_grpc.MemoryServiceStub]):
    """
    gRPC client for Memory Service.
    Consumed exclusively by ContextCollector for baseline memory context retrieval.
    """

    def _create_stub(self, channel: Channel) -> memory_pb2_grpc.MemoryServiceStub:
        return memory_pb2_grpc.MemoryServiceStub(channel)

    async def get_memory_context(
        self,
        user_id: str,
        conversation_id: str,
        query: str = "",
        max_tokens: int = 4000,
        trace_id: str = "",
    ) -> MemoryContext:
        """Fetch unified memory context (short-term history + long-term facts)."""
        stub = await self.get_stub()
        req = memory_pb2.GetMemoryContextRequest(
            conversation_id=conversation_id,
            user_id=user_id,
            query=query,
            scope=memory_pb2.MemoryScope.ALL,
            max_tokens=max_tokens,
            trace_id=trace_id,
        )
        metadata = self.build_metadata(
            trace_id=trace_id, user_id=user_id, conversation_id=conversation_id
        )
        timeout = self.deadline_ms / 1000.0

        try:
            resp: memory_pb2.GetMemoryContextResponse = await stub.GetMemoryContext(
                req,
                metadata=metadata,
                timeout=timeout,
            )

            messages = [
                Message(
                    role=Role(m.role.lower())
                    if m.role.lower() in [r.value for r in Role]
                    else Role.USER,
                    content=m.content,
                    timestamp=datetime.fromtimestamp(m.timestamp / 1000.0, tz=UTC)
                    if m.timestamp
                    else datetime.now(UTC),
                    name=m.message_id or None,
                )
                for m in resp.short_term_messages
            ]

            facts = [
                Fact(
                    fact_id=f"fact_{i}",
                    statement=f.content,
                    confidence=round(float(f.confidence), 4) if f.confidence > 0 else 1.0,
                    source=f.category or "long_term",
                    created_at=datetime.fromtimestamp(f.last_updated / 1000.0, tz=UTC)
                    if f.last_updated
                    else datetime.now(UTC),
                )
                for i, f in enumerate(resp.long_term_facts)
            ]

            return MemoryContext(
                short_term_messages=messages,
                long_term_facts=facts,
            )
        except Exception as exc:
            self.handle_rpc_error(exc, service_name="MemoryService")
            raise

    async def get_short_term_memory(
        self,
        user_id: str,
        conversation_id: str,
        limit: int = 20,
        trace_id: str = "",
    ) -> list[Message]:
        """Fetch short-term conversation history only."""
        stub = await self.get_stub()
        req = memory_pb2.GetShortTermMemoryRequest(
            conversation_id=conversation_id,
            user_id=user_id,
            limit=limit,
            trace_id=trace_id,
        )
        metadata = self.build_metadata(
            trace_id=trace_id, user_id=user_id, conversation_id=conversation_id
        )
        timeout = self.deadline_ms / 1000.0

        try:
            resp: memory_pb2.GetShortTermMemoryResponse = await stub.GetShortTermMemory(
                req,
                metadata=metadata,
                timeout=timeout,
            )
            return [
                Message(
                    role=Role(m.role.lower())
                    if m.role.lower() in [r.value for r in Role]
                    else Role.USER,
                    content=m.content,
                    timestamp=datetime.fromtimestamp(m.timestamp / 1000.0, tz=UTC)
                    if m.timestamp
                    else datetime.now(UTC),
                    name=m.message_id or None,
                )
                for m in resp.messages
            ]
        except Exception as exc:
            self.handle_rpc_error(exc, service_name="MemoryService")
            raise
