"""
Retrieval Service (RAG / Vector DB) gRPC Client.
Implements LLD v2.0 Section 8 and HLD v2.0 Section 25.4.
"""

from grpc.aio import Channel

from app.context.schemas import DocumentChunk, RetrievalContext
from app.grpc.clients.base import BaseGRPCClient
from app.grpc.proto import retrieval_pb2, retrieval_pb2_grpc


class RetrievalServiceClient(BaseGRPCClient[retrieval_pb2_grpc.RetrievalServiceStub]):
    """
    gRPC client for Retrieval Service.
    Consumed exclusively by ContextCollector for document chunks and semantic knowledge base retrieval.
    """

    def _create_stub(self, channel: Channel) -> retrieval_pb2_grpc.RetrievalServiceStub:
        return retrieval_pb2_grpc.RetrievalServiceStub(channel)

    async def get_relevant_chunks(
        self,
        user_id: str,
        conversation_id: str = "",
        query: str = "",
        file_ids: list[str] | None = None,
        top_k: int = 5,
        min_relevance_score: float = 0.0,
        max_tokens: int = 4000,
        trace_id: str = "",
    ) -> RetrievalContext:
        """Fetch ranked document chunks for file query or general knowledge base search."""
        stub = await self.get_stub()
        file_ids_list = file_ids or []
        scope = (
            retrieval_pb2.RetrievalScope.FILES
            if file_ids_list
            else retrieval_pb2.RetrievalScope.ALL
        )

        req = retrieval_pb2.GetRelevantChunksRequest(
            user_id=user_id,
            conversation_id=conversation_id,
            query=query,
            top_k=top_k,
            scope=scope,
            file_ids=file_ids_list,
            min_relevance_score=min_relevance_score,
            max_tokens=max_tokens,
            trace_id=trace_id,
        )
        metadata = self.build_metadata(
            trace_id=trace_id, user_id=user_id, conversation_id=conversation_id
        )
        timeout = self.deadline_ms / 1000.0

        try:
            resp: retrieval_pb2.GetRelevantChunksResponse = await stub.GetRelevantChunks(
                req,
                metadata=metadata,
                timeout=timeout,
            )

            chunks = [
                DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    file_id=chunk.source_file_id,
                    content=chunk.content,
                    score=round(float(chunk.relevance_score), 4),
                    metadata=dict(chunk.metadata),
                )
                for chunk in resp.chunks
            ]

            return RetrievalContext(
                chunks=chunks,
                total_chunks=len(chunks),
                query=query,
            )
        except Exception as exc:
            self.handle_rpc_error(exc, service_name="RetrievalService")
            raise
