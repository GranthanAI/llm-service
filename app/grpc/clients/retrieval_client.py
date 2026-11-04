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
        dense_query: str | None = None,
        previous_chunk_ids: list[str] | None = None,
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
            previous_chunk_ids=list(previous_chunk_ids or []),
        )
        if dense_query:                       # optional field: unset unless there is a rewrite to send
            req.dense_query = dense_query
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

            chunks = _map_chunks(resp.chunks)

            # An older Retrieval Service never sets `status` (reads as OK even when empty).
            status = retrieval_pb2.RetrievalStatus.Name(resp.status)
            if status == "OK" and not chunks:
                status = "NO_MATCH"
            return RetrievalContext(
                chunks=chunks,
                total_chunks=len(chunks),
                query=query,
                status=status,
                status_detail=resp.status_detail,
                requested_count=resp.requested_count,
                authorized_count=resp.authorized_count,
                ready_count=resp.ready_count,
                coverage=retrieval_pb2.Coverage.Name(resp.coverage),
                top_score=float(resp.top_score) if resp.HasField("top_score") else None,
                total_tokens=resp.total_tokens,
            )
        except Exception as exc:
            self.handle_rpc_error(exc, service_name="RetrievalService")
            raise

    async def get_document_chunks(
        self,
        user_id: str,
        file_ids: list[str],
        max_tokens: int = 0,
        trace_id: str = "",
        conversation_id: str = "",
    ) -> RetrievalContext:
        """Every chunk of the given files in document order, up to `max_tokens` (`truncated` is set when cut short).

        Raises GRPCError for an UNIMPLEMENTED (older) server; callers fall back to top-k retrieval."""
        stub = await self.get_stub()
        req = retrieval_pb2.GetDocumentChunksRequest(
            user_id=user_id, file_ids=list(file_ids), max_tokens=max_tokens, trace_id=trace_id
        )
        metadata = self.build_metadata(trace_id=trace_id, user_id=user_id, conversation_id=conversation_id)
        try:
            resp = await stub.GetDocumentChunks(req, metadata=metadata, timeout=self.deadline_ms / 1000.0)
            chunks = _map_chunks(resp.chunks)
            status = retrieval_pb2.RetrievalStatus.Name(resp.status)
            if status == "OK" and not chunks:
                status = "NO_MATCH"
            return RetrievalContext(
                chunks=chunks,
                total_chunks=len(chunks),
                status=status,
                status_detail=resp.status_detail,
                requested_count=resp.requested_count,
                authorized_count=resp.authorized_count,
                ready_count=resp.ready_count,
                full_text=True,
                truncated=resp.truncated,
                total_tokens=resp.total_tokens,
            )
        except Exception as exc:
            self.handle_rpc_error(exc, service_name="RetrievalService")
            raise


def _map_chunks(proto_chunks) -> list[DocumentChunk]:
    chunks = []
    for chunk in proto_chunks:
        metadata = dict(chunk.metadata)
        page = metadata.get("page_number", "")
        chunks.append(
            DocumentChunk(
                chunk_id=chunk.chunk_id,
                file_id=chunk.source_file_id,
                source_name=chunk.source_name,
                content=chunk.content,
                score=round(float(chunk.relevance_score), 4),
                page_number=int(page) if page.isdigit() else None,
                section=metadata.get("section", ""),
                metadata=metadata,
            )
        )
    return chunks
