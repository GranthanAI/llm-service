"""
Knowledge Graph Service gRPC Client.
Implements LLD v2.0 Section 8 and HLD v2.0 Section 25.3.
"""

from grpc.aio import Channel

from app.context.schemas import EntityNode, GraphContext, RelationshipEdge
from app.grpc.clients.base import BaseGRPCClient
from app.grpc.proto import graph_pb2, graph_pb2_grpc


class GraphServiceClient(BaseGRPCClient[graph_pb2_grpc.GraphServiceStub]):
    """
    gRPC client for Knowledge Graph Service.
    Consumed exclusively by ContextCollector for baseline graph context retrieval.
    """

    def _create_stub(self, channel: Channel) -> graph_pb2_grpc.GraphServiceStub:
        return graph_pb2_grpc.GraphServiceStub(channel)

    async def get_graph_context(
        self,
        user_id: str,
        conversation_id: str,
        query: str = "",
        max_nodes: int = 50,
        max_depth: int = 2,
        max_tokens: int = 4000,
        trace_id: str = "",
    ) -> GraphContext:
        """Fetch relevant subgraph context (entities, relationships, text summary)."""
        stub = await self.get_stub()
        req = graph_pb2.GetGraphContextRequest(
            user_id=user_id,
            conversation_id=conversation_id,
            query=query,
            max_nodes=max_nodes,
            max_depth=max_depth,
            max_tokens=max_tokens,
            trace_id=trace_id,
        )
        metadata = self.build_metadata(
            trace_id=trace_id, user_id=user_id, conversation_id=conversation_id
        )
        timeout = self.deadline_ms / 1000.0

        try:
            resp: graph_pb2.GetGraphContextResponse = await stub.GetGraphContext(
                req,
                metadata=metadata,
                timeout=timeout,
            )

            entities = [
                EntityNode(
                    id=node.node_id,
                    name=node.label or node.node_id,
                    type=node.node_type or "Concept",
                    properties=dict(node.properties),
                    description=node.properties.get("description"),
                )
                for node in resp.nodes
            ]

            relationships = [
                RelationshipEdge(
                    source_id=rel.from_node_id,
                    target_id=rel.to_node_id,
                    relation_type=rel.relationship_type,
                    properties=dict(rel.properties),
                )
                for rel in resp.relationships
            ]

            return GraphContext(
                entities=entities,
                relationships=relationships,
                subgraph_summary=resp.subgraph_summary or None,
            )
        except Exception as exc:
            self.handle_rpc_error(exc, service_name="GraphService")
            raise

    async def get_nodes_by_ids(
        self,
        user_id: str,
        node_ids: list[str],
        trace_id: str = "",
    ) -> list[EntityNode]:
        """Fetch specific graph entities by ID list."""
        stub = await self.get_stub()
        req = graph_pb2.GetNodesByIdsRequest(
            user_id=user_id,
            node_ids=node_ids,
            trace_id=trace_id,
        )
        metadata = self.build_metadata(trace_id=trace_id, user_id=user_id)
        timeout = self.deadline_ms / 1000.0

        try:
            resp: graph_pb2.GetNodesByIdsResponse = await stub.GetNodesByIds(
                req,
                metadata=metadata,
                timeout=timeout,
            )
            return [
                EntityNode(
                    id=node.node_id,
                    name=node.label or node.node_id,
                    type=node.node_type or "Concept",
                    properties=dict(node.properties),
                )
                for node in resp.nodes
            ]
        except Exception as exc:
            self.handle_rpc_error(exc, service_name="GraphService")
            raise
