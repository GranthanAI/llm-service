"""
Context Merger, Deduplication, and Relevance Ranking.
Implements LLD v2.0 Section 9.1 and Section 9.6.
"""

import hashlib
from datetime import UTC, datetime

from app.context.schemas import (
    ContextBundle,
    DocumentChunk,
    EntityNode,
    Fact,
    GraphContext,
    MemoryContext,
    RelationshipEdge,
    RetrievalContext,
)


class ContextMerger:
    """
    Normalizes, deduplicates, and ranks baseline context gathered in parallel
    from Memory, Graph, and Retrieval services.
    """

    def merge(
        self,
        memory: MemoryContext | None = None,
        graph: GraphContext | None = None,
        retrieval: RetrievalContext | None = None,
        missing_sources: list[str] | None = None,
    ) -> ContextBundle:
        """Merge gathered contexts into a sanitized, deduplicated ContextBundle."""
        missing = missing_sources or []
        degraded = len(missing) > 0

        # Process and deduplicate retrieval document chunks
        sanitized_retrieval: RetrievalContext | None = None
        if retrieval and retrieval.chunks:
            deduped_chunks = self._deduplicate_chunks(retrieval.chunks, graph)
            ranked_chunks = self._rank_chunks(deduped_chunks)
            sanitized_retrieval = RetrievalContext(
                chunks=ranked_chunks,
                total_chunks=len(ranked_chunks),
                query=retrieval.query,
            )
        elif retrieval:
            sanitized_retrieval = retrieval

        # Process and deduplicate graph context
        sanitized_graph = self._deduplicate_graph(graph) if graph else None

        # Process and deduplicate memory context
        sanitized_memory = self._deduplicate_memory(memory) if memory else None

        return ContextBundle(
            memory=sanitized_memory,
            graph=sanitized_graph,
            retrieval=sanitized_retrieval,
            degraded=degraded,
            missing_sources=missing,
            collected_at=datetime.now(UTC),
        )

    def _generate_fingerprint(self, text: str) -> str:
        """Generate SHA-256 fingerprint for text snippet (first 200 chars normalized)."""
        normalized = text[:200].lower().strip().encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    def _deduplicate_chunks(
        self,
        chunks: list[DocumentChunk],
        graph: GraphContext | None = None,
    ) -> list[DocumentChunk]:
        """
        Deduplicate document chunks against each other and against graph node descriptions.
        Implements LLD v2.0 Section 9.6.
        """
        seen_fingerprints: set[str] = set()

        # Seed seen fingerprints with graph node descriptions if present
        if graph and graph.entities:
            for entity in graph.entities:
                if entity.description:
                    seen_fingerprints.add(self._generate_fingerprint(entity.description))

        unique_chunks: list[DocumentChunk] = []
        for chunk in chunks:
            fp = self._generate_fingerprint(chunk.content)
            if fp not in seen_fingerprints:
                seen_fingerprints.add(fp)
                unique_chunks.append(chunk)

        return unique_chunks

    def _rank_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Rank document chunks by relevance score in descending order."""
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    def _deduplicate_graph(self, graph: GraphContext) -> GraphContext:
        """Deduplicate graph nodes by ID and edges by (source, target, type)."""
        seen_node_ids: set[str] = set()
        unique_nodes: list[EntityNode] = []
        for node in graph.entities:
            if node.id not in seen_node_ids:
                seen_node_ids.add(node.id)
                unique_nodes.append(node)

        seen_edges: set[tuple[str, str, str]] = set()
        unique_edges: list[RelationshipEdge] = []
        for edge in graph.relationships:
            edge_key = (edge.source_id, edge.target_id, edge.relation_type)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                unique_edges.append(edge)

        return GraphContext(
            entities=unique_nodes,
            relationships=unique_edges,
            subgraph_summary=graph.subgraph_summary,
        )

    def _deduplicate_memory(self, memory: MemoryContext) -> MemoryContext:
        """Deduplicate long-term facts by statement fingerprint."""
        seen_fact_fps: set[str] = set()
        unique_facts: list[Fact] = []
        for fact in memory.long_term_facts:
            fp = self._generate_fingerprint(fact.statement)
            if fp not in seen_fact_fps:
                seen_fact_fps.add(fp)
                unique_facts.append(fact)

        return MemoryContext(
            short_term_messages=memory.short_term_messages,
            long_term_facts=unique_facts,
        )
