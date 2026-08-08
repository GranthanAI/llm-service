"""
Context Collector Data Schemas.
Implements LLD v2.0 Section 9 and HLD v2.0 Section 8.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Role(StrEnum):
    """Conversation message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """Normalized chat message representation across conversation history and prompt assembly."""

    role: Role = Role.USER
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Fact(BaseModel):
    """Long-term memory extracted fact about the user or preferences."""

    fact_id: str
    statement: str
    confidence: float = 1.0
    source: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryContext(BaseModel):
    """Memory service response bundle."""

    short_term_messages: list[Message] = Field(default_factory=list)
    long_term_facts: list[Fact] = Field(default_factory=list)


class EntityNode(BaseModel):
    """Knowledge graph entity node."""

    id: str
    name: str
    type: str = "Concept"
    properties: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


class RelationshipEdge(BaseModel):
    """Knowledge graph relationship edge connecting two entities."""

    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphContext(BaseModel):
    """Knowledge graph service response bundle."""

    entities: list[EntityNode] = Field(default_factory=list)
    relationships: list[RelationshipEdge] = Field(default_factory=list)
    subgraph_summary: str | None = None


class DocumentChunk(BaseModel):
    """Retrieved document chunk from vector/RAG service."""

    chunk_id: str
    file_id: str
    content: str
    score: float = 0.0
    page_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalContext(BaseModel):
    """Retrieval service response bundle."""

    chunks: list[DocumentChunk] = Field(default_factory=list)
    total_chunks: int = 0
    query: str | None = None


class ContextBundle(BaseModel):
    """
    Unified context payload from baseline providers (Memory, Graph, Retrieval).
    Constructed unconditionally on every request by ContextCollector.
    """

    memory: MemoryContext | None = None
    graph: GraphContext | None = None
    retrieval: RetrievalContext | None = None
    degraded: bool = False
    missing_sources: list[str] = Field(default_factory=list)
    collected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
