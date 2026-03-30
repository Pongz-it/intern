"""Core data models for Agent RAG."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class MessageRole(str, Enum):
    """Message role types."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Represents a tool call made by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]

    @property
    def parsed_arguments(self) -> dict[str, Any]:
        """Return parsed arguments (already structured as dict)."""
        return self.arguments

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        import json
        # Arguments must be a JSON string for OpenAI API
        args_str = json.dumps(self.arguments) if isinstance(self.arguments, dict) else str(self.arguments)
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": args_str,
            },
        }


@dataclass
class Citation:
    """Represents a citation to a source document."""
    citation_num: int
    document_id: str
    chunk_id: int
    content: str
    title: Optional[str] = None
    link: Optional[str] = None
    source_type: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "citation_num": self.citation_num,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "title": self.title,
            "link": self.link,
            "source_type": self.source_type,
        }


@dataclass
class Message:
    """Represents a message in the conversation."""
    role: str
    content: str
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    citations: Optional[list[Citation]] = None

    def to_llm_message(self) -> dict[str, Any]:
        """Convert to LLM message format."""
        msg: dict[str, Any] = {
            "role": self.role,
            "content": self.content if self.content is not None else "",
        }
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


@dataclass
class AgentResponse:
    """Response from an agent."""
    content: str
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    messages: list["Message"] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    reasoning: Optional[str] = None

    # Deep Research specific fields
    research_plan: Optional[str] = None
    intermediate_reports: Optional[list[str]] = None
    is_clarification: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "content": self.content,
            "citations": [c.to_dict() for c in self.citations],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
        }
        if self.reasoning:
            result["reasoning"] = self.reasoning
        if self.research_plan:
            result["research_plan"] = self.research_plan
        if self.intermediate_reports:
            result["intermediate_reports"] = self.intermediate_reports
        if self.is_clarification:
            result["is_clarification"] = self.is_clarification
        return result


@dataclass
class KGRelationship:
    """Knowledge graph relationship triplet."""
    source: str
    rel_type: str
    target: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format."""
        return {
            "source": self.source,
            "rel_type": self.rel_type,
            "target": self.target,
        }


@dataclass
class Chunk:
    """Represents a document chunk with enhanced features."""
    document_id: str
    chunk_id: int
    content: str
    embedding: Optional[list[float]] = None

    # Metadata
    title: Optional[str] = None
    source_type: Optional[str] = None
    link: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Enhanced metadata fields
    semantic_identifier: Optional[str] = None
    metadata_suffix: Optional[str] = None  # Natural language metadata description
    metadata_list: list[str] = field(default_factory=list)  # For array-based filtering
    blurb: Optional[str] = None  # Short excerpt for display
    section_continuation: bool = False

    # Multi-embedding support
    title_embedding: Optional[list[float]] = None
    embeddings: Optional[dict[str, list[float]]] = None  # Map of embedding_id -> vector
    skip_title_embedding: bool = False

    # Large chunk support
    large_chunk_reference_ids: list[int] = field(default_factory=list)

    # Knowledge graph fields
    kg_entities: list[str] = field(default_factory=list)
    kg_relationships: list[KGRelationship] = field(default_factory=list)
    kg_terms: list[str] = field(default_factory=list)

    # RAG context enhancement
    doc_summary: Optional[str] = None
    chunk_context: Optional[str] = None

    # Ranking fields
    boost: float = 0.0
    aggregated_chunk_boost_factor: float = 1.0
    hidden: bool = False

    # Ownership
    primary_owners: list[str] = field(default_factory=list)
    secondary_owners: list[str] = field(default_factory=list)

    # Search result fields
    score: float = 0.0
    match_highlights: list[str] = field(default_factory=list)
    content_summary: Optional[str] = None  # Dynamic summary with highlights

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    doc_updated_at: Optional[int] = None  # Unix timestamp for recency ranking

    # Multi-tenant support
    tenant_id: Optional[str] = None

    # Image support
    image_file_name: Optional[str] = None

    # Internal chunking fields (prefixed with _)
    _title_prefix: Optional[str] = None
    _metadata_suffix_semantic: Optional[str] = None
    _metadata_suffix_keyword: Optional[str] = None
    _mini_chunk_texts: list[str] = field(default_factory=list)

    @property
    def unique_id(self) -> str:
        """Get unique identifier for this chunk."""
        return f"{self.document_id}_{self.chunk_id}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "title": self.title,
            "source_type": self.source_type,
            "link": self.link,
            "metadata": self.metadata,
            "score": self.score,
        }
        # Add optional enhanced fields if present
        if self.semantic_identifier:
            result["semantic_identifier"] = self.semantic_identifier
        if self.metadata_suffix:
            result["metadata_suffix"] = self.metadata_suffix
        if self.doc_summary:
            result["doc_summary"] = self.doc_summary
        if self.chunk_context:
            result["chunk_context"] = self.chunk_context
        if self.boost != 0.0:
            result["boost"] = self.boost
        if self.kg_entities:
            result["kg_entities"] = self.kg_entities
        if self.kg_relationships:
            result["kg_relationships"] = [r.to_dict() for r in self.kg_relationships]
        if self.large_chunk_reference_ids:
            result["large_chunk_reference_ids"] = self.large_chunk_reference_ids
        if self.match_highlights:
            result["match_highlights"] = self.match_highlights
        if self.content_summary:
            result["content_summary"] = self.content_summary
        if self.primary_owners:
            result["primary_owners"] = self.primary_owners
        if self.secondary_owners:
            result["secondary_owners"] = self.secondary_owners
        if self.updated_at:
            result["updated_at"] = self.updated_at.isoformat()
        return result

    def to_vespa_fields(self) -> dict[str, Any]:
        """Convert to Vespa document fields format."""
        fields: dict[str, Any] = {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
        }

        # Content fields
        if self.title:
            fields["title"] = self.title
        if self.blurb:
            fields["blurb"] = self.blurb
        if self.semantic_identifier:
            fields["semantic_identifier"] = self.semantic_identifier
        if self.section_continuation:
            fields["section_continuation"] = self.section_continuation

        # Embedding fields (Vespa tensor format)
        if self.embedding:
            fields["embeddings"] = {"blocks": {"0": self.embedding}}
        if self.embeddings:
            fields["embeddings"] = {"blocks": self.embeddings}
        if self.title_embedding:
            fields["title_embedding"] = {"values": self.title_embedding}
        if self.skip_title_embedding:
            fields["skip_title"] = self.skip_title_embedding

        # Metadata fields
        if self.source_type:
            fields["source_type"] = self.source_type
        if self.link:
            fields["source_links"] = self.link
        if self.metadata:
            import json
            fields["metadata"] = json.dumps(self.metadata)
        if self.metadata_list:
            fields["metadata_list"] = self.metadata_list
        if self.metadata_suffix:
            fields["metadata_suffix"] = self.metadata_suffix
        if self.image_file_name:
            fields["image_file_name"] = self.image_file_name

        # RAG context fields
        if self.doc_summary:
            fields["doc_summary"] = self.doc_summary
        if self.chunk_context:
            fields["chunk_context"] = self.chunk_context
        if self.large_chunk_reference_ids:
            fields["large_chunk_reference_ids"] = self.large_chunk_reference_ids

        # Knowledge graph fields
        if self.kg_entities:
            fields["kg_entities"] = self.kg_entities
        if self.kg_relationships:
            fields["kg_relationships"] = [r.to_dict() for r in self.kg_relationships]
        if self.kg_terms:
            fields["kg_terms"] = self.kg_terms

        # Ranking fields
        if self.boost != 0.0:
            fields["boost"] = self.boost
        if self.aggregated_chunk_boost_factor != 1.0:
            fields["aggregated_chunk_boost_factor"] = self.aggregated_chunk_boost_factor
        if self.hidden:
            fields["hidden"] = self.hidden
        if self.doc_updated_at:
            fields["doc_updated_at"] = self.doc_updated_at

        # Ownership fields
        if self.primary_owners:
            fields["primary_owners"] = self.primary_owners
        if self.secondary_owners:
            fields["secondary_owners"] = self.secondary_owners

        # Multi-tenant
        if self.tenant_id:
            fields["tenant_id"] = self.tenant_id

        return fields


@dataclass
class Section:
    """Represents a section of a document (multiple consecutive chunks)."""
    center_chunk: Chunk
    chunks: list[Chunk]
    combined_content: str

    @property
    def document_id(self) -> str:
        """Get document ID."""
        return self.center_chunk.document_id

    @property
    def start_chunk_id(self) -> int:
        """Get starting chunk ID."""
        return min(c.chunk_id for c in self.chunks)

    @property
    def end_chunk_id(self) -> int:
        """Get ending chunk ID."""
        return max(c.chunk_id for c in self.chunks)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "document_id": self.document_id,
            "start_chunk_id": self.start_chunk_id,
            "end_chunk_id": self.end_chunk_id,
            "combined_content": self.combined_content,
            "center_chunk": self.center_chunk.to_dict(),
        }


@dataclass
class SearchFilters:
    """Filters for document search."""
    source_types: Optional[list[str]] = None
    time_cutoff: Optional[datetime] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    document_ids: Optional[list[str]] = None
    document_sets: Optional[list[str]] = None
    user_folder: Optional[int] = None
    user_project: Optional[list[int]] = None

    # Extension point for custom filters
    custom_filters: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result: dict[str, Any] = {}
        if self.source_types:
            result["source_types"] = self.source_types
        if self.time_cutoff:
            result["time_cutoff"] = self.time_cutoff.isoformat()
        if self.tags:
            result["tags"] = self.tags
        if self.metadata:
            result["metadata"] = self.metadata
        if self.document_ids:
            result["document_ids"] = self.document_ids
        if self.document_sets:
            result["document_sets"] = self.document_sets
        if self.user_folder is not None:
            result["user_folder"] = self.user_folder
        if self.user_project:
            result["user_project"] = self.user_project
        if self.custom_filters:
            result["custom_filters"] = self.custom_filters
        return result


@dataclass
class SearchResult:
    """Result from a search operation."""
    chunks: list[Chunk]
    sections: list[Section] = field(default_factory=list)
    total_hits: int = 0
    query: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "sections": [s.to_dict() for s in self.sections],
            "total_hits": self.total_hits,
            "query": self.query,
        }
