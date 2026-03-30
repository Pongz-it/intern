"""Embedding models and data structures."""

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_rag.core.models import Chunk


@dataclass
class ChunkEmbedding:
    """
    Embedding structure for a chunk.

    Separates full chunk embedding from mini-chunk embeddings for multipass indexing.

    Design rationale (from Onyx embedder.py):
    - full_embedding: Used for standard retrieval and large chunk retrieval
    - mini_chunk_embeddings: Used for multipass indexing (higher precision recall)
    """

    full_embedding: list[float]
    """Full chunk embedding vector (dimension = config.model_dimension)."""

    mini_chunk_embeddings: list[list[float]] = field(default_factory=list)
    """List of mini-chunk embeddings for multipass indexing (if enabled)."""

    def __post_init__(self):
        """Validate embedding structure."""
        if not self.full_embedding:
            raise ValueError("full_embedding cannot be empty")

        if self.mini_chunk_embeddings:
            full_dim = len(self.full_embedding)
            for i, mini_emb in enumerate(self.mini_chunk_embeddings):
                if len(mini_emb) != full_dim:
                    raise ValueError(
                        f"Mini-chunk embedding {i} dimension {len(mini_emb)} "
                        f"doesn't match full embedding dimension {full_dim}"
                    )


@dataclass
class IndexChunk:
    """
    Complete chunk with embeddings ready for indexing.

    Combines Chunk model with ChunkEmbedding and optional title embedding.
    """

    chunk: Chunk
    """Original chunk with content and metadata."""

    embeddings: ChunkEmbedding
    """Chunk embeddings (full + mini-chunks)."""

    title_embedding: Optional[list[float]] = None
    """
    Optional separate title embedding for caching optimization.

    If config.enable_title_embedding is True, title is embedded separately
    and cached to avoid redundant computation across chunks of same document.
    """

    @property
    def chunk_id(self) -> int:
        """Convenience accessor for chunk ID."""
        return self.chunk.chunk_id

    @property
    def document_id(self) -> str:
        """Convenience accessor for document ID."""
        return self.chunk.document_id

    @property
    def full_embedding(self) -> list[float]:
        """Convenience accessor for full chunk embedding."""
        return self.embeddings.full_embedding

    @property
    def mini_chunk_embeddings(self) -> list[list[float]]:
        """Convenience accessor for mini-chunk embeddings."""
        return self.embeddings.mini_chunk_embeddings

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dict with chunk data and embeddings
        """
        result = {
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "content": self.chunk.content,
            "full_embedding": self.full_embedding,
        }

        if self.mini_chunk_embeddings:
            result["mini_chunk_embeddings"] = self.mini_chunk_embeddings

        if self.title_embedding:
            result["title_embedding"] = self.title_embedding

        # Include chunk metadata
        if self.chunk.title:
            result["title"] = self.chunk.title
        if self.chunk.source_type:
            result["source_type"] = self.chunk.source_type
        if self.chunk.link:
            result["link"] = self.chunk.link
        if self.chunk.metadata:
            result["metadata"] = self.chunk.metadata

        # Include chunking metadata
        if self.chunk.semantic_identifier:
            result["semantic_identifier"] = self.chunk.semantic_identifier
        if self.chunk.metadata_suffix:
            result["metadata_suffix"] = self.chunk.metadata_suffix
        if self.chunk.blurb:
            result["blurb"] = self.chunk.blurb

        # Include contextual RAG fields
        if self.chunk.doc_summary:
            result["doc_summary"] = self.chunk.doc_summary
        if self.chunk.chunk_context:
            result["chunk_context"] = self.chunk.chunk_context

        # Include large chunk references
        if self.chunk.large_chunk_reference_ids:
            result["large_chunk_reference_ids"] = self.chunk.large_chunk_reference_ids

        return result


@dataclass
class FailedDocument:
    """
    Document that failed embedding with error details.

    Used for per-document failure isolation in batch embedding.
    """

    document_id: str
    """Document ID that failed."""

    chunks: list[Chunk]
    """Chunks that failed to embed."""

    error: str
    """Error message."""

    error_type: str
    """Error type (e.g., 'APIError', 'TimeoutError', 'ValidationError')."""

    retry_count: int = 0
    """Number of retry attempts."""

    recoverable: bool = True
    """Whether error is recoverable (should retry)."""


@dataclass
class EmbeddingBatchResult:
    """
    Result of batch embedding operation.

    Separates successful embeddings from failures for isolation handling.
    """

    indexed_chunks: list[IndexChunk]
    """Successfully embedded chunks ready for indexing."""

    failed_documents: list[FailedDocument]
    """Documents that failed embedding with error details."""

    total_chunks: int
    """Total number of chunks attempted."""

    successful_chunks: int
    """Number of successfully embedded chunks."""

    @property
    def success_rate(self) -> float:
        """Calculate embedding success rate."""
        if self.total_chunks == 0:
            return 0.0
        return self.successful_chunks / self.total_chunks

    @property
    def has_failures(self) -> bool:
        """Check if any documents failed."""
        return len(self.failed_documents) > 0

    def to_summary(self) -> dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dict with batch embedding statistics
        """
        return {
            "total_chunks": self.total_chunks,
            "successful_chunks": self.successful_chunks,
            "failed_chunks": self.total_chunks - self.successful_chunks,
            "failed_documents": len(self.failed_documents),
            "success_rate": self.success_rate,
            "errors": [
                {
                    "document_id": fd.document_id,
                    "error_type": fd.error_type,
                    "error": fd.error,
                    "retry_count": fd.retry_count,
                }
                for fd in self.failed_documents
            ],
        }


@dataclass
class EmbeddingFailure:
    """
    Specific chunk embedding failure details.

    Used internally for tracking individual chunk failures.
    """

    chunk_id: int
    """Chunk ID that failed."""

    document_id: str
    """Document ID containing the chunk."""

    error: str
    """Error message."""

    error_type: str
    """Error type."""

    text_length: int
    """Length of text that failed to embed."""

    retry_attempt: int = 0
    """Retry attempt number."""
