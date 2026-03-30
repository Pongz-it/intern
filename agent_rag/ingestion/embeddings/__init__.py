"""Embedding module with failure isolation and batch optimization.

This module provides:
- EmbeddingConfig: Configuration for embedding operations
- ChunkEmbedding: Embedding structure with full and mini-chunk embeddings
- IndexChunk: Complete chunk with embeddings ready for indexing
- FailedDocument: Failed document tracking with error details
- EmbeddingBatchResult: Batch embedding results with success/failure separation
- IndexingEmbedder: Abstract base class for indexing embedders
- DefaultIndexingEmbedder: Default implementation with title caching
- embed_chunks_with_failure_handling: Main entry point with failure isolation
"""

from agent_rag.ingestion.embeddings.config import (
    DEFAULT_EMBEDDING_CONFIG,
    EmbeddingConfig,
)
from agent_rag.ingestion.embeddings.embedder import (
    DefaultIndexingEmbedder,
    IndexingEmbedder,
)
from agent_rag.ingestion.embeddings.failure_handler import (
    embed_chunks_with_failure_handling,
    retry_failed_documents,
)
from agent_rag.ingestion.embeddings.models import (
    ChunkEmbedding,
    EmbeddingBatchResult,
    EmbeddingFailure,
    FailedDocument,
    IndexChunk,
)

__all__ = [
    # Configuration
    "EmbeddingConfig",
    "DEFAULT_EMBEDDING_CONFIG",
    # Models
    "ChunkEmbedding",
    "IndexChunk",
    "FailedDocument",
    "EmbeddingBatchResult",
    "EmbeddingFailure",
    # Embedders
    "IndexingEmbedder",
    "DefaultIndexingEmbedder",
    # Handlers
    "embed_chunks_with_failure_handling",
    "retry_failed_documents",
]
