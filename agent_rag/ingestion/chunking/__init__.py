"""Document chunking module.

Provides chunking functionality for document ingestion:
- BaseChunker: Abstract base class for chunkers
- ChunkerRegistry: Registry for chunker selection
- ChunkingConfig: Configuration for chunking behavior
- Specialized chunkers: Code, Table, Image
- Contextual RAG: Document summary and chunk context generation
"""

from agent_rag.ingestion.chunking.base import (
    BaseChunker,
    ChunkCandidate,
    count_tokens,
    split_text_by_sentences,
    truncate_to_tokens,
)
from agent_rag.ingestion.chunking.chunkers import (
    CodeChunker,
    ImageChunker,
    TableChunker,
)
from agent_rag.ingestion.chunking.config import (
    DEFAULT_CHUNKING_CONFIG,
    ChunkingConfig,
)
from agent_rag.ingestion.chunking.contextual_rag import (
    ContextualRAGGenerator,
    ContextualRAGResult,
    apply_contextual_rag,
    generate_contextual_rag_for_chunks,
    should_use_contextual_rag,
)
from agent_rag.ingestion.chunking.registry import (
    ChunkerRegistry,
    get_chunker_registry,
    register_chunker,
)

__all__ = [
    # Base
    "BaseChunker",
    "ChunkCandidate",
    "count_tokens",
    "truncate_to_tokens",
    "split_text_by_sentences",
    # Config
    "ChunkingConfig",
    "DEFAULT_CHUNKING_CONFIG",
    # Registry
    "ChunkerRegistry",
    "get_chunker_registry",
    "register_chunker",
    # Specialized chunkers
    "CodeChunker",
    "ImageChunker",
    "TableChunker",
    # Contextual RAG
    "ContextualRAGGenerator",
    "ContextualRAGResult",
    "apply_contextual_rag",
    "generate_contextual_rag_for_chunks",
    "should_use_contextual_rag",
]
