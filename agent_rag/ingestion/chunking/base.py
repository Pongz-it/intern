"""Base chunker interface and extensibility framework."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from agent_rag.core.models import Chunk
from agent_rag.ingestion.chunking.config import ChunkingConfig
from agent_rag.ingestion.models import IngestionItem
from agent_rag.ingestion.parsing.base import ParsedDocument


class BaseChunker(ABC):
    """
    Base chunker interface for document-type-specific chunking strategies.

    All chunkers must implement:
    - supports(): Check if chunker can handle document/source type
    - chunk(): Generate chunks from parsed document

    Chunkers are selected by ChunkerRegistry based on:
    - Source type match
    - MIME type match
    - Document characteristics
    - Priority (higher priority chunkers tried first)
    """

    @abstractmethod
    def supports(self, source_type: str, mime_type: str, document: ParsedDocument) -> bool:
        """
        Check if this chunker supports the given document type.

        Args:
            source_type: Source type (file, url, text, markdown)
            mime_type: MIME type
            document: Parsed document to analyze

        Returns:
            True if chunker can handle this document type
        """
        pass

    @abstractmethod
    def chunk(
        self,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """
        Chunk document into pieces.

        Args:
            document: Parsed document with text, images, metadata
            item: Ingestion item for context (tenant_id, document_id, etc.)
            config: Chunking configuration

        Returns:
            List of Chunk objects ready for embedding

        Raises:
            Exception: If chunking fails
        """
        pass

    @property
    def priority(self) -> int:
        """
        Chunker priority for selection.

        Higher priority chunkers are tried first when multiple
        chunkers support the same document type.

        Default: 0
        Recommended range: -100 to 100

        Returns:
            Priority value (higher = tried first)
        """
        return 0

    @property
    def name(self) -> str:
        """
        Chunker name for logging and debugging.

        Default: Class name

        Returns:
            Chunker name
        """
        return self.__class__.__name__


# ============================================================================
# Helper Dataclasses for Internal Chunking State
# ============================================================================


class ChunkCandidate:
    """
    Internal representation of a chunk candidate during chunking.

    Used by chunkers to build chunks before converting to final Chunk model.
    """

    def __init__(
        self,
        chunk_id: int,
        content: str,
        title_prefix: str = "",
        metadata_suffix_semantic: str = "",
        metadata_suffix_keyword: str = "",
        blurb: str = "",
        section_continuation: bool = False,
        source_links: Optional[dict[int, str]] = None,
        image_file_id: Optional[str] = None,
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.title_prefix = title_prefix
        self.metadata_suffix_semantic = metadata_suffix_semantic
        self.metadata_suffix_keyword = metadata_suffix_keyword
        self.blurb = blurb
        self.section_continuation = section_continuation
        self.source_links = source_links or {}
        self.image_file_id = image_file_id

        # Will be populated later
        self.mini_chunk_texts: Optional[list[str]] = None
        self.large_chunk_reference_ids: list[int] = []
        self.doc_summary: str = ""
        self.chunk_context: str = ""
        self.contextual_rag_reserved_tokens: int = 0

    def to_chunk(
        self,
        document_id: str,
        title: Optional[str] = None,
        source_type: Optional[str] = None,
        link: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Chunk:
        """
        Convert ChunkCandidate to final Chunk model.

        Args:
            document_id: Document ID
            title: Document title
            source_type: Source type
            link: Source link
            metadata: Additional metadata

        Returns:
            Chunk instance
        """
        return Chunk(
            document_id=document_id,
            chunk_id=self.chunk_id,
            content=self.content,
            title=title,
            source_type=source_type,
            link=link,
            metadata=metadata or {},
            semantic_identifier=self.title_prefix,
            metadata_suffix=self.metadata_suffix_semantic,
            blurb=self.blurb,
            section_continuation=self.section_continuation,
            large_chunk_reference_ids=self.large_chunk_reference_ids,
            doc_summary=self.doc_summary,
            chunk_context=self.chunk_context,
            # Internal fields for chunking (prefixed with _)
            _title_prefix=self.title_prefix,
            _metadata_suffix_semantic=self.metadata_suffix_semantic,
            _metadata_suffix_keyword=self.metadata_suffix_keyword,
            _mini_chunk_texts=self.mini_chunk_texts,
        )


# ============================================================================
# Utility Functions for Chunking
# ============================================================================


def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """
    Count tokens in text.

    Uses tiktoken if available, otherwise approximates as words * 1.3.

    Args:
        text: Text to count tokens for
        tokenizer: Optional tokenizer instance

    Returns:
        Token count
    """
    if not text:
        return 0

    if tokenizer is not None:
        # Use provided tokenizer
        return len(tokenizer.encode(text))

    # Try tiktoken as default
    try:
        import tiktoken

        # Use cl100k_base (GPT-4, text-embedding-ada-002)
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        pass

    # Fallback: approximate as words * 1.3
    word_count = len(text.split())
    return int(word_count * 1.3)


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    tokenizer: Optional[Any] = None,
) -> str:
    """
    Truncate text to maximum token count.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens
        tokenizer: Optional tokenizer instance

    Returns:
        Truncated text
    """
    if count_tokens(text, tokenizer) <= max_tokens:
        return text

    # Binary search for correct length
    left, right = 0, len(text)

    while left < right:
        mid = (left + right + 1) // 2
        if count_tokens(text[:mid], tokenizer) <= max_tokens:
            left = mid
        else:
            right = mid - 1

    return text[:left]


def split_text_by_sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    Simple sentence boundary detection.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    import re

    # Simple sentence splitting
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)

    return [s.strip() for s in sentences if s.strip()]
