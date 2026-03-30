"""Indexing embedder with title caching and batch optimization."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

from agent_rag.core.models import Chunk
from agent_rag.embedding.interface import Embedder
from agent_rag.ingestion.embeddings.config import EmbeddingConfig
from agent_rag.ingestion.embeddings.models import ChunkEmbedding, IndexChunk

logger = logging.getLogger(__name__)


class IndexingEmbedder(ABC):
    """
    Abstract base class for indexing embedders.

    Defines interface for converting Chunk models to IndexChunk with embeddings.
    """

    @abstractmethod
    async def embed_chunks(
        self,
        chunks: list[Chunk],
        config: EmbeddingConfig,
    ) -> list[IndexChunk]:
        """
        Embed chunks and return IndexChunk instances.

        Args:
            chunks: List of chunks to embed
            config: Embedding configuration

        Returns:
            List of IndexChunk with embeddings

        Raises:
            Exception: If embedding fails
        """
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear any internal caches."""
        pass


class DefaultIndexingEmbedder(IndexingEmbedder):
    """
    Default indexing embedder with title caching and batch optimization.

    Features:
    - Title embedding caching [P1] - avoid redundant title computations
    - Batch embedding optimization
    - Mini-chunk embedding support (multipass indexing)
    - Contextual RAG embedding (optional averaging)
    - LRU cache for title embeddings with configurable size
    """

    def __init__(self, embedder: Embedder):
        """
        Initialize indexing embedder.

        Args:
            embedder: Core embedder instance for embedding API calls
        """
        self.embedder = embedder

        # [P1] Title embedding cache - LRU cache with configurable size
        self._title_embed_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_max_size: int = 1000  # Will be updated from config

    async def embed_chunks(
        self,
        chunks: list[Chunk],
        config: EmbeddingConfig,
    ) -> list[IndexChunk]:
        """
        Embed chunks with title caching and batch optimization.

        Process:
        1. Group chunks by document_id for title caching
        2. For each document:
           - Embed title once and cache (if enabled)
           - Assemble chunk texts with title/metadata (if enabled)
           - Embed chunk contents in batches
           - Embed mini-chunks if multipass enabled
           - Optionally average summary embeddings
        3. Return IndexChunk instances

        Args:
            chunks: List of chunks to embed
            config: Embedding configuration

        Returns:
            List of IndexChunk with embeddings
        """
        # Update cache size from config
        self._cache_max_size = config.title_embed_cache_size

        # Group chunks by document_id for title caching
        chunks_by_doc: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            doc_id = chunk.document_id
            if doc_id not in chunks_by_doc:
                chunks_by_doc[doc_id] = []
            chunks_by_doc[doc_id].append(chunk)

        # Embed chunks document by document
        indexed_chunks: list[IndexChunk] = []

        for doc_id, doc_chunks in chunks_by_doc.items():
            logger.info(
                f"Embedding {len(doc_chunks)} chunks for document {doc_id} "
                f"(title_cache={'enabled' if config.cache_title_embeddings else 'disabled'})"
            )

            # Get or compute title embedding (if enabled)
            title_embedding: Optional[list[float]] = None
            if config.enable_title_embedding and doc_chunks[0].title:
                title_embedding = await self._get_title_embedding(
                    doc_chunks[0].title, config
                )

            # Embed each chunk
            for chunk in doc_chunks:
                try:
                    # Assemble chunk text for embedding
                    chunk_text = self._assemble_chunk_text(chunk, config)

                    # Embed full chunk content (run sync method in thread)
                    full_embedding = await asyncio.to_thread(self.embedder.embed, chunk_text)

                    # Normalize if configured
                    if config.normalize_embeddings:
                        full_embedding = self._normalize_embedding(full_embedding)

                    # [P0] Embed mini-chunks if multipass enabled
                    mini_chunk_embeddings: list[list[float]] = []
                    if chunk._mini_chunk_texts:
                        mini_chunk_embeddings = await self._embed_mini_chunks(
                            chunk._mini_chunk_texts, config
                        )

                    # [P1] Optionally average summary embeddings into full_embedding
                    if (
                        config.average_summary_embeddings
                        and config.enable_title_embedding
                    ):
                        full_embedding = await self._average_summary_embeddings(
                            chunk, full_embedding, config
                        )

                    # Create ChunkEmbedding
                    chunk_embedding = ChunkEmbedding(
                        full_embedding=full_embedding,
                        mini_chunk_embeddings=mini_chunk_embeddings,
                    )

                    # Create IndexChunk
                    index_chunk = IndexChunk(
                        chunk=chunk,
                        embeddings=chunk_embedding,
                        title_embedding=title_embedding,
                    )

                    indexed_chunks.append(index_chunk)

                except Exception as e:
                    logger.error(
                        f"Failed to embed chunk {chunk.chunk_id} "
                        f"of document {doc_id}: {e}"
                    )
                    raise

        logger.info(
            f"Successfully embedded {len(indexed_chunks)} chunks "
            f"(cache size: {len(self._title_embed_cache)})"
        )

        return indexed_chunks

    async def _get_title_embedding(
        self,
        title: str,
        config: EmbeddingConfig,
    ) -> list[float]:
        """
        Get or compute title embedding with caching [P1].

        Args:
            title: Document title
            config: Embedding configuration

        Returns:
            Title embedding vector
        """
        if not config.cache_title_embeddings:
            # No caching, embed directly (run sync method in thread)
            embedding = await asyncio.to_thread(self.embedder.embed, title)
            if config.normalize_embeddings:
                embedding = self._normalize_embedding(embedding)
            return embedding

        # Check cache
        if title in self._title_embed_cache:
            logger.debug(f"Title embedding cache hit: {title[:50]}...")
            # Move to end (LRU)
            self._title_embed_cache.move_to_end(title)
            return self._title_embed_cache[title]

        # Cache miss - compute and cache (run sync method in thread)
        logger.debug(f"Title embedding cache miss: {title[:50]}...")
        embedding = await asyncio.to_thread(self.embedder.embed, title)
        if config.normalize_embeddings:
            embedding = self._normalize_embedding(embedding)

        # Add to cache with LRU eviction
        self._title_embed_cache[title] = embedding
        self._title_embed_cache.move_to_end(title)

        # Evict oldest if cache exceeds max size
        while len(self._title_embed_cache) > self._cache_max_size:
            oldest_title = next(iter(self._title_embed_cache))
            del self._title_embed_cache[oldest_title]
            logger.debug(f"Evicted title from cache: {oldest_title[:50]}...")

        return embedding

    def _assemble_chunk_text(
        self,
        chunk: Chunk,
        config: EmbeddingConfig,
    ) -> str:
        """
        Assemble chunk text for embedding.

        Combines title prefix, content, and metadata suffix based on config.

        Args:
            chunk: Chunk to assemble text from
            config: Embedding configuration

        Returns:
            Assembled text for embedding
        """
        parts = []

        # Add embedding prefix if configured
        if config.embedding_prefix:
            parts.append(config.embedding_prefix)

        # Add title prefix (if enabled and available)
        if config.title_prefix_enabled and chunk._title_prefix:
            parts.append(chunk._title_prefix)

        # Add main content
        parts.append(chunk.content)

        # Add metadata suffix (if enabled and available)
        if config.metadata_suffix_enabled:
            if config.use_semantic_metadata and chunk._metadata_suffix_semantic:
                parts.append(chunk._metadata_suffix_semantic)
            elif chunk._metadata_suffix_keyword:
                parts.append(chunk._metadata_suffix_keyword)

        # Join parts with newlines
        assembled_text = "\n\n".join(parts)

        # Truncate if needed
        if config.truncate_long_texts:
            assembled_text = self._truncate_text(
                assembled_text, config.max_input_tokens
            )

        return assembled_text

    async def _embed_mini_chunks(
        self,
        mini_chunk_texts: list[str],
        config: EmbeddingConfig,
    ) -> list[list[float]]:
        """
        Embed mini-chunks for multipass indexing [P0].

        Args:
            mini_chunk_texts: List of mini-chunk texts
            config: Embedding configuration

        Returns:
            List of mini-chunk embeddings
        """
        if not mini_chunk_texts:
            return []

        logger.debug(f"Embedding {len(mini_chunk_texts)} mini-chunks")

        # Embed mini-chunks in batch (run sync method in thread)
        mini_embeddings = await asyncio.to_thread(self.embedder.embed_batch, mini_chunk_texts)

        # Normalize if configured
        if config.normalize_embeddings:
            mini_embeddings = [
                self._normalize_embedding(emb) for emb in mini_embeddings
            ]

        return mini_embeddings

    async def _average_summary_embeddings(
        self,
        chunk: Chunk,
        content_embedding: list[float],
        config: EmbeddingConfig,
    ) -> list[float]:
        """
        Average doc_summary and chunk_context embeddings into full_embedding [P1].

        Args:
            chunk: Chunk with contextual RAG content
            content_embedding: Embedding of chunk content
            config: Embedding configuration

        Returns:
            Averaged embedding
        """
        embeddings_to_average = [content_embedding]

        # Embed doc_summary if present (run sync method in thread)
        if chunk.doc_summary:
            doc_summary_emb = await asyncio.to_thread(self.embedder.embed, chunk.doc_summary)
            if config.normalize_embeddings:
                doc_summary_emb = self._normalize_embedding(doc_summary_emb)
            embeddings_to_average.append(doc_summary_emb)

        # Embed chunk_context if present (run sync method in thread)
        if chunk.chunk_context:
            chunk_context_emb = await asyncio.to_thread(self.embedder.embed, chunk.chunk_context)
            if config.normalize_embeddings:
                chunk_context_emb = self._normalize_embedding(chunk_context_emb)
            embeddings_to_average.append(chunk_context_emb)

        # Average embeddings
        if len(embeddings_to_average) == 1:
            return content_embedding

        logger.debug(
            f"Averaging {len(embeddings_to_average)} embeddings "
            f"(content + summary fields)"
        )

        dim = len(content_embedding)
        averaged = [0.0] * dim

        for emb in embeddings_to_average:
            for i in range(dim):
                averaged[i] += emb[i]

        for i in range(dim):
            averaged[i] /= len(embeddings_to_average)

        # Normalize averaged embedding
        if config.normalize_embeddings:
            averaged = self._normalize_embedding(averaged)

        return averaged

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        """
        Normalize embedding to unit length.

        Args:
            embedding: Embedding vector

        Returns:
            Normalized embedding
        """
        import math

        magnitude = math.sqrt(sum(x * x for x in embedding))

        if magnitude == 0:
            return embedding

        return [x / magnitude for x in embedding]

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to maximum token count.

        Uses approximate tokenization (words * 1.3) for efficiency.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens

        Returns:
            Truncated text
        """
        # Approximate: 1 word ≈ 1.3 tokens
        max_words = int(max_tokens / 1.3)

        words = text.split()
        if len(words) <= max_words:
            return text

        logger.warning(
            f"Truncating text from {len(words)} words to {max_words} "
            f"(~{max_tokens} tokens)"
        )

        return " ".join(words[:max_words])

    def clear_cache(self) -> None:
        """Clear title embedding cache."""
        cache_size = len(self._title_embed_cache)
        self._title_embed_cache.clear()
        logger.info(f"Cleared title embedding cache ({cache_size} entries)")

    def get_cache_stats(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with cache size and max size
        """
        return {
            "cache_size": len(self._title_embed_cache),
            "cache_max_size": self._cache_max_size,
        }
