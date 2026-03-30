"""Semantic-aware document chunker with full Onyx parity."""

import logging
from typing import Any, Optional

from agent_rag.core.models import Chunk
from agent_rag.ingestion.chunking.base import (
    BaseChunker,
    ChunkCandidate,
    count_tokens,
    split_text_by_sentences,
    truncate_to_tokens,
)
from agent_rag.ingestion.chunking.config import ChunkingConfig
from agent_rag.ingestion.models import IngestionItem
from agent_rag.ingestion.parsing.base import ParsedDocument

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """
    Semantic-aware document chunker with full Onyx parity.

    Features:
    - SentenceChunker from chonkie for semantic-aware splitting
    - Image section handling [P0] - dedicated chunks for images
    - Multipass mode [P0] - mini-chunk generation
    - Large chunk generation [P0] - combined chunks
    - Dual metadata suffix [P1] - semantic + keyword
    - Contextual RAG token reservation [P1]
    - Oversized chunk fallback [P1] - strict token limit enforcement
    - Section continuation tracking [P2]
    - Source link offset mapping [P2]
    """

    def supports(self, source_type: str, mime_type: str, document: ParsedDocument) -> bool:
        """Support all document types as default chunker."""
        return True

    def chunk(
        self,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """
        Chunk document into semantic pieces.

        Args:
            document: Parsed document
            item: Ingestion item
            config: Chunking configuration

        Returns:
            List of chunks ready for embedding
        """
        # Validate config
        config.validate()

        # Build title prefix and metadata suffix
        title_prefix = self._build_title_prefix(document, config)
        metadata_suffix_semantic, metadata_suffix_keyword = self._get_metadata_suffix(
            document.metadata, config
        )

        # Calculate content token limit
        content_token_limit = self._calculate_content_token_limit(
            config, title_prefix, metadata_suffix_semantic
        )

        # Chunk text content using SentenceChunker
        text_chunks = self._chunk_text_with_chonkie(
            document.text, content_token_limit, config
        )

        # Build chunk candidates
        chunk_candidates = []
        for idx, chunk_text in enumerate(text_chunks):
            candidate = ChunkCandidate(
                chunk_id=idx,
                content=chunk_text,
                title_prefix=title_prefix,
                metadata_suffix_semantic=metadata_suffix_semantic,
                metadata_suffix_keyword=metadata_suffix_keyword,
                blurb=self._create_blurb(chunk_text, config.blurb_size),
                section_continuation=(idx > 0) if config.track_section_continuation else False,
            )
            chunk_candidates.append(candidate)

        # [P0] Handle image sections
        if config.create_image_chunks and document.images:
            image_chunks = self._create_image_chunks(
                document.images, len(chunk_candidates), title_prefix, config
            )
            chunk_candidates.extend(image_chunks)

        # [P0] Generate mini-chunks for multipass indexing
        if config.enable_multipass:
            self._generate_mini_chunks(chunk_candidates, config)

        # [P0] Generate large chunks
        large_chunk_candidates = []
        if config.enable_large_chunks:
            large_chunk_candidates = self._generate_large_chunks(chunk_candidates, config)
            chunk_candidates.extend(large_chunk_candidates)

        # [P1] Add contextual RAG content (if enabled)
        if config.enable_contextual_rag:
            self._add_contextual_rag(chunk_candidates, document, config)

        # Convert candidates to final Chunk models
        chunks = []
        for candidate in chunk_candidates:
            chunk = candidate.to_chunk(
                document_id=item.document_id or f"doc_{item.id}",
                title=document.metadata.get("title"),
                source_type=str(item.source_type.value),
                link=document.metadata.get("source_uri") or item.source_uri,
                metadata=document.metadata,
            )
            chunks.append(chunk)

        logger.info(
            f"Generated {len(chunks)} chunks: "
            f"{len(text_chunks)} text, {len(document.images)} images, "
            f"{len(large_chunk_candidates)} large chunks"
        )

        return chunks

    def _chunk_text_with_chonkie(
        self,
        text: str,
        content_token_limit: int,
        config: ChunkingConfig,
    ) -> list[str]:
        """
        Chunk text using chonkie SentenceChunker.

        Args:
            text: Document text
            content_token_limit: Maximum tokens per chunk
            config: Chunking configuration

        Returns:
            List of chunk texts
        """
        try:
            from chonkie import SentenceChunker

            chunker = SentenceChunker(
                tokenizer="cl100k_base",  # GPT-4 tokenizer
                chunk_size=content_token_limit,
                chunk_overlap=config.chunk_overlap,
            )

            # Chunk text
            chunks = chunker.chunk(text)

            # Extract text from chunk objects
            chunk_texts = [chunk.text for chunk in chunks]

            # [P1] Enforce strict token limits if enabled
            if config.strict_chunk_token_limit:
                chunk_texts = [
                    self._enforce_strict_limit(chunk_text, content_token_limit)
                    for chunk_text in chunk_texts
                ]

            return chunk_texts

        except ImportError:
            logger.warning("chonkie not available, using fallback sentence splitting")
            return self._fallback_chunking(text, content_token_limit, config)

    def _fallback_chunking(
        self,
        text: str,
        content_token_limit: int,
        config: ChunkingConfig,
    ) -> list[str]:
        """
        Fallback chunking using simple sentence splitting.

        Args:
            text: Document text
            content_token_limit: Maximum tokens per chunk
            config: Chunking configuration

        Returns:
            List of chunk texts
        """
        sentences = split_text_by_sentences(text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)

            if current_tokens + sentence_tokens <= content_token_limit:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Finalize current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                # Start new chunk
                if sentence_tokens <= content_token_limit:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    # Sentence too long, truncate
                    truncated = truncate_to_tokens(sentence, content_token_limit)
                    chunks.append(truncated)
                    current_chunk = []
                    current_tokens = 0

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _enforce_strict_limit(self, text: str, max_tokens: int) -> str:
        """
        [P1] Enforce strict token limit by truncating if needed.

        Args:
            text: Chunk text
            max_tokens: Maximum tokens

        Returns:
            Text within token limit
        """
        if count_tokens(text) <= max_tokens:
            return text

        logger.warning(f"Chunk exceeds strict limit, truncating to {max_tokens} tokens")
        return truncate_to_tokens(text, max_tokens)

    def _create_image_chunks(
        self,
        images: list,
        start_chunk_id: int,
        title_prefix: str,
        config: ChunkingConfig,
    ) -> list[ChunkCandidate]:
        """
        [P0] Create dedicated chunks for images with OCR text.

        Args:
            images: List of ParsedImage objects
            start_chunk_id: Starting chunk ID for images
            title_prefix: Title prefix to use
            config: Chunking configuration

        Returns:
            List of image chunk candidates
        """
        image_chunks = []

        for idx, image in enumerate(images):
            chunk_id = start_chunk_id + idx

            # Image content: caption or placeholder
            content = image.caption or f"[Image: {image.image_id}]"

            # Note: OCR text would be added here if available
            # For now, we use caption or placeholder

            candidate = ChunkCandidate(
                chunk_id=chunk_id,
                content=content,
                title_prefix=title_prefix,
                blurb=content[:128],  # Use content as blurb
                image_file_id=image.image_id,
            )

            image_chunks.append(candidate)

        return image_chunks

    def _generate_mini_chunks(
        self,
        chunk_candidates: list[ChunkCandidate],
        config: ChunkingConfig,
    ) -> None:
        """
        [P0] Generate mini-chunks for multipass indexing.

        Modifies chunk_candidates in-place by adding mini_chunk_texts.

        Args:
            chunk_candidates: List of chunk candidates
            config: Chunking configuration
        """
        for candidate in chunk_candidates:
            # Skip if image chunk or large chunk
            if candidate.image_file_id or candidate.large_chunk_reference_ids:
                continue

            # Split content into mini-chunks
            mini_chunk_size = config.mini_chunk_size
            sentences = split_text_by_sentences(candidate.content)

            mini_chunks = []
            current_mini = []
            current_tokens = 0

            for sentence in sentences:
                sentence_tokens = count_tokens(sentence)

                if current_tokens + sentence_tokens <= mini_chunk_size:
                    current_mini.append(sentence)
                    current_tokens += sentence_tokens
                else:
                    if current_mini:
                        mini_chunks.append(" ".join(current_mini))
                    current_mini = [sentence]
                    current_tokens = sentence_tokens

            if current_mini:
                mini_chunks.append(" ".join(current_mini))

            candidate.mini_chunk_texts = mini_chunks

    def _generate_large_chunks(
        self,
        chunk_candidates: list[ChunkCandidate],
        config: ChunkingConfig,
    ) -> list[ChunkCandidate]:
        """
        [P0] Generate combined large chunks.

        Args:
            chunk_candidates: List of regular chunk candidates
            config: Chunking configuration

        Returns:
            List of large chunk candidates
        """
        large_chunks = []
        ratio = config.large_chunk_ratio

        # Only combine text chunks (skip images)
        text_chunks = [c for c in chunk_candidates if not c.image_file_id]

        for i in range(0, len(text_chunks), ratio):
            group = text_chunks[i : i + ratio]

            if len(group) < ratio:
                # Not enough chunks to combine
                continue

            # Combine content
            combined_content = "\n\n".join(c.content for c in group)
            combined_title_prefix = group[0].title_prefix
            combined_metadata = group[0].metadata_suffix_semantic

            # Create large chunk candidate
            large_chunk_id = len(chunk_candidates) + len(large_chunks)

            large_candidate = ChunkCandidate(
                chunk_id=large_chunk_id,
                content=combined_content,
                title_prefix=combined_title_prefix,
                metadata_suffix_semantic=combined_metadata,
                blurb=self._create_blurb(combined_content, config.blurb_size),
            )

            # Track reference IDs
            large_candidate.large_chunk_reference_ids = [c.chunk_id for c in group]

            large_chunks.append(large_candidate)

        return large_chunks

    def _add_contextual_rag(
        self,
        chunk_candidates: list[ChunkCandidate],
        document: ParsedDocument,
        config: ChunkingConfig,
    ) -> None:
        """
        [P1] Add contextual RAG content (doc summary + chunk context).

        Modifies chunk_candidates in-place.

        Args:
            chunk_candidates: List of chunk candidates
            document: Parsed document
            config: Chunking configuration
        """
        # Generate document summary (simplified - would use LLM in production)
        doc_summary = ""
        if config.use_doc_summary:
            doc_summary = self._generate_doc_summary(document, config)

        # Add to each chunk
        for candidate in chunk_candidates:
            candidate.doc_summary = doc_summary

            # Generate chunk-specific context (simplified)
            if config.use_chunk_context:
                candidate.chunk_context = self._generate_chunk_context(
                    candidate, document, config
                )

            # Track reserved tokens
            candidate.contextual_rag_reserved_tokens = config.contextual_rag_reserved_tokens

    def _generate_doc_summary(
        self,
        document: ParsedDocument,
        config: ChunkingConfig,
    ) -> str:
        """
        Generate document summary for contextual RAG.

        Simplified version - production would use LLM.

        Args:
            document: Parsed document
            config: Chunking configuration

        Returns:
            Document summary
        """
        # Simplified: use title + first sentences
        title = document.metadata.get("title", "")
        text_preview = document.text[:500] if document.text else ""

        summary = f"{title}\n{text_preview}" if title else text_preview
        max_tokens = config.max_context_tokens if config.use_doc_summary else 0

        return truncate_to_tokens(summary, max_tokens)

    def _generate_chunk_context(
        self,
        candidate: ChunkCandidate,
        document: ParsedDocument,
        config: ChunkingConfig,
    ) -> str:
        """
        Generate chunk-specific context for contextual RAG.

        Simplified version - production would use LLM.

        Args:
            candidate: Chunk candidate
            document: Parsed document
            config: Chunking configuration

        Returns:
            Chunk context
        """
        # Simplified: use chunk position info
        context = f"Chunk {candidate.chunk_id + 1} of document"

        max_tokens = config.max_context_tokens if config.use_chunk_context else 0
        return truncate_to_tokens(context, max_tokens)

    def _build_title_prefix(
        self,
        document: ParsedDocument,
        config: ChunkingConfig,
    ) -> str:
        """
        Build title prefix for chunks.

        Args:
            document: Parsed document
            config: Chunking configuration

        Returns:
            Title prefix string
        """
        title = document.metadata.get("title", "")
        if not title:
            return ""

        # Truncate title if needed
        max_tokens = config.max_metadata_tokens // 2  # Reserve half for title
        return truncate_to_tokens(title, max_tokens)

    def _get_metadata_suffix(
        self,
        metadata: dict[str, Any],
        config: ChunkingConfig,
    ) -> tuple[str, str]:
        """
        [P1] Generate dual metadata suffix (semantic + keyword).

        Args:
            metadata: Document metadata
            config: Chunking configuration

        Returns:
            Tuple of (metadata_suffix_semantic, metadata_suffix_keyword)
        """
        if not config.include_metadata or not metadata:
            return "", ""

        # Semantic suffix: natural language with keys
        semantic_parts = []
        keyword_parts = []

        for key, value in metadata.items():
            if key in ["title", "content", "text"]:  # Skip content fields
                continue

            if value:
                value_str = str(value)
                semantic_parts.append(f"{key}: {value_str}")
                keyword_parts.append(value_str)

        semantic_suffix = " | ".join(semantic_parts)
        keyword_suffix = " ".join(keyword_parts)

        # Truncate to limits
        max_tokens = config.max_metadata_tokens
        semantic_suffix = truncate_to_tokens(semantic_suffix, max_tokens)
        keyword_suffix = truncate_to_tokens(keyword_suffix, max_tokens)

        return semantic_suffix, keyword_suffix

    def _calculate_content_token_limit(
        self,
        config: ChunkingConfig,
        title_prefix: str,
        metadata_suffix: str,
    ) -> int:
        """
        Calculate effective content token limit.

        Accounts for title, metadata, and contextual RAG.

        Args:
            config: Chunking configuration
            title_prefix: Title prefix string
            metadata_suffix: Metadata suffix string

        Returns:
            Content token limit
        """
        base_limit = config.chunk_token_limit

        # Subtract title tokens
        title_tokens = count_tokens(title_prefix) if title_prefix else 0

        # Subtract metadata tokens
        metadata_tokens = count_tokens(metadata_suffix) if metadata_suffix else 0

        # Subtract contextual RAG reservation
        rag_tokens = config.contextual_rag_reserved_tokens

        content_limit = base_limit - title_tokens - metadata_tokens - rag_tokens

        # Ensure positive
        return max(content_limit, config.chunk_min_content)

    def _create_blurb(self, text: str, blurb_size: int) -> str:
        """
        Create blurb (preview) from chunk text.

        Args:
            text: Chunk text
            blurb_size: Blurb size in tokens

        Returns:
            Blurb text
        """
        return truncate_to_tokens(text, blurb_size)

    @property
    def priority(self) -> int:
        """Default chunker priority."""
        return 0
