"""Contextual RAG: Document summary and chunk context generation.

This module provides LLM-based enhancement for chunks:
- doc_summary: A concise summary of the entire document
- chunk_context: Context that situates a chunk within the document

These enhancements improve retrieval quality by adding semantic context
that helps connect individual chunks to the broader document meaning.

Reference: Anthropic's Contextual Retrieval technique
https://www.anthropic.com/news/contextual-retrieval
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from agent_rag.core.models import Chunk
from agent_rag.ingestion.chunking.base import count_tokens, truncate_to_tokens
from agent_rag.ingestion.chunking.config import ChunkingConfig
from agent_rag.llm.interface import LLM, LLMMessage

logger = logging.getLogger(__name__)


# ============================================================================
# Prompts for Contextual RAG
# ============================================================================

DOC_SUMMARY_SYSTEM_PROMPT = """You are a document summarization assistant. Your task is to create a concise summary of the document that captures its main topics, purpose, and key information.

The summary should:
1. Be 2-4 sentences long
2. Capture the document's main purpose and topic
3. Include key entities, concepts, or themes
4. Be useful for understanding what the document is about

Do NOT include:
- Unnecessary preamble like "This document..."
- Opinions or interpretations
- Information not present in the document"""

DOC_SUMMARY_USER_PROMPT = """Please provide a concise summary of the following document:

<document>
{document_text}
</document>

Summary:"""

CHUNK_CONTEXT_SYSTEM_PROMPT = """You are a context generation assistant. Your task is to provide a brief context that situates a specific chunk within its source document.

The context should:
1. Be 1-2 sentences long
2. Explain what section or topic the chunk relates to
3. Connect the chunk to the broader document theme
4. Help someone understand where this chunk fits in the document

Do NOT include:
- The chunk content itself
- Unnecessary preamble
- Information not inferable from the document and chunk"""

CHUNK_CONTEXT_USER_PROMPT = """Here is the document summary:
<document_summary>
{doc_summary}
</document_summary>

Here is a chunk from this document:
<chunk>
{chunk_content}
</chunk>

Please provide a brief context that situates this chunk within the document:"""


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ContextualRAGResult:
    """Result of contextual RAG generation."""

    doc_summary: Optional[str] = None
    """Summary of the entire document."""

    chunk_contexts: Optional[dict[int, str]] = None
    """Map of chunk_id -> context string."""

    tokens_used: int = 0
    """Total tokens consumed by LLM calls."""

    errors: list[str] = None
    """Any errors encountered during generation."""

    def __post_init__(self):
        if self.chunk_contexts is None:
            self.chunk_contexts = {}
        if self.errors is None:
            self.errors = []


# ============================================================================
# Contextual RAG Generator
# ============================================================================


class ContextualRAGGenerator:
    """
    Generates contextual RAG enhancements for document chunks.

    Features:
    - Document summary generation
    - Per-chunk context generation
    - Token budget management
    - Batch processing for efficiency
    - Error handling with fallbacks

    Usage:
        generator = ContextualRAGGenerator(llm, config)
        result = await generator.generate(document_text, chunks)

        for chunk in chunks:
            chunk.doc_summary = result.doc_summary
            chunk.chunk_context = result.chunk_contexts.get(chunk.chunk_id)
    """

    def __init__(
        self,
        llm: LLM,
        config: ChunkingConfig,
        max_doc_tokens: int = 8000,
        max_chunk_tokens: int = 1000,
        max_concurrent_requests: int = 5,
    ):
        """
        Initialize the generator.

        Args:
            llm: LLM instance for generation
            config: Chunking configuration
            max_doc_tokens: Max tokens for document in summary prompt
            max_chunk_tokens: Max tokens for chunk in context prompt
            max_concurrent_requests: Max concurrent LLM requests
        """
        self.llm = llm
        self.config = config
        self.max_doc_tokens = max_doc_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.max_concurrent = max_concurrent_requests
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def generate(
        self,
        document_text: str,
        chunks: list[Chunk],
    ) -> ContextualRAGResult:
        """
        Generate contextual RAG enhancements for chunks.

        Args:
            document_text: Full document text
            chunks: List of chunks to enhance

        Returns:
            ContextualRAGResult with doc_summary and chunk_contexts
        """
        result = ContextualRAGResult()

        if not self.config.enable_contextual_rag:
            logger.debug("Contextual RAG disabled in config")
            return result

        # Step 1: Generate document summary (if enabled)
        if self.config.use_doc_summary:
            try:
                result.doc_summary = await self._generate_doc_summary(document_text)
                logger.debug(f"Generated doc summary: {len(result.doc_summary or '')} chars")
            except Exception as e:
                error_msg = f"Failed to generate doc summary: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)

        # Step 2: Generate chunk contexts (if enabled)
        if self.config.use_chunk_context and chunks:
            try:
                contexts = await self._generate_chunk_contexts(
                    document_summary=result.doc_summary or "",
                    chunks=chunks,
                )
                result.chunk_contexts = contexts
                logger.debug(f"Generated {len(contexts)} chunk contexts")
            except Exception as e:
                error_msg = f"Failed to generate chunk contexts: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)

        return result

    async def _generate_doc_summary(self, document_text: str) -> str:
        """
        Generate a summary of the document.

        Args:
            document_text: Full document text

        Returns:
            Document summary string
        """
        # Truncate document if too long
        truncated_doc = truncate_to_tokens(document_text, self.max_doc_tokens)

        messages = [
            LLMMessage(role="system", content=DOC_SUMMARY_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=DOC_SUMMARY_USER_PROMPT.format(document_text=truncated_doc),
            ),
        ]

        response = self.llm.chat(
            messages=messages,
            max_tokens=self.config.max_context_tokens,
            temperature=0.0,
        )

        return response.content.strip()

    async def _generate_chunk_contexts(
        self,
        document_summary: str,
        chunks: list[Chunk],
    ) -> dict[int, str]:
        """
        Generate context for each chunk.

        Args:
            document_summary: Summary of the document
            chunks: List of chunks

        Returns:
            Map of chunk_id -> context string
        """
        # Process chunks in batches with concurrency control
        tasks = []
        for chunk in chunks:
            task = self._generate_single_chunk_context(document_summary, chunk)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        contexts = {}
        for chunk, result in zip(chunks, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to generate context for chunk {chunk.chunk_id}: {result}")
                continue
            if result:
                contexts[chunk.chunk_id] = result

        return contexts

    async def _generate_single_chunk_context(
        self,
        document_summary: str,
        chunk: Chunk,
    ) -> Optional[str]:
        """
        Generate context for a single chunk.

        Args:
            document_summary: Summary of the document
            chunk: Chunk to generate context for

        Returns:
            Context string or None
        """
        async with self._semaphore:
            # Truncate chunk content if too long
            chunk_content = truncate_to_tokens(chunk.content, self.max_chunk_tokens)

            messages = [
                LLMMessage(role="system", content=CHUNK_CONTEXT_SYSTEM_PROMPT),
                LLMMessage(
                    role="user",
                    content=CHUNK_CONTEXT_USER_PROMPT.format(
                        doc_summary=document_summary,
                        chunk_content=chunk_content,
                    ),
                ),
            ]

            # Use synchronous call wrapped in executor for thread safety
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.chat(
                    messages=messages,
                    max_tokens=self.config.max_context_tokens // 2,  # Context is shorter
                    temperature=0.0,
                ),
            )

            return response.content.strip()


# ============================================================================
# Helper Functions
# ============================================================================


def apply_contextual_rag(
    chunks: list[Chunk],
    result: ContextualRAGResult,
) -> list[Chunk]:
    """
    Apply contextual RAG results to chunks.

    Modifies chunks in-place, adding doc_summary and chunk_context.

    Args:
        chunks: List of chunks to update
        result: ContextualRAGResult with generated content

    Returns:
        Updated chunks (same objects, modified in-place)
    """
    for chunk in chunks:
        # Apply document summary to all chunks
        if result.doc_summary:
            chunk.doc_summary = result.doc_summary

        # Apply chunk-specific context
        if result.chunk_contexts and chunk.chunk_id in result.chunk_contexts:
            chunk.chunk_context = result.chunk_contexts[chunk.chunk_id]

    return chunks


async def generate_contextual_rag_for_chunks(
    llm: LLM,
    document_text: str,
    chunks: list[Chunk],
    config: ChunkingConfig,
) -> list[Chunk]:
    """
    Convenience function to generate and apply contextual RAG.

    Args:
        llm: LLM instance
        document_text: Full document text
        chunks: Chunks to enhance
        config: Chunking configuration

    Returns:
        Enhanced chunks with doc_summary and chunk_context
    """
    if not config.enable_contextual_rag:
        return chunks

    generator = ContextualRAGGenerator(llm, config)
    result = await generator.generate(document_text, chunks)

    return apply_contextual_rag(chunks, result)


def should_use_contextual_rag(
    document_text: str,
    config: ChunkingConfig,
) -> bool:
    """
    Determine if contextual RAG should be used for this document.

    Based on design document rules:
    - Skip for single-chunk documents (summary IS the chunk)
    - Skip if disabled in config

    Args:
        document_text: Full document text
        config: Chunking configuration

    Returns:
        True if contextual RAG should be applied
    """
    if not config.enable_contextual_rag:
        return False

    if not config.use_doc_summary and not config.use_chunk_context:
        return False

    # Calculate if document fits in single chunk
    doc_tokens = count_tokens(document_text)
    effective_limit = config.effective_chunk_token_limit

    # If document fits in one chunk, contextual RAG adds no value
    if doc_tokens <= effective_limit:
        logger.debug("Document fits in single chunk, skipping contextual RAG")
        return False

    return True
