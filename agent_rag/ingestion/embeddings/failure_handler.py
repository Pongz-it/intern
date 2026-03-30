"""Embedding failure handling with per-document isolation."""

import asyncio
import logging
from typing import Optional

from agent_rag.core.models import Chunk
from agent_rag.ingestion.embeddings.config import EmbeddingConfig
from agent_rag.ingestion.embeddings.embedder import IndexingEmbedder
from agent_rag.ingestion.embeddings.models import (
    EmbeddingBatchResult,
    FailedDocument,
    IndexChunk,
)

logger = logging.getLogger(__name__)


async def embed_chunks_with_failure_handling(
    chunks: list[Chunk],
    embedder: IndexingEmbedder,
    config: EmbeddingConfig,
) -> EmbeddingBatchResult:
    """
    Embed chunks with per-document failure isolation [P1].

    Strategy:
    1. Group chunks by document_id
    2. Attempt batch embedding for all chunks
    3. If batch fails and fail_on_batch_error=False:
       - Retry each document in isolation
       - Collect successful embeddings
       - Track failed documents with error details
    4. Return EmbeddingBatchResult with successes and failures

    Args:
        chunks: List of chunks to embed
        embedder: Indexing embedder instance
        config: Embedding configuration

    Returns:
        EmbeddingBatchResult with indexed chunks and failures
    """
    total_chunks = len(chunks)
    logger.info(
        f"Starting batch embedding for {total_chunks} chunks "
        f"(fail_on_batch_error={config.fail_on_batch_error})"
    )

    # Try batch embedding first
    try:
        indexed_chunks = await embedder.embed_chunks(chunks, config)

        logger.info(
            f"Batch embedding succeeded: {len(indexed_chunks)} chunks embedded"
        )

        return EmbeddingBatchResult(
            indexed_chunks=indexed_chunks,
            failed_documents=[],
            total_chunks=total_chunks,
            successful_chunks=len(indexed_chunks),
        )

    except Exception as batch_error:
        logger.warning(f"Batch embedding failed: {batch_error}")

        # If fail_on_batch_error=True, propagate error
        if config.fail_on_batch_error:
            logger.error(
                "fail_on_batch_error=True, propagating batch embedding error"
            )
            raise

        # Otherwise, isolate failures per document
        logger.info(
            "fail_on_batch_error=False, isolating failures per document "
            f"(retry_failed_documents={config.retry_failed_documents})"
        )

        return await _embed_with_document_isolation(chunks, embedder, config)


async def _embed_with_document_isolation(
    chunks: list[Chunk],
    embedder: IndexingEmbedder,
    config: EmbeddingConfig,
) -> EmbeddingBatchResult:
    """
    Embed chunks with per-document failure isolation.

    Args:
        chunks: List of chunks to embed
        embedder: Indexing embedder instance
        config: Embedding configuration

    Returns:
        EmbeddingBatchResult with successes and failures
    """
    # Group chunks by document_id
    chunks_by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        doc_id = chunk.document_id
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = []
        chunks_by_doc[doc_id].append(chunk)

    logger.info(
        f"Attempting per-document embedding for {len(chunks_by_doc)} documents"
    )

    # Embed each document in isolation
    indexed_chunks: list[IndexChunk] = []
    failed_documents: list[FailedDocument] = []

    for doc_id, doc_chunks in chunks_by_doc.items():
        try:
            # Attempt embedding for this document
            doc_indexed = await _embed_document_with_retry(
                doc_id, doc_chunks, embedder, config
            )

            indexed_chunks.extend(doc_indexed)
            logger.debug(
                f"Document {doc_id}: {len(doc_indexed)} chunks embedded successfully"
            )

        except Exception as doc_error:
            # Document failed after retries
            error_type = type(doc_error).__name__

            failed_doc = FailedDocument(
                document_id=doc_id,
                chunks=doc_chunks,
                error=str(doc_error),
                error_type=error_type,
                retry_count=config.max_document_retries if config.retry_failed_documents else 0,
                recoverable=_is_recoverable_error(doc_error),
            )

            failed_documents.append(failed_doc)

            logger.error(
                f"Document {doc_id} failed after retries: {error_type}: {doc_error}"
            )

    # Create result
    successful_chunks = len(indexed_chunks)
    total_chunks = len(chunks)

    result = EmbeddingBatchResult(
        indexed_chunks=indexed_chunks,
        failed_documents=failed_documents,
        total_chunks=total_chunks,
        successful_chunks=successful_chunks,
    )

    logger.info(
        f"Per-document embedding complete: "
        f"{successful_chunks}/{total_chunks} chunks succeeded "
        f"({result.success_rate:.1%}), "
        f"{len(failed_documents)} documents failed"
    )

    return result


async def _embed_document_with_retry(
    doc_id: str,
    chunks: list[Chunk],
    embedder: IndexingEmbedder,
    config: EmbeddingConfig,
    retry_count: int = 0,
) -> list[IndexChunk]:
    """
    Embed document chunks with retry logic.

    Args:
        doc_id: Document ID
        chunks: Chunks to embed
        embedder: Indexing embedder instance
        config: Embedding configuration
        retry_count: Current retry attempt

    Returns:
        List of indexed chunks

    Raises:
        Exception: If embedding fails after max retries
    """
    try:
        indexed_chunks = await embedder.embed_chunks(chunks, config)
        return indexed_chunks

    except Exception as error:
        # Check if we should retry
        if not config.retry_failed_documents:
            logger.debug(
                f"Document {doc_id} failed (retry disabled): {type(error).__name__}"
            )
            raise

        if retry_count >= config.max_document_retries:
            logger.error(
                f"Document {doc_id} failed after {retry_count} retries: {error}"
            )
            raise

        # Check if error is recoverable
        if not _is_recoverable_error(error):
            logger.warning(
                f"Document {doc_id} failed with non-recoverable error: {type(error).__name__}"
            )
            raise

        # Retry with exponential backoff
        retry_count += 1
        delay = config.retry_delay * (2 ** (retry_count - 1))

        logger.warning(
            f"Document {doc_id} failed (attempt {retry_count}/{config.max_document_retries}), "
            f"retrying in {delay:.1f}s: {type(error).__name__}"
        )

        await asyncio.sleep(delay)

        return await _embed_document_with_retry(
            doc_id, chunks, embedder, config, retry_count
        )


def _is_recoverable_error(error: Exception) -> bool:
    """
    Check if error is recoverable (should retry).

    Args:
        error: Exception that occurred

    Returns:
        True if error is recoverable, False otherwise
    """
    error_type = type(error).__name__

    # Recoverable errors (typically transient)
    recoverable_errors = {
        "TimeoutError",
        "ConnectionError",
        "HTTPError",
        "APIError",
        "RateLimitError",
        "ServiceUnavailableError",
        "InternalServerError",
    }

    # Non-recoverable errors (typically permanent)
    non_recoverable_errors = {
        "ValidationError",
        "ValueError",
        "TypeError",
        "AuthenticationError",
        "PermissionError",
        "InvalidRequestError",
    }

    # Check exact type name
    if error_type in recoverable_errors:
        return True

    if error_type in non_recoverable_errors:
        return False

    # Default: recoverable if not explicitly non-recoverable
    # Conservative approach for unknown errors
    logger.debug(f"Unknown error type {error_type}, treating as recoverable")
    return True


async def retry_failed_documents(
    failed_documents: list[FailedDocument],
    embedder: IndexingEmbedder,
    config: EmbeddingConfig,
) -> EmbeddingBatchResult:
    """
    Retry embedding for previously failed documents.

    Args:
        failed_documents: List of failed documents to retry
        embedder: Indexing embedder instance
        config: Embedding configuration

    Returns:
        EmbeddingBatchResult with retry results
    """
    logger.info(f"Retrying {len(failed_documents)} failed documents")

    # Filter recoverable failures
    recoverable_docs = [fd for fd in failed_documents if fd.recoverable]

    if not recoverable_docs:
        logger.warning("No recoverable documents to retry")
        return EmbeddingBatchResult(
            indexed_chunks=[],
            failed_documents=failed_documents,
            total_chunks=sum(len(fd.chunks) for fd in failed_documents),
            successful_chunks=0,
        )

    logger.info(
        f"Retrying {len(recoverable_docs)} recoverable documents "
        f"(skipping {len(failed_documents) - len(recoverable_docs)} non-recoverable)"
    )

    # Collect all chunks to retry
    chunks_to_retry = []
    for failed_doc in recoverable_docs:
        chunks_to_retry.extend(failed_doc.chunks)

    # Retry with document isolation
    return await _embed_with_document_isolation(chunks_to_retry, embedder, config)
