"""Hatchet tasks for indexing phase (chunking, embedding, indexing)."""

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from agent_rag.core.database import AsyncSessionLocal
from agent_rag.core.models import Chunk
from agent_rag.document_index.interface import DocumentIndex
from agent_rag.document_index.memory.memory_index import MemoryIndex
from agent_rag.document_index.vespa.vespa_index import VespaIndex
from agent_rag.embedding.interface import Embedder
from agent_rag.embedding.providers.litellm_embedder import LiteLLMEmbedder as OpenAIEmbedder  # Compatibility alias
from agent_rag.ingestion.chunking.config import ChunkingConfig
from agent_rag.ingestion.chunking.registry import get_chunker_registry
from agent_rag.ingestion.embeddings.config import EmbeddingConfig
from agent_rag.ingestion.embeddings.embedder import DefaultIndexingEmbedder
from agent_rag.ingestion.embeddings.failure_handler import (
    embed_chunks_with_failure_handling,
)
from agent_rag.ingestion.models import IngestionItem, IngestionStatus
from agent_rag.ingestion.parsing.base import ParsedDocument
from agent_rag.ingestion.storage import get_minio_adapter
from agent_rag.ingestion.text_normalize import normalize_chunk_content

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Schemas for Task Inputs/Outputs
# ============================================================================


class ChunkDocumentInput(BaseModel):
    """Input for chunk_document_task."""

    item_id: str
    tenant_id: str
    source_type: str
    parsed_ref: str  # MinIO path to parsed text
    chunking_config: dict = {}


class ChunkDocumentOutput(BaseModel):
    """Output from chunk_document_task."""

    item_id: str
    chunk_count: int
    mini_chunk_count: int
    large_chunk_count: int
    image_chunk_count: int


class EmbedChunksInput(BaseModel):
    """Input for embed_chunks_task."""

    item_id: str
    tenant_id: str
    chunk_count: int
    embedding_config: dict = {}


class EmbedChunksOutput(BaseModel):
    """Output from embed_chunks_task."""

    item_id: str
    embedded_chunk_count: int
    failed_chunk_count: int
    success_rate: float


class IndexChunksInput(BaseModel):
    """Input for index_chunks_task."""

    item_id: str
    tenant_id: str
    embedded_chunk_count: int
    index_name: str = "default"


class IndexChunksOutput(BaseModel):
    """Output from index_chunks_task."""

    item_id: str
    indexed_chunk_count: int
    index_name: str
    success: bool


# ============================================================================
# Hatchet Task: chunk_document_task
# ============================================================================


async def chunk_document_task(input: ChunkDocumentInput) -> ChunkDocumentOutput:
    """
    Task 7: Chunk document into semantic pieces.

    Uses ChunkerRegistry to select appropriate chunker based on source_type.

    Features:
    - Semantic-aware chunking with chonkie SentenceChunker
    - Image section handling (dedicated chunks)
    - Multipass mode (mini-chunks)
    - Large chunk generation
    - Contextual RAG support

    Returns:
        ChunkDocumentOutput with chunking statistics
    """
    logger.info(
        f"Chunking document: item_id={input.item_id}, "
        f"source_type={input.source_type}"
    )

    # Load chunking config
    chunking_config = ChunkingConfig(**input.chunking_config)

    # Retrieve parsed text from MinIO
    storage = get_minio_adapter()
    parsed_text = await storage.retrieve_parsed_text(
        tenant_id=input.tenant_id,
        item_id=input.item_id,
    )

    # Merge OCR text (if any) into parsed text
    parsed_text = await _merge_ocr_text(
        storage=storage,
        tenant_id=input.tenant_id,
        item_id=input.item_id,
        parsed_text=parsed_text,
    )

    # Retrieve IngestionItem
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select

        stmt = select(IngestionItem).where(IngestionItem.id == input.item_id)
        result = await session.execute(stmt)
        item = result.scalar_one_or_none()

        if not item:
            raise ValueError(f"IngestionItem not found: {input.item_id}")

        # Create ParsedDocument
        parsed_doc = ParsedDocument(
            text=parsed_text,
            metadata=item.metadata_json or {},
            images=[],  # Images already stored separately
            links=[],
            tables=[],
        )

        # Get chunker from registry
        registry = get_chunker_registry()
        chunker = registry.get_chunker(
            source_type=input.source_type,
            mime_type=item.mime_type or "",
            document=parsed_doc,
        )

        # Chunk document
        chunks = chunker.chunk(
            document=parsed_doc,
            item=item,
            config=chunking_config,
        )

        # Count chunk types
        regular_chunks = [c for c in chunks if not c.large_chunk_reference_ids]
        image_chunks = [c for c in chunks if hasattr(c, "image_file_id")]
        large_chunks = [c for c in chunks if c.large_chunk_reference_ids]

        # Count mini-chunks
        mini_chunk_count = sum(
            len(c._mini_chunk_texts or []) for c in chunks if hasattr(c, "_mini_chunk_texts")
        )

        # Store chunks to database (temporary in-memory for this workflow)
        # In production, chunks would be stored in a chunks table
        # For now, we'll store them as JSON in IngestionItem.metadata_

        import json

        chunks_data = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "blurb": c.blurb,
                "semantic_identifier": c.semantic_identifier,
                "metadata_suffix": c.metadata_suffix,
                "section_continuation": c.section_continuation,
                "large_chunk_reference_ids": c.large_chunk_reference_ids,
            }
            for c in chunks
        ]

        # Update IngestionItem
        item.chunk_count = len(chunks)

        # Ensure metadata_json is mutable and track changes
        if item.metadata_json is None:
            item.metadata_json = {}
        item.metadata_json["chunks"] = chunks_data  # Store for retrieval in embed task

        # Mark JSON field as modified so SQLAlchemy knows to save it
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(item, "metadata_json")

        await session.commit()

        logger.info(
            f"Document chunked: {len(chunks)} total chunks "
            f"({len(regular_chunks)} regular, {len(large_chunks)} large, "
            f"{len(image_chunks)} image, {mini_chunk_count} mini-chunks)"
        )

        return ChunkDocumentOutput(
            item_id=input.item_id,
            chunk_count=len(chunks),
            mini_chunk_count=mini_chunk_count,
            large_chunk_count=len(large_chunks),
            image_chunk_count=len(image_chunks),
        )


async def _merge_ocr_text(
    storage,
    tenant_id: str,
    item_id: str,
    parsed_text: str,
) -> str:
    """Append OCR text (if present) to parsed text."""
    ocr_prefix = f"ocr/{tenant_id}/{item_id}/"
    try:
        ocr_objects = await storage.list_objects(prefix=ocr_prefix)
    except Exception:
        return parsed_text

    if not ocr_objects:
        return parsed_text

    ocr_texts: list[str] = []
    for obj in ocr_objects:
        try:
            ocr_data = storage.get_ocr_result(obj.object_name)
            text = ocr_data.get("text", "")
            if text:
                ocr_texts.append(text)
        except Exception:
            continue

    if not ocr_texts:
        return parsed_text

    merged = parsed_text.rstrip()
    merged += "\n\n[OCR]\n\n" + "\n\n".join(ocr_texts)
    return merged


# ============================================================================
# Hatchet Task: embed_chunks_task
# ============================================================================


async def embed_chunks_task(input: EmbedChunksInput) -> EmbedChunksOutput:
    """
    Task 8: Embed chunks with title caching and failure handling.

    Uses DefaultIndexingEmbedder with:
    - Title embedding caching (avoid redundant computations)
    - Batch embedding optimization
    - Mini-chunk embedding (multipass indexing)
    - Per-document failure isolation

    Returns:
        EmbedChunksOutput with embedding statistics
    """
    logger.info(
        f"Embedding chunks: item_id={input.item_id}, "
        f"chunk_count={input.chunk_count}"
    )

    # Load embedding config
    embedding_config = EmbeddingConfig(**input.embedding_config)

    # Retrieve chunks from IngestionItem metadata
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select

        stmt = select(IngestionItem).where(IngestionItem.id == input.item_id)
        result = await session.execute(stmt)
        item = result.scalar_one_or_none()

        if not item:
            raise ValueError(f"IngestionItem not found: {input.item_id}")

        # Reconstruct chunks from metadata
        chunks_data = item.metadata_json.get("chunks", [])

        if not chunks_data:
            raise ValueError(f"No chunks found in metadata for item {input.item_id}")

        chunks = [
            Chunk(
                document_id=item.document_id or str(item.id),
                chunk_id=c["chunk_id"],
                content=normalize_chunk_content(c["content"]),
                title=item.metadata_json.get("title"),
                source_type=str(item.source_type.value),
                link=item.source_uri,
                metadata=item.metadata_json or {},
                semantic_identifier=c.get("semantic_identifier", ""),
                metadata_suffix=c.get("metadata_suffix", ""),
                blurb=c.get("blurb", ""),
                section_continuation=c.get("section_continuation", False),
                large_chunk_reference_ids=c.get("large_chunk_reference_ids", []),
            )
            for c in chunks_data
        ]

        # Initialize embedder with config from environment
        from agent_rag.core.env_config import get_embedding_config_from_env
        emb_config = get_embedding_config_from_env()
        base_embedder = OpenAIEmbedder(emb_config)
        indexing_embedder = DefaultIndexingEmbedder(embedder=base_embedder)

        # Embed chunks with failure handling
        batch_result = await embed_chunks_with_failure_handling(
            chunks=chunks,
            embedder=indexing_embedder,
            config=embedding_config,
        )

        # Store indexed chunks to metadata for index_chunks_task
        indexed_chunks_data = [
            {
                "chunk_id": ic.chunk_id,
                "document_id": ic.document_id,
                "full_embedding": ic.full_embedding,
                "mini_chunk_embeddings": ic.mini_chunk_embeddings,
                "title_embedding": ic.title_embedding,
            }
            for ic in batch_result.indexed_chunks
        ]

        logger.info(f"[DEBUG] Storing {len(indexed_chunks_data)} indexed_chunks to metadata_json")
        item.metadata_json["indexed_chunks"] = indexed_chunks_data
        logger.info(f"[DEBUG] metadata_json['indexed_chunks'] set: {len(item.metadata_json.get('indexed_chunks', []))} items")

        # Log failures if any
        if batch_result.has_failures:
            failure_summary = batch_result.to_summary()
            logger.warning(
                f"Embedding failures: {failure_summary['failed_documents']} documents, "
                f"{failure_summary['failed_chunks']} chunks"
            )
            item.metadata_json["embedding_failures"] = failure_summary

        # Mark JSON field as modified so SQLAlchemy knows to save it
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(item, "metadata_json")

        logger.info(f"[DEBUG] Before commit, metadata_json has indexed_chunks: {len(item.metadata_json.get('indexed_chunks', []))} items")

        await session.commit()

        # Verify the data was saved
        await session.refresh(item)
        saved_count = len(item.metadata_json.get("indexed_chunks", []))
        logger.info(f"[DEBUG] After commit, metadata_json['indexed_chunks'] has {saved_count} items")

        logger.info(
            f"Chunks embedded: {batch_result.successful_chunks}/{batch_result.total_chunks} "
            f"({batch_result.success_rate:.1%} success rate)"
        )

        return EmbedChunksOutput(
            item_id=input.item_id,
            embedded_chunk_count=batch_result.successful_chunks,
            failed_chunk_count=batch_result.total_chunks
            - batch_result.successful_chunks,
            success_rate=batch_result.success_rate,
        )


# ============================================================================
# Hatchet Task: index_chunks_task
# ============================================================================


async def index_chunks_task(input: IndexChunksInput) -> IndexChunksOutput:
    """
    Task 9: Index embedded chunks to vector database.

    Supports multiple index backends:
    - VespaIndex (default, production-ready)
    - MemoryIndex (testing, development)

    Returns:
        IndexChunksOutput with indexing status
    """
    logger.info(
        f"Indexing chunks: item_id={input.item_id}, "
        f"chunk_count={input.embedded_chunk_count}, index={input.index_name}"
    )

    # Retrieve indexed chunks from IngestionItem metadata
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select

        stmt = select(IngestionItem).where(IngestionItem.id == input.item_id)
        result = await session.execute(stmt)
        item = result.scalar_one_or_none()

        if not item:
            raise ValueError(f"IngestionItem not found: {input.item_id}")

        indexed_chunks_data = item.metadata_json.get("indexed_chunks", [])

        logger.info(f"[DEBUG] Retrieved indexed_chunks: {len(indexed_chunks_data)} items from metadata_json")

        if not indexed_chunks_data:
            logger.warning(
                f"No indexed chunks found in metadata for item {input.item_id}"
            )
            return IndexChunksOutput(
                item_id=input.item_id,
                indexed_chunk_count=0,
                index_name=input.index_name,
                success=False,
            )

        try:
            # Get document index based on configuration
            import os
            index_type = os.getenv("AGENT_RAG_INDEX_TYPE", "vespa").lower()
            doc_index: DocumentIndex

            if index_type == "vespa":
                logger.info("Using VespaIndex for indexing")
                doc_index = VespaIndex()
            else:
                logger.info(f"Using MemoryIndex for indexing (type={index_type})")
                persist_path = os.getenv("AGENT_RAG_INDEX_PERSIST_PATH", "./data/memory_index.json")
                doc_index = MemoryIndex(persist_path=persist_path)

            # Reconstruct Chunk objects with embeddings from indexed_chunks_data
            # Note: We need to merge chunk metadata from earlier stages with embeddings
            chunks_metadata = item.metadata_json.get("chunks", [])

            # Create a mapping from chunk_id to chunk metadata for fast lookup
            chunks_map = {c["chunk_id"]: c for c in chunks_metadata}

            # Build full Chunk objects with embeddings
            chunks_to_index: list[Chunk] = []
            for idx_data in indexed_chunks_data:
                chunk_id = idx_data["chunk_id"]
                document_id = idx_data["document_id"]
                full_embedding = idx_data["full_embedding"]

                # Log embedding dimension
                emb_dim = len(full_embedding) if isinstance(full_embedding, list) else 0
                logger.info(f"[DEBUG] Building chunk {document_id}_{chunk_id}: embedding dimension={emb_dim}, expected=2560")

                # Get corresponding chunk metadata
                chunk_meta = chunks_map.get(chunk_id, {})

                # Filter metadata to exclude workflow-internal fields (chunks, indexed_chunks contain large data)
                # Only keep small, essential metadata fields
                filtered_chunk_metadata = {}
                if item.metadata_json:
                    # Fields to exclude (workflow-internal, contain large data)
                    exclude_keys = {"chunks", "indexed_chunks", "embedding_failures"}
                    for key, value in item.metadata_json.items():
                        if key not in exclude_keys:
                            # Only include small values
                            if isinstance(value, (str, int, float, bool, type(None))):
                                if isinstance(value, str) and len(value) > 500:
                                    continue  # Skip large strings
                                filtered_chunk_metadata[key] = value
                            elif isinstance(value, list) and len(value) <= 10:
                                # Only include small lists
                                filtered_chunk_metadata[key] = value

                # Create Chunk with all fields
                chunk = Chunk(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    content=normalize_chunk_content(chunk_meta.get("content", "")),
                    embedding=full_embedding,

                    # Metadata fields
                    title=item.metadata_json.get("title"),
                    source_type=str(item.source_type.value),
                    link=item.source_uri,
                    metadata=filtered_chunk_metadata,

                    # Enhanced fields from chunking
                    semantic_identifier=chunk_meta.get("semantic_identifier", ""),
                    metadata_suffix=chunk_meta.get("metadata_suffix", ""),
                    blurb=chunk_meta.get("blurb", ""),
                    section_continuation=chunk_meta.get("section_continuation", False),

                    # Large chunk support
                    large_chunk_reference_ids=chunk_meta.get("large_chunk_reference_ids", []),

                    # Multi-embedding support
                    title_embedding=idx_data.get("title_embedding"),
                    embeddings={
                        "full": idx_data["full_embedding"],
                        **{f"mini_{i}": emb for i, emb in enumerate(idx_data.get("mini_chunk_embeddings", []))}
                    } if idx_data.get("mini_chunk_embeddings") else None,

                    # Tenant isolation
                    tenant_id=input.tenant_id,
                )

                chunks_to_index.append(chunk)

            # Actually index chunks to DocumentIndex
            indexed_ids = doc_index.index_chunks(chunks_to_index)

            logger.info(
                f"Successfully indexed {len(indexed_ids)} chunks to {input.index_name}"
            )

            # Update IngestionItem status
            item.status = IngestionStatus.INDEXED
            item.completed_at = datetime.utcnow()
            await session.commit()

            return IndexChunksOutput(
                item_id=input.item_id,
                indexed_chunk_count=len(indexed_ids),
                index_name=input.index_name,
                success=True,
            )

        except Exception as e:
            logger.error(f"Failed to index chunks for {input.item_id}: {e}")

            # Attempt rollback of any partially indexed chunks
            rollback_ok = False
            try:
                if item.document_id:
                    rollback_ok = doc_index.delete_document(item.document_id)
            except Exception as rollback_error:
                logger.warning(
                    f"Rollback failed for {input.item_id}: {rollback_error}"
                )

            # Update status to FAILED or FAILED_PARTIAL
            if rollback_ok:
                item.status = IngestionStatus.FAILED
            else:
                item.status = IngestionStatus.FAILED_PARTIAL
            item.error = str(e)
            item.completed_at = datetime.utcnow()
            await session.commit()

            return IndexChunksOutput(
                item_id=input.item_id,
                indexed_chunk_count=0,
                index_name=input.index_name,
                success=False,
            )
