"""Content deduplication logic for ingestion."""

import hashlib
from typing import Optional
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_rag.core.env_config import ingestion_config
from agent_rag.ingestion.models import (
    DeduplicationResult,
    IngestionItem,
    IngestionStatus,
)


def compute_content_hash(content: bytes, source_type: str) -> str:
    """
    Compute SHA-256 hash of content for deduplication.

    Combines normalized content bytes with source_type to ensure
    different source types produce different hashes even for identical content.

    Args:
        content: Raw content bytes
        source_type: Source type (file, url, text, markdown)

    Returns:
        SHA-256 hash as hexadecimal string
    """
    hasher = hashlib.sha256()
    hasher.update(content)
    hasher.update(source_type.encode("utf-8"))
    return hasher.hexdigest()


async def check_duplicate(
    session: AsyncSession,
    tenant_id: str,
    content_hash: str,
    document_id: Optional[str] = None,
    force_reindex: bool = False,
) -> DeduplicationResult:
    """
    Check if content is duplicate and determine action.

    Deduplication Logic:
    1. If document_id is provided:
       - Check if document_id exists with different hash → update
       - Check if document_id exists with same hash → skip or reindex
    2. If no document_id provided:
       - Check (tenant_id, content_hash) for existing items
       - If found with status=INDEXED → skip (unless force_reindex)
       - If found with status=PROCESSING → skip (duplicate workflow)
       - If found with status=FAILED → reprocess if config allows

    Args:
        session: Database session
        tenant_id: Tenant ID
        content_hash: Content SHA-256 hash
        document_id: Optional explicit document ID
        force_reindex: Force reindexing even if already indexed

    Returns:
        DeduplicationResult with action recommendation
    """
    # Case 1: Explicit document_id provided
    if document_id:
        # Check if document_id already exists
        # Use .first() instead of .scalar_one_or_none() to handle potential duplicates
        stmt = select(IngestionItem).where(
            and_(
                IngestionItem.tenant_id == tenant_id,
                IngestionItem.document_id == document_id,
            )
        ).order_by(IngestionItem.created_at.desc()).limit(1)
        result = await session.execute(stmt)
        existing_item = result.scalar_one_or_none()

        if existing_item:
            # Same document_id exists
            if existing_item.content_hash == content_hash:
                # Same content, same document_id
                if force_reindex:
                    return DeduplicationResult(
                        is_duplicate=True,
                        existing_item_id=existing_item.id,
                        existing_document_id=document_id,
                        action="reindex",
                        message=(
                            f"Force reindexing existing document_id {document_id} "
                            f"with same content_hash"
                        ),
                    )
                else:
                    return DeduplicationResult(
                        is_duplicate=True,
                        existing_item_id=existing_item.id,
                        existing_document_id=document_id,
                        action="skip_duplicate",
                        message=(
                            f"Document {document_id} already exists with same content. "
                            f"Use force_reindex=true to reindex."
                        ),
                    )
            else:
                # Different content, same document_id → update scenario
                return DeduplicationResult(
                    is_duplicate=False,
                    existing_item_id=existing_item.id,
                    existing_document_id=document_id,
                    action="update",
                    message=(
                        f"Document {document_id} exists with different content. "
                        f"Will delete old chunks and reindex."
                    ),
                )
        else:
            # document_id doesn't exist yet, create new
            return DeduplicationResult(
                is_duplicate=False,
                existing_item_id=None,
                existing_document_id=None,
                action="create_new",
                message=f"Creating new document with explicit document_id {document_id}",
            )

    # Case 2: No explicit document_id, use hash-based dedup
    # Check tenant-scoped deduplication
    # Use .limit(1) to handle potential duplicates
    stmt = select(IngestionItem).where(
        and_(
            IngestionItem.tenant_id == tenant_id,
            IngestionItem.content_hash == content_hash,
        )
    ).order_by(IngestionItem.created_at.desc()).limit(1)
    result = await session.execute(stmt)
    existing_item = result.scalar_one_or_none()

    if not existing_item:
        # No duplicate found, create new
        return DeduplicationResult(
            is_duplicate=False,
            existing_item_id=None,
            existing_document_id=None,
            action="create_new",
            message="No duplicate found, creating new ingestion item",
        )

    # Duplicate found, check status
    if existing_item.status == IngestionStatus.INDEXED:
        # Already indexed
        if force_reindex:
            return DeduplicationResult(
                is_duplicate=True,
                existing_item_id=existing_item.id,
                existing_document_id=existing_item.document_id,
                action="reindex",
                message=(
                    f"Force reindexing document {existing_item.document_id} "
                    f"with content_hash {content_hash}"
                ),
            )
        else:
            return DeduplicationResult(
                is_duplicate=True,
                existing_item_id=existing_item.id,
                existing_document_id=existing_item.document_id,
                action="skip_duplicate",
                message=(
                    f"Content already indexed as document_id {existing_item.document_id}. "
                    f"Use force_reindex=true to reindex."
                ),
            )

    elif existing_item.status == IngestionStatus.PROCESSING:
        # Already being processed
        return DeduplicationResult(
            is_duplicate=True,
            existing_item_id=existing_item.id,
            existing_document_id=existing_item.document_id,
            action="skip_duplicate",
            message=(
                f"Content is already being processed (item_id {existing_item.id}). "
                f"Wait for completion."
            ),
        )

    elif existing_item.status == IngestionStatus.FAILED:
        # Previous attempt failed
        if ingestion_config.dedup_reprocess_failed:
            return DeduplicationResult(
                is_duplicate=True,
                existing_item_id=existing_item.id,
                existing_document_id=existing_item.document_id,
                action="reprocess",
                message=(
                    f"Previous ingestion failed (item_id {existing_item.id}). "
                    f"Creating new attempt per AGENT_RAG_DEDUP_REPROCESS_FAILED=true"
                ),
            )
        else:
            return DeduplicationResult(
                is_duplicate=True,
                existing_item_id=existing_item.id,
                existing_document_id=existing_item.document_id,
                action="skip_duplicate",
                message=(
                    f"Content previously failed (item_id {existing_item.id}). "
                    f"Set AGENT_RAG_DEDUP_REPROCESS_FAILED=true to retry."
                ),
            )

    elif existing_item.status == IngestionStatus.DUPLICATE:
        # Marked as duplicate before
        return DeduplicationResult(
            is_duplicate=True,
            existing_item_id=existing_item.id,
            existing_document_id=existing_item.document_id,
            action="skip_duplicate",
            message=f"Content is a known duplicate (item_id {existing_item.id})",
        )

    else:  # PENDING or unknown status
        # Pending but not yet processed, treat as new
        return DeduplicationResult(
            is_duplicate=True,
            existing_item_id=existing_item.id,
            existing_document_id=existing_item.document_id,
            action="reprocess",
            message=(
                f"Found pending item {existing_item.id}, will use existing record"
            ),
        )


async def check_cross_tenant_duplicate(
    session: AsyncSession,
    content_hash: str,
) -> Optional[IngestionItem]:
    """
    Check for cross-tenant duplicate (optional feature).

    Only used when AGENT_RAG_DEDUP_CROSS_TENANT=true.
    Even with global dedup, index isolation by tenant must be maintained.

    Args:
        session: Database session
        content_hash: Content SHA-256 hash

    Returns:
        First matching IngestionItem across all tenants, or None
    """
    if not ingestion_config.dedup_cross_tenant:
        return None

    stmt = (
        select(IngestionItem)
        .where(IngestionItem.content_hash == content_hash)
        .where(IngestionItem.status == IngestionStatus.INDEXED)
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def update_dedup_status(
    session: AsyncSession,
    item_id: UUID,
    action: str,
    message: str = "",
) -> None:
    """
    Update the dedup status of an ingestion item.

    Args:
        session: Database session
        item_id: Ingestion item UUID to update
        action: Action taken (skip, reindex, update, create_new, etc.)
        message: Optional status message
    """
    stmt = select(IngestionItem).where(IngestionItem.id == item_id)
    result = await session.execute(stmt)
    item = result.scalar_one_or_none()

    if item:
        # Update metadata with dedup info
        if item.metadata is None:
            item.metadata = {}
        item.metadata["dedup_action"] = action
        item.metadata["dedup_message"] = message
        await session.commit()


async def mark_as_duplicate(
    session: AsyncSession,
    item_id: UUID,
    original_document_id: str,
) -> None:
    """
    Mark an ingestion item as duplicate.

    Updates status to DUPLICATE and sets document_id to reference
    the original indexed document.

    Args:
        session: Database session
        item_id: Ingestion item UUID to mark
        original_document_id: Document ID of original item
    """
    stmt = select(IngestionItem).where(IngestionItem.id == item_id)
    result = await session.execute(stmt)
    item = result.scalar_one_or_none()

    if item:
        item.status = IngestionStatus.DUPLICATE
        item.document_id = original_document_id
        await session.commit()


async def get_existing_document_chunks_count(
    session: AsyncSession,
    tenant_id: str,
    document_id: str,
) -> int:
    """
    Get count of existing chunks for a document.

    Used when updating existing documents to determine if chunks
    need to be deleted before reindexing.

    Args:
        session: Database session
        tenant_id: Tenant ID
        document_id: Document ID

    Returns:
        Number of existing chunks (from ingestion_items.chunk_count)
    """
    stmt = select(IngestionItem).where(
        and_(
            IngestionItem.tenant_id == tenant_id,
            IngestionItem.document_id == document_id,
            IngestionItem.status == IngestionStatus.INDEXED,
        )
    )
    result = await session.execute(stmt)
    item = result.scalar_one_or_none()

    if item and item.chunk_count:
        return item.chunk_count
    return 0


async def delete_old_chunks(
    tenant_id: str,
    existing_item_id: str,
    existing_document_id: Optional[str] = None,
) -> int:
    """
    Delete old chunks from vector index when reindexing.

    Called during force_reindex or update operations to ensure
    clean slate before reindexing.

    Args:
        tenant_id: Tenant ID
        existing_item_id: Existing ingestion item ID
        existing_document_id: Document ID in vector index (if known)

    Returns:
        Number of chunks deleted
    """
    from agent_rag.core.database import AsyncSessionLocal
    from agent_rag.document_index.vespa import VespaIndex

    deleted_count = 0

    # Get document_id from existing item if not provided
    if not existing_document_id:
        async with AsyncSessionLocal() as session:
            stmt = select(IngestionItem).where(IngestionItem.id == existing_item_id)
            result = await session.execute(stmt)
            item = result.scalar_one_or_none()
            if item:
                existing_document_id = item.document_id

    if existing_document_id:
        try:
            # Delete chunks from vector index
            doc_index = VespaIndex()
            existing_chunks = doc_index.get_chunks_by_document(existing_document_id)
            deleted_count = len(existing_chunks)
            doc_index.delete_document(existing_document_id)
        except Exception as e:
            # Log but don't fail - chunks may not exist yet
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to delete old chunks for {existing_document_id}: {e}")

    # Also clean up MinIO artifacts for the old item
    from agent_rag.ingestion.storage import get_minio_adapter
    try:
        storage = get_minio_adapter()
        storage.delete_item_artifacts(tenant_id, existing_item_id)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to delete MinIO artifacts for {existing_item_id}: {e}")

    return deleted_count


def normalize_content_for_hash(content: bytes) -> bytes:
    """
    Normalize content bytes before hashing.

    Applies consistent normalization to ensure semantically identical
    content produces the same hash regardless of minor formatting differences.

    Normalization steps:
    - Strip trailing whitespace from each line
    - Normalize line endings to \\n
    - Remove BOM if present

    Args:
        content: Raw content bytes

    Returns:
        Normalized content bytes
    """
    # Decode with UTF-8, handling potential errors
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        # For binary files, return as-is
        return content

    # Remove BOM if present
    if text.startswith("\ufeff"):
        text = text[1:]

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    normalized = "\n".join(lines)

    return normalized.encode("utf-8")
