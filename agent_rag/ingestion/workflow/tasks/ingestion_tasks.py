"""Hatchet tasks for ingestion phase (fetch, dedup, store, parse)."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from agent_rag.core.database import AsyncSessionLocal
from agent_rag.core.env_config import ingestion_config
from agent_rag.ingestion.dedup import (
    check_duplicate,
    compute_content_hash,
    normalize_content_for_hash,
    update_dedup_status,
)
from agent_rag.ingestion.models import IngestionItem, IngestionStatus, SourceType
from agent_rag.ingestion.parsing.registry import get_parser_registry
from agent_rag.ingestion.chunking.base import count_tokens
from agent_rag.ingestion.parsing.utils import truncate_text
from agent_rag.ingestion.storage import get_minio_adapter

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Schemas for Task Inputs/Outputs
# ============================================================================


class FetchContentInput(BaseModel):
    """Input for fetch_content_task."""

    tenant_id: str
    source_type: str
    source_uri: str
    filename: str
    document_id: Optional[str] = None
    metadata: dict = {}
    force_reindex: bool = False


class FetchContentOutput(BaseModel):
    """Output from fetch_content_task."""

    item_id: str
    tenant_id: str
    source_type: str
    content_hash: str
    raw_content_path: str
    filename: str


class DedupCheckInput(BaseModel):
    """Input for dedup_check_task."""

    tenant_id: str
    content_hash: str
    document_id: Optional[str] = None
    force_reindex: bool = False


class DedupCheckOutput(BaseModel):
    """Output from dedup_check_task."""

    is_duplicate: bool
    action: str  # "skip", "reindex", "process"
    existing_item_id: Optional[str] = None
    message: str


class StoreContentInput(BaseModel):
    """Input for store_content_task."""

    item_id: str
    tenant_id: str
    filename: str
    content: bytes
    content_type: str


class StoreContentOutput(BaseModel):
    """Output from store_content_task."""

    item_id: str
    raw_content_path: str


class ParseDocumentInput(BaseModel):
    """Input for parse_document_task."""

    item_id: str
    tenant_id: str
    source_type: str
    filename: str
    content: bytes
    mime_type: Optional[str] = None


class ParseDocumentOutput(BaseModel):
    """Output from parse_document_task."""

    item_id: str
    parsed_ref: str  # MinIO path to parsed text
    text_length: int
    image_count: int
    table_count: int
    link_count: int


# ============================================================================
# Hatchet Task: fetch_content_task
# ============================================================================


async def fetch_content_task(input: FetchContentInput) -> FetchContentOutput:
    """
    Task 1: Fetch content from source_uri.

    For file: Read local file
    For url: Fetch HTTP content
    For text/markdown: Use content directly

    Returns:
        FetchContentOutput with item_id and content_hash
    """
    logger.info(
        f"Fetching content: tenant={input.tenant_id}, "
        f"source={input.source_type}, uri={input.source_uri}"
    )

    # Fetch content based on source type
    if input.source_type == "file":
        content = await _fetch_file_content(input.source_uri)
    elif input.source_type == "url":
        content = await _fetch_url_content(input.source_uri)
    elif input.source_type in ["text", "markdown"]:
        content = input.source_uri.encode("utf-8")  # source_uri is content itself
    else:
        raise ValueError(f"Unsupported source_type: {input.source_type}")

    # Calculate content hash using normalized content + source_type
    normalized_content = normalize_content_for_hash(content)
    content_hash = compute_content_hash(normalized_content, input.source_type)

    # Create IngestionItem
    async with AsyncSessionLocal() as session:
        item = IngestionItem(
            tenant_id=input.tenant_id,
            source_type=SourceType(input.source_type),
            source_uri=input.source_uri,
            content_hash=content_hash,
            file_name=input.filename,
            document_id=input.document_id,
            metadata_json=input.metadata,
            status=IngestionStatus.PROCESSING,
            size_bytes=len(content),
        )

        session.add(item)
        await session.commit()
        await session.refresh(item)

        item_id = str(item.id)

    # Store raw content (sync method, run in thread)
    storage = get_minio_adapter()
    raw_content_path = await asyncio.to_thread(
        storage.store_raw_content,
        tenant_id=input.tenant_id,
        item_id=item_id,
        filename=input.filename,
        content=content,
        content_type=_guess_content_type(input.filename),
    )

    # Update content_ref in DB
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select

        stmt = select(IngestionItem).where(IngestionItem.id == item_id)
        result = await session.execute(stmt)
        item = result.scalar_one_or_none()
        if item:
            item.content_ref = raw_content_path
            await session.commit()

    logger.info(
        f"Content fetched: item_id={item_id}, hash={content_hash[:8]}, "
        f"size={len(content)} bytes, path={raw_content_path}"
    )

    return FetchContentOutput(
        item_id=item_id,
        tenant_id=input.tenant_id,
        source_type=input.source_type,
        content_hash=content_hash,
        raw_content_path=raw_content_path,
        filename=input.filename,
    )


# ============================================================================
# Hatchet Task: dedup_check_task
# ============================================================================


async def dedup_check_task(input: DedupCheckInput) -> DedupCheckOutput:
    """
    Task 2: Check for duplicate content.

    Uses dedup.py logic to determine if content should be:
    - Skipped (duplicate already indexed)
    - Reindexed (duplicate but forced)
    - Processed (new content)

    Returns:
        DedupCheckOutput with action decision
    """
    logger.info(
        f"Checking duplicate: tenant={input.tenant_id}, "
        f"hash={input.content_hash[:8]}, document_id={input.document_id}"
    )

    async with AsyncSessionLocal() as session:
        dedup_result = await check_duplicate(
            session=session,
            tenant_id=input.tenant_id,
            content_hash=input.content_hash,
            document_id=input.document_id,
            force_reindex=input.force_reindex,
        )

        # Determine action based on result
        if dedup_result.action == "skip_duplicate":
            action = "skip"
        elif dedup_result.action in ("reindex", "update"):
            action = "reindex"
        elif dedup_result.action in ("reprocess", "create_new"):
            action = "process"
        else:
            action = "process"

        message = dedup_result.message

        logger.info(
            f"Dedup check result: action={action}, "
            f"existing_id={dedup_result.existing_item_id}, message={message}"
        )

        return DedupCheckOutput(
            is_duplicate=dedup_result.is_duplicate,
            action=action,
            existing_item_id=str(dedup_result.existing_item_id) if dedup_result.existing_item_id else None,
            message=message,
        )


# ============================================================================
# Hatchet Task: store_content_task
# ============================================================================


async def store_content_task(input: StoreContentInput) -> StoreContentOutput:
    """
    Task 3: Store raw content to MinIO.

    Redundant with fetch_content_task's storage, but kept for DAG flexibility.

    Returns:
        StoreContentOutput with storage path
    """
    logger.info(
        f"Storing content: item_id={input.item_id}, "
        f"tenant={input.tenant_id}, size={len(input.content)} bytes"
    )

    storage = get_minio_adapter()
    raw_content_path = await asyncio.to_thread(
        storage.store_raw_content,
        tenant_id=input.tenant_id,
        item_id=input.item_id,
        filename=input.filename,
        content=input.content,
        content_type=input.content_type,
    )

    logger.info(f"Content stored: path={raw_content_path}")

    return StoreContentOutput(
        item_id=input.item_id,
        raw_content_path=raw_content_path,
    )


# ============================================================================
# Hatchet Task: parse_document_task
# ============================================================================


async def parse_document_task(input: ParseDocumentInput) -> ParseDocumentOutput:
    """
    Task 4: Parse document content using ParserRegistry.

    Extracts:
    - Text content
    - Images with metadata
    - Tables
    - Links

    Stores parsed text to MinIO.

    Returns:
        ParseDocumentOutput with parsing stats
    """
    logger.info(
        f"Parsing document: item_id={input.item_id}, "
        f"source_type={input.source_type}, filename={input.filename}"
    )

    # Get parser from registry
    registry = get_parser_registry()

    from pathlib import Path

    extension = Path(input.filename).suffix or ""

    try:
        # Parse document
        parsed_doc = registry.parse(
            content=input.content,
            filename=input.filename,
            source_type=input.source_type,
            mime_type=input.mime_type or "",
        )

        # Get max document length from config
        max_chars = ingestion_config.max_document_chars

        # Track original length for metadata
        original_length = len(parsed_doc.text)
        original_token_count = count_tokens(parsed_doc.text)
        was_truncated = False

        # Truncate if exceeds max length
        if original_length > max_chars:
            marker = "\n\n[Document truncated due to length limit]\n\n"
            available = max_chars - len(marker)
            if available > 0:
                head_len = available // 2
                tail_len = available - head_len
                parsed_doc.text = (
                    parsed_doc.text[:head_len]
                    + marker
                    + parsed_doc.text[-tail_len:]
                )
            else:
                parsed_doc.text = truncate_text(
                    parsed_doc.text,
                    max_length=max_chars,
                    suffix=marker.strip(),
                    preserve_words=True,
                )
            was_truncated = True
            logger.warning(
                f"Document truncated: {original_length} -> {len(parsed_doc.text)} chars "
                f"(max={max_chars})"
            )

        logger.info(
            f"Document parsed: {len(parsed_doc.text)} chars, "
            f"{len(parsed_doc.images)} images, "
            f"{len(parsed_doc.tables)} tables, "
            f"{len(parsed_doc.links)} links"
        )

        # Store parsed text to MinIO (sync method, run in thread)
        storage = get_minio_adapter()
        parsed_text_path = await asyncio.to_thread(
            storage.store_parsed_text,
            tenant_id=input.tenant_id,
            item_id=input.item_id,
            text=parsed_doc.text,
        )

        # Store images to MinIO (if any)
        for image in parsed_doc.images:
            if image.content:
                # Extract extension from mime_type (e.g., "image/png" → "png")
                extension = image.mime_type.split('/')[-1] if '/' in image.mime_type else 'png'
                await asyncio.to_thread(
                    storage.store_image,
                    tenant_id=input.tenant_id,
                    item_id=input.item_id,
                    image_id=image.image_id,
                    image_content=image.content,
                    extension=extension,
                )

        # Update IngestionItem with parsing results
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select

            stmt = select(IngestionItem).where(IngestionItem.id == input.item_id)
            result = await session.execute(stmt)
            item = result.scalar_one_or_none()

            if item:
                item.parsed_ref = parsed_text_path
                item.image_count = len(parsed_doc.images)
                item.table_count = len(parsed_doc.tables)

                # Record original length and truncation info in metadata
                # (per design doc: record original_char_count in metadata)
                if item.metadata_json is None:
                    item.metadata_json = {}
                item.metadata_json["original_char_count"] = original_length
                item.metadata_json["final_char_count"] = len(parsed_doc.text)
                item.metadata_json["original_token_count"] = original_token_count
                item.metadata_json["final_token_count"] = count_tokens(parsed_doc.text)
                if was_truncated:
                    item.metadata_json["was_truncated"] = True
                    item.metadata_json["truncation_limit"] = max_chars

                await session.commit()

        logger.info(f"Parsed text stored: path={parsed_text_path}")

        return ParseDocumentOutput(
            item_id=input.item_id,
            parsed_ref=parsed_text_path,
            text_length=len(parsed_doc.text),
            image_count=len(parsed_doc.images),
            table_count=len(parsed_doc.tables),
            link_count=len(parsed_doc.links),
        )

    except Exception as e:
        logger.error(f"Failed to parse document {input.item_id}: {e}")

        # Update status to FAILED
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select

            stmt = select(IngestionItem).where(IngestionItem.id == input.item_id)
            result = await session.execute(stmt)
            item = result.scalar_one_or_none()

            if item:
                item.status = IngestionStatus.FAILED
                item.error = str(e)
                item.completed_at = datetime.utcnow()
                await session.commit()

        raise


# ============================================================================
# Helper Functions
# ============================================================================


async def _fetch_file_content(file_path: str) -> bytes:
    """Fetch content from local file."""
    import aiofiles

    async with aiofiles.open(file_path, "rb") as f:
        return await f.read()


async def _fetch_url_content(url: str) -> bytes:
    """Fetch content from URL."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.read()


def _guess_content_type(filename: str) -> str:
    """Guess MIME type from filename extension."""
    import mimetypes

    content_type, _ = mimetypes.guess_type(filename)
    return content_type or "application/octet-stream"
