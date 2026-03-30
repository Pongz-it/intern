"""Hatchet DAG workflow for document ingestion pipeline.

Workflow Stages:
1. Fetch Content → Dedup Check → [Skip or Process]
2. Parse Document → Extract Images (optional)
3. OCR Images (if images exist)
4. Chunk Document → Embed Chunks → Index Chunks

Error Handling:
- Task-level retries with exponential backoff
- Per-document failure isolation in embedding
- Status tracking in IngestionItem table
- Webhook callbacks on completion/failure

Concurrency Control:
- Tenant-level rate limiting
- Worker role separation (ingestion-worker, ocr-worker, indexing-worker)
"""

import logging
from datetime import datetime, timedelta

from hatchet_sdk import ConcurrencyExpression, ConcurrencyLimitStrategy, Context
from hatchet_sdk.rate_limit import RateLimit, RateLimitDuration
from pydantic import BaseModel

from agent_rag.ingestion.workflow.tasks.indexing_tasks import (
    ChunkDocumentInput,
    EmbedChunksInput,
    IndexChunksInput,
    chunk_document_task,
    embed_chunks_task,
    index_chunks_task,
)
from agent_rag.ingestion.workflow.tasks.ingestion_tasks import (
    DedupCheckInput,
    FetchContentInput,
    ParseDocumentInput,
    dedup_check_task,
    fetch_content_task,
    parse_document_task,
)
from agent_rag.ingestion.workflow.tasks.ocr_tasks import (
    ExtractImagesInput,
    OCRImagesInput,
    extract_images_task,
    ocr_images_task,
)

logger = logging.getLogger(__name__)

# Import hatchet client (assuming initialized in hatchet_client.py)
from hatchet_client import hatchet


# ============================================================================
# Workflow Input Schema
# ============================================================================


class IngestionWorkflowInput(BaseModel):
    """Input for the complete ingestion workflow."""

    tenant_id: str
    source_type: str  # file, url, text, markdown
    source_uri: str
    filename: str
    document_id: str | None = None
    metadata: dict = {}
    force_reindex: bool = False

    # Configuration overrides
    chunking_config: dict = {}
    embedding_config: dict = {}
    ocr_provider: str = "tesseract"
    index_name: str = "default"

    # Webhook callback
    webhook_url: str | None = None


# ============================================================================
# Hatchet Workflow Definition
# ============================================================================

ingestion_workflow = hatchet.workflow(
    name="DocumentIngestionWorkflow",
    # Tenant-level concurrency control
    concurrency=ConcurrencyExpression(
        expression="input.tenant_id",  # Group by tenant_id
        max_runs=10,  # Max 10 concurrent ingestions per tenant
        limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
    ),
    input_validator=IngestionWorkflowInput,
)


# ============================================================================
# Task 1: Fetch Content
# ============================================================================


@ingestion_workflow.task(
    name="fetch-content",
    execution_timeout=timedelta(minutes=5),
    retries=3,
    backoff_factor=2.0,
    rate_limits=[
        # Global rate limit for all fetches
        RateLimit(
            static_key="fetch-content-global",
            units=1,
            limit=100,
            duration=RateLimitDuration.MINUTE,
        ),
        # Per-tenant rate limit
        RateLimit(
            dynamic_key="input.tenant_id",
            units=1,
            limit=50,
            duration=RateLimitDuration.MINUTE,
        ),
    ],
)
async def fetch_content(input: IngestionWorkflowInput, ctx: Context):
    """
    Fetch content from source_uri and store to MinIO.

    Returns:
        FetchContentOutput with item_id and content_hash
    """
    ctx.log(f"Fetching content: {input.source_type}://{input.source_uri}")

    fetch_input = FetchContentInput(
        tenant_id=input.tenant_id,
        source_type=input.source_type,
        source_uri=input.source_uri,
        filename=input.filename,
        document_id=input.document_id,
        metadata=input.metadata,
        force_reindex=input.force_reindex,
    )

    result = await fetch_content_task(fetch_input)

    ctx.log(
        f"Content fetched: item_id={result.item_id}, hash={result.content_hash[:8]}"
    )

    return result


# ============================================================================
# Task 2: Deduplication Check
# ============================================================================


@ingestion_workflow.task(
    name="dedup-check",
    parents=[fetch_content],
    execution_timeout=timedelta(seconds=30),
    retries=2,
)
async def dedup_check(input: IngestionWorkflowInput, ctx: Context):
    """
    Check if content is duplicate.

    Returns:
        DedupCheckOutput with action decision (skip, reindex, process)
    """
    # Get content_hash from fetch_content output with safe access
    fetch_output = ctx.task_output(fetch_content)

    # Handle EmptyModel or dict/object access safely
    if hasattr(fetch_output, 'content_hash'):
        content_hash = fetch_output.content_hash
    elif isinstance(fetch_output, dict) and 'content_hash' in fetch_output:
        content_hash = fetch_output['content_hash']
    else:
        ctx.log(f"ERROR: fetch_output has no content_hash, type={type(fetch_output)}, output={fetch_output}")
        raise ValueError("Missing content_hash in fetch_output")

    ctx.log(f"Checking duplicate: hash={content_hash[:8] if content_hash else 'None'}")

    dedup_input = DedupCheckInput(
        tenant_id=input.tenant_id,
        content_hash=content_hash,
        document_id=input.document_id,
        force_reindex=input.force_reindex,
    )

    result = await dedup_check_task(dedup_input)

    ctx.log(f"Dedup result: action={result.action}, {result.message}")

    return result


# ============================================================================
# Task 3: Parse Document (conditional)
# ============================================================================


@ingestion_workflow.task(
    name="parse-document",
    parents=[dedup_check],
    execution_timeout=timedelta(minutes=10),
    retries=3,
    backoff_factor=2.0,
)
async def parse_document(input: IngestionWorkflowInput, ctx: Context):
    """
    Parse document content (skip if duplicate).

    Returns:
        ParseDocumentOutput with parsing statistics
    """
    # Check if we should skip with safe access
    dedup_output = ctx.task_output(dedup_check)

    # Handle EmptyModel or dict/object access safely
    if hasattr(dedup_output, 'action'):
        dedup_action = dedup_output.action
    elif isinstance(dedup_output, dict) and 'action' in dedup_output:
        dedup_action = dedup_output['action']
    else:
        ctx.log(f"ERROR: dedup_output has no action, type={type(dedup_output)}")
        raise ValueError("Missing action in dedup_output")

    if dedup_action == "skip":
        ctx.log("Skipping parse: duplicate content")
        return None

    fetch_output = ctx.task_output(fetch_content)

    # Get item_id safely
    if hasattr(fetch_output, 'item_id'):
        current_item_id = fetch_output.item_id
    elif isinstance(fetch_output, dict) and 'item_id' in fetch_output:
        current_item_id = fetch_output['item_id']
    else:
        ctx.log(f"ERROR: fetch_output has no item_id, type={type(fetch_output)}")
        raise ValueError("Missing item_id in fetch_output")

    # Handle reindex/update: delete old chunks first
    # IMPORTANT: Only delete if existing_item_id is DIFFERENT from current item_id
    # to avoid deleting the content we just stored
    existing_item_id = getattr(dedup_output, 'existing_item_id', None) or dedup_output.get("existing_item_id") if isinstance(dedup_output, dict) else None
    if (
        dedup_action in ("reindex", "update")
        and existing_item_id
        and existing_item_id != current_item_id
    ):
        ctx.log(f"Reindexing: deleting old chunks for item {existing_item_id}")
        try:
            from agent_rag.ingestion.dedup import delete_old_chunks
            await delete_old_chunks(
                tenant_id=input.tenant_id,
                existing_item_id=existing_item_id,
                existing_document_id=getattr(dedup_output, 'existing_document_id', None) or dedup_output.get("existing_document_id") if isinstance(dedup_output, dict) else None,
            )
            ctx.log("Old chunks deleted successfully")
        except Exception as e:
            ctx.log(f"Warning: Failed to delete old chunks: {e}")
            # Continue with reindexing even if delete fails
    elif existing_item_id == current_item_id:
        ctx.log(f"Skipping old content deletion: existing_item_id equals current_item_id")

    ctx.log(f"Parsing document: item_id={current_item_id}")

    # Retrieve content from MinIO for parsing
    from agent_rag.ingestion.storage import get_minio_adapter

    storage = get_minio_adapter()
    content = await storage.retrieve_raw_content(
        tenant_id=input.tenant_id,
        item_id=current_item_id,
        filename=input.filename,
    )

    parse_input = ParseDocumentInput(
        item_id=current_item_id,
        tenant_id=input.tenant_id,
        source_type=input.source_type,
        filename=input.filename,
        content=content,
        mime_type=None,
    )

    result = await parse_document_task(parse_input)

    ctx.log(
        f"Document parsed: {result.text_length} chars, "
        f"{result.image_count} images, {result.table_count} tables"
    )

    return result


# ============================================================================
# Task 4: Extract Images (conditional)
# ============================================================================


@ingestion_workflow.task(
    name="extract-images",
    parents=[parse_document],
    execution_timeout=timedelta(minutes=5),
)
async def extract_images(input: IngestionWorkflowInput, ctx: Context):
    """
    Extract images from parsed document (skip if no images).

    Returns:
        ExtractImagesOutput with image IDs
    """
    parse_output = ctx.task_output(parse_document)

    # Handle EmptyModel safely
    if hasattr(parse_output, 'get'):
        image_count = parse_output.get("image_count", 0)
    elif hasattr(parse_output, 'image_count'):
        image_count = parse_output.image_count
    else:
        ctx.log("Skipping image extraction: no parsed output")
        return None

    if image_count == 0:
        ctx.log("Skipping image extraction: no images")
        return None

    fetch_output = ctx.task_output(fetch_content)

    # Get item_id safely
    if hasattr(fetch_output, 'item_id'):
        current_item_id = fetch_output.item_id
    elif isinstance(fetch_output, dict) and 'item_id' in fetch_output:
        current_item_id = fetch_output['item_id']
    else:
        ctx.log(f"ERROR: fetch_output has no item_id, type={type(fetch_output)}")
        raise ValueError("Missing item_id in fetch_output")

    # Get parsed_ref safely
    if hasattr(parse_output, 'parsed_ref'):
        parsed_ref = parse_output.parsed_ref
    elif isinstance(parse_output, dict) and 'parsed_ref' in parse_output:
        parsed_ref = parse_output['parsed_ref']
    else:
        ctx.log(f"ERROR: parse_output has no parsed_ref, type={type(parse_output)}")
        raise ValueError("Missing parsed_ref in parse_output")

    ctx.log(f"Extracting {image_count} images")

    extract_input = ExtractImagesInput(
        item_id=current_item_id,
        tenant_id=input.tenant_id,
        parsed_ref=parsed_ref,
        image_count=image_count,
    )

    result = await extract_images_task(extract_input)

    ctx.log(f"Images extracted: {result.image_count} images")

    return result


# ============================================================================
# Task 5: OCR Images (conditional)
# ============================================================================


@ingestion_workflow.task(
    name="ocr-images",
    parents=[extract_images],
    execution_timeout=timedelta(minutes=30),
    retries=2,
)
async def ocr_images(input: IngestionWorkflowInput, ctx: Context):
    """
    Perform OCR on extracted images (skip if no images).

    Returns:
        OCRImagesOutput with OCR results
    """
    from agent_rag.core.env_config import IngestionEnvConfig

    if not IngestionEnvConfig().ocr_enabled:
        ctx.log("Skipping OCR: disabled by configuration")
        return None

    # Handle case when extract_images returned None (no images in document)
    try:
        extract_output = ctx.task_output(extract_images)
    except ValueError:
        ctx.log("Skipping OCR: extract_images did not produce output")
        return None

    if not extract_output:
        ctx.log("Skipping OCR: no images")
        return None

    # Handle EmptyModel safely
    if hasattr(extract_output, 'get'):
        ocr_image_count = extract_output.get("image_count", 0)
    elif hasattr(extract_output, 'image_count'):
        ocr_image_count = extract_output.image_count
    else:
        ctx.log("Skipping OCR: extract output has no image_count")
        return None

    if ocr_image_count == 0:
        ctx.log("Skipping OCR: no images")
        return None

    fetch_output = ctx.task_output(fetch_content)

    # Get item_id safely
    if hasattr(fetch_output, 'item_id'):
        current_item_id = fetch_output.item_id
    elif isinstance(fetch_output, dict) and 'item_id' in fetch_output:
        current_item_id = fetch_output['item_id']
    else:
        ctx.log(f"ERROR: fetch_output has no item_id, type={type(fetch_output)}")
        raise ValueError("Missing item_id in fetch_output")

    # Get image_ids safely
    if hasattr(extract_output, 'image_ids'):
        image_ids = extract_output.image_ids
    elif isinstance(extract_output, dict) and 'image_ids' in extract_output:
        image_ids = extract_output['image_ids']
    else:
        ctx.log(f"ERROR: extract_output has no image_ids, type={type(extract_output)}")
        raise ValueError("Missing image_ids in extract_output")

    ctx.log(f"Performing OCR on {ocr_image_count} images")

    ocr_input = OCRImagesInput(
        item_id=current_item_id,
        tenant_id=input.tenant_id,
        image_ids=image_ids,
        ocr_provider=input.ocr_provider,
    )

    result = await ocr_images_task(ocr_input)

    ctx.log(f"OCR complete: {result.ocr_count} images processed")

    return result


# ============================================================================
# Task 6: Chunk Document
# ============================================================================


@ingestion_workflow.task(
    name="chunk-document",
    parents=[parse_document, ocr_images],  # Wait for parse and optional OCR
    execution_timeout=timedelta(minutes=10),
    retries=2,
)
async def chunk_document(input: IngestionWorkflowInput, ctx: Context):
    """
    Chunk document into semantic pieces.

    Returns:
        ChunkDocumentOutput with chunking statistics
    """
    # Handle case when parse_document returned None (e.g., duplicate skipped)
    try:
        parse_output = ctx.task_output(parse_document)
    except ValueError:
        ctx.log("Skipping chunk: parse_document did not produce output")
        return None

    if not parse_output:
        ctx.log("Skipping chunk: no parsed document")
        return None

    fetch_output = ctx.task_output(fetch_content)

    # Get item_id safely
    if hasattr(fetch_output, 'item_id'):
        current_item_id = fetch_output.item_id
    elif isinstance(fetch_output, dict) and 'item_id' in fetch_output:
        current_item_id = fetch_output['item_id']
    else:
        ctx.log(f"ERROR: fetch_output has no item_id, type={type(fetch_output)}")
        raise ValueError("Missing item_id in fetch_output")

    # Get parsed_ref safely
    if hasattr(parse_output, 'parsed_ref'):
        parsed_ref = parse_output.parsed_ref
    elif isinstance(parse_output, dict) and 'parsed_ref' in parse_output:
        parsed_ref = parse_output['parsed_ref']
    else:
        ctx.log(f"ERROR: parse_output has no parsed_ref, type={type(parse_output)}")
        raise ValueError("Missing parsed_ref in parse_output")

    ctx.log(f"Chunking document: item_id={current_item_id}")

    chunk_input = ChunkDocumentInput(
        item_id=current_item_id,
        tenant_id=input.tenant_id,
        source_type=input.source_type,
        parsed_ref=parsed_ref,
        chunking_config=input.chunking_config,
    )

    result = await chunk_document_task(chunk_input)

    ctx.log(
        f"Document chunked: {result.chunk_count} chunks "
        f"({result.large_chunk_count} large, {result.image_chunk_count} image, "
        f"{result.mini_chunk_count} mini-chunks)"
    )

    return result


# ============================================================================
# Task 7: Embed Chunks
# ============================================================================


@ingestion_workflow.task(
    name="embed-chunks",
    parents=[chunk_document],
    execution_timeout=timedelta(minutes=20),
    retries=3,
    backoff_factor=2.0,
    rate_limits=[
        # Global embedding rate limit
        RateLimit(
            static_key="embed-chunks-global",
            units=1,
            limit=50,
            duration=RateLimitDuration.MINUTE,
        ),
    ],
)
async def embed_chunks(input: IngestionWorkflowInput, ctx: Context):
    """
    Embed chunks with title caching and failure handling.

    Returns:
        EmbedChunksOutput with embedding statistics
    """
    # Handle case when chunk_document returned None
    try:
        chunk_output = ctx.task_output(chunk_document)
    except ValueError:
        ctx.log("Skipping embed: chunk_document did not produce output")
        return None

    if not chunk_output:
        ctx.log("Skipping embed: no chunks")
        return None

    fetch_output = ctx.task_output(fetch_content)

    # Get item_id safely
    if hasattr(fetch_output, 'item_id'):
        current_item_id = fetch_output.item_id
    elif isinstance(fetch_output, dict) and 'item_id' in fetch_output:
        current_item_id = fetch_output['item_id']
    else:
        ctx.log(f"ERROR: fetch_output has no item_id, type={type(fetch_output)}")
        raise ValueError("Missing item_id in fetch_output")

    # Get chunk_count safely
    if hasattr(chunk_output, 'chunk_count'):
        chunk_count = chunk_output.chunk_count
    elif isinstance(chunk_output, dict) and 'chunk_count' in chunk_output:
        chunk_count = chunk_output['chunk_count']
    else:
        ctx.log(f"ERROR: chunk_output has no chunk_count, type={type(chunk_output)}")
        raise ValueError("Missing chunk_count in chunk_output")

    ctx.log(f"Embedding {chunk_count} chunks")

    embed_input = EmbedChunksInput(
        item_id=current_item_id,
        tenant_id=input.tenant_id,
        chunk_count=chunk_count,
        embedding_config=input.embedding_config,
    )

    result = await embed_chunks_task(embed_input)

    ctx.log(
        f"Chunks embedded: {result.embedded_chunk_count}/{chunk_count} "
        f"({result.success_rate:.1%} success rate)"
    )

    return result


# ============================================================================
# Task 8: Index Chunks
# ============================================================================


@ingestion_workflow.task(
    name="index-chunks",
    parents=[embed_chunks],
    execution_timeout=timedelta(minutes=15),
    retries=3,
    backoff_factor=2.0,
)
async def index_chunks(input: IngestionWorkflowInput, ctx: Context):
    """
    Index embedded chunks to vector database.

    Returns:
        IndexChunksOutput with indexing status
    """
    # Handle case when embed_chunks returned None
    try:
        embed_output = ctx.task_output(embed_chunks)
    except ValueError:
        ctx.log("Skipping index: embed_chunks did not produce output")
        return None

    if not embed_output:
        ctx.log("Skipping index: no embedded chunks")
        return None

    fetch_output = ctx.task_output(fetch_content)

    # Get item_id safely
    if hasattr(fetch_output, 'item_id'):
        current_item_id = fetch_output.item_id
    elif isinstance(fetch_output, dict) and 'item_id' in fetch_output:
        current_item_id = fetch_output['item_id']
    else:
        ctx.log(f"ERROR: fetch_output has no item_id, type={type(fetch_output)}")
        raise ValueError("Missing item_id in fetch_output")

    # Get embedded_chunk_count safely
    if hasattr(embed_output, 'embedded_chunk_count'):
        embedded_chunk_count = embed_output.embedded_chunk_count
    elif isinstance(embed_output, dict) and 'embedded_chunk_count' in embed_output:
        embedded_chunk_count = embed_output['embedded_chunk_count']
    else:
        ctx.log(f"ERROR: embed_output has no embedded_chunk_count, type={type(embed_output)}")
        raise ValueError("Missing embedded_chunk_count in embed_output")

    ctx.log(f"Indexing {embedded_chunk_count} chunks")

    index_input = IndexChunksInput(
        item_id=current_item_id,
        tenant_id=input.tenant_id,
        embedded_chunk_count=embedded_chunk_count,
        index_name=input.index_name,
    )

    result = await index_chunks_task(index_input)

    if result.success:
        ctx.log(
            f"Chunks indexed successfully: {result.indexed_chunk_count} chunks "
            f"to {result.index_name}"
        )
    else:
        ctx.log(f"Indexing failed for item {current_item_id}")

    return result


# ============================================================================
# Failure Handler
# ============================================================================


@ingestion_workflow.on_failure_task()
async def handle_failure(input: IngestionWorkflowInput, ctx: Context):
    """
    Handle workflow failure.

    - Update IngestionItem status to FAILED
    - Send webhook notification (if configured)
    - Log error details
    """
    ctx.log(f"Workflow failed for tenant {input.tenant_id}")

    # Get errors
    errors = ctx.task_run_errors
    run_id = ctx.workflow_run_id

    ctx.log(f"Workflow run ID: {run_id}")
    ctx.log(f"Errors: {errors}")

    # Update IngestionItem status to FAILED
    # (if fetch_content completed and we have an item_id)
    try:
        fetch_output = ctx.task_output(fetch_content)
        if fetch_output:
            # Get item_id safely
            if hasattr(fetch_output, 'item_id'):
                current_item_id = fetch_output.item_id
            elif isinstance(fetch_output, dict) and 'item_id' in fetch_output:
                current_item_id = fetch_output['item_id']
            else:
                ctx.log(f"ERROR: fetch_output has no item_id, type={type(fetch_output)}")
                current_item_id = None

            if current_item_id:
                from agent_rag.core.database import AsyncSessionLocal
                from agent_rag.ingestion.models import IngestionStatus
                from sqlalchemy import select

                async with AsyncSessionLocal() as session:
                    from agent_rag.ingestion.models import IngestionItem

                    stmt = select(IngestionItem).where(
                        IngestionItem.id == current_item_id
                    )
                    result = await session.execute(stmt)
                    item = result.scalar_one_or_none()

                    if item:
                        item.status = IngestionStatus.FAILED
                        item.error = str(errors)
                        item.completed_at = datetime.utcnow()
                        await session.commit()

                        ctx.log(f"Updated item {current_item_id} status to FAILED")
    except Exception as e:
        ctx.log(f"Failed to update item status: {e}")

    # Send webhook notification (if configured)
    if input.webhook_url:
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                payload = {
                    "status": "failed",
                    "tenant_id": input.tenant_id,
                    "source_uri": input.source_uri,
                    "workflow_run_id": run_id,
                    "errors": str(errors),
                }

                async with session.post(input.webhook_url, json=payload) as response:
                    ctx.log(
                        f"Webhook notification sent: {response.status} "
                        f"to {input.webhook_url}"
                    )
        except Exception as e:
            ctx.log(f"Failed to send webhook notification: {e}")

    return {"handled": True, "run_id": run_id, "errors": str(errors)}
