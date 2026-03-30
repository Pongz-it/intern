"""FastAPI endpoints for document ingestion."""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_rag.core.database import AsyncSessionLocal
from agent_rag.ingestion.api.schemas import (
    BatchIngestRequest,
    BatchIngestResponse,
    ErrorResponse,
    IngestFileRequest,
    IngestResponse,
    IngestTextRequest,
    IngestURLRequest,
    IngestionStatsResponse,
    IngestionStatusResponse,
)
from agent_rag.ingestion.models import IngestionItem, IngestionStatus
from agent_rag.ingestion.workflow.ingestion_workflow import (
    IngestionWorkflowInput,
    ingestion_workflow,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ingestion", tags=["ingestion"])


# ============================================================================
# POST /ingest/file - Ingest a file
# ============================================================================


@router.post(
    "/ingest/file",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a file",
    description="Submit a file for ingestion and indexing",
)
async def ingest_file(request: IngestFileRequest) -> IngestResponse:
    """
    Ingest a file from local filesystem or URL.

    Triggers Hatchet workflow for async processing.

    Args:
        request: File ingestion request

    Returns:
        IngestResponse with item_id and workflow_run_id
    """
    logger.info(
        f"Ingestion request: tenant={request.tenant_id}, "
        f"source={request.source_uri}, filename={request.filename}"
    )

    try:
        # Create workflow input
        workflow_input = IngestionWorkflowInput(
            tenant_id=request.tenant_id,
            source_type="file",
            source_uri=request.source_uri,
            filename=request.filename,
            document_id=request.document_id,
            metadata=request.metadata,
            force_reindex=request.force_reindex,
            chunking_config=request.chunking_config,
            embedding_config=request.embedding_config,
            ocr_provider=request.ocr_provider,
            index_name=request.index_name,
            webhook_url=str(request.webhook_url) if request.webhook_url else None,
        )

        # Trigger Hatchet workflow (async)
        run_ref = ingestion_workflow.run_no_wait(workflow_input)

        # Note: item_id will be generated in fetch_content_task
        # For now, use workflow_run_id as placeholder

        logger.info(
            f"Workflow triggered: run_id={run_ref.workflow_run_id}, "
            f"tenant={request.tenant_id}"
        )

        # Return response (item_id will be populated after fetch_content completes)
        return IngestResponse(
            item_id=UUID(int=0),  # Placeholder, will be updated after workflow starts
            tenant_id=request.tenant_id,
            source_type="file",
            source_uri=request.source_uri,
            status="PENDING",
            workflow_run_id=run_ref.workflow_run_id,
            created_at=run_ref.created_at if hasattr(run_ref, "created_at") else None,
            message="Ingestion workflow submitted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to submit ingestion workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="IngestionError",
                message=f"Failed to submit ingestion workflow: {str(e)}",
            ).dict(),
        )


# ============================================================================
# POST /ingest/url - Ingest a URL
# ============================================================================


@router.post(
    "/ingest/url",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a URL",
    description="Fetch and ingest content from a URL",
)
async def ingest_url(request: IngestURLRequest) -> IngestResponse:
    """
    Ingest content from a URL.

    Args:
        request: URL ingestion request

    Returns:
        IngestResponse with item_id and workflow_run_id
    """
    logger.info(f"URL ingestion request: tenant={request.tenant_id}, url={request.source_url}")

    try:
        # Extract filename from URL
        from pathlib import Path
        from urllib.parse import urlparse

        parsed_url = urlparse(str(request.source_url))
        filename = Path(parsed_url.path).name or "document.html"

        workflow_input = IngestionWorkflowInput(
            tenant_id=request.tenant_id,
            source_type="url",
            source_uri=str(request.source_url),
            filename=filename,
            document_id=request.document_id,
            metadata=request.metadata,
            force_reindex=request.force_reindex,
            chunking_config=request.chunking_config,
            embedding_config=request.embedding_config,
            ocr_provider=request.ocr_provider,
            index_name=request.index_name,
            webhook_url=str(request.webhook_url) if request.webhook_url else None,
        )

        run_ref = ingestion_workflow.run_no_wait(workflow_input)

        logger.info(f"URL workflow triggered: run_id={run_ref.workflow_run_id}")

        return IngestResponse(
            item_id=UUID(int=0),
            tenant_id=request.tenant_id,
            source_type="url",
            source_uri=str(request.source_url),
            status="PENDING",
            workflow_run_id=run_ref.workflow_run_id,
            created_at=run_ref.created_at if hasattr(run_ref, "created_at") else None,
            message="URL ingestion workflow submitted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to submit URL ingestion workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="IngestionError",
                message=f"Failed to submit URL ingestion workflow: {str(e)}",
            ).dict(),
        )


# ============================================================================
# POST /ingest/text - Ingest raw text
# ============================================================================


@router.post(
    "/ingest/text",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest text content",
    description="Ingest raw text or markdown content",
)
async def ingest_text(request: IngestTextRequest) -> IngestResponse:
    """
    Ingest raw text or markdown content.

    Args:
        request: Text ingestion request

    Returns:
        IngestResponse with item_id and workflow_run_id
    """
    logger.info(
        f"Text ingestion request: tenant={request.tenant_id}, "
        f"content_length={len(request.text_content)}"
    )

    try:
        workflow_input = IngestionWorkflowInput(
            tenant_id=request.tenant_id,
            source_type=request.content_type,
            source_uri=request.text_content,  # Content itself as source_uri
            filename=request.filename,
            document_id=request.document_id,
            metadata=request.metadata,
            force_reindex=request.force_reindex,
            chunking_config=request.chunking_config,
            embedding_config=request.embedding_config,
            index_name=request.index_name,
            webhook_url=str(request.webhook_url) if request.webhook_url else None,
        )

        run_ref = ingestion_workflow.run_no_wait(workflow_input)

        logger.info(f"Text workflow triggered: run_id={run_ref.workflow_run_id}")

        return IngestResponse(
            item_id=UUID(int=0),
            tenant_id=request.tenant_id,
            source_type=request.content_type,
            source_uri=f"text:{len(request.text_content)} chars",
            status="PENDING",
            workflow_run_id=run_ref.workflow_run_id,
            created_at=run_ref.created_at if hasattr(run_ref, "created_at") else None,
            message="Text ingestion workflow submitted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to submit text ingestion workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="IngestionError",
                message=f"Failed to submit text ingestion workflow: {str(e)}",
            ).dict(),
        )


# ============================================================================
# POST /ingest/batch - Batch ingestion
# ============================================================================


@router.post(
    "/ingest/batch",
    response_model=BatchIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch ingestion",
    description="Ingest multiple documents in a single request",
)
async def ingest_batch(request: BatchIngestRequest) -> BatchIngestResponse:
    """
    Ingest multiple documents in batch.

    Args:
        request: Batch ingestion request

    Returns:
        BatchIngestResponse with batch statistics
    """
    logger.info(
        f"Batch ingestion request: tenant={request.tenant_id}, "
        f"item_count={len(request.items)}"
    )

    import uuid

    batch_id = str(uuid.uuid4())
    item_ids = []
    accepted = 0
    rejected = 0

    for item_request in request.items:
        try:
            # Determine source type
            if isinstance(item_request, IngestFileRequest):
                source_type = "file"
                source_uri = item_request.source_uri
            elif isinstance(item_request, IngestURLRequest):
                source_type = "url"
                source_uri = str(item_request.source_url)
            elif isinstance(item_request, IngestTextRequest):
                source_type = item_request.content_type
                source_uri = item_request.text_content
            else:
                logger.warning(f"Unknown item type: {type(item_request)}")
                rejected += 1
                continue

            # Create workflow input
            workflow_input = IngestionWorkflowInput(
                tenant_id=request.tenant_id,
                source_type=source_type,
                source_uri=source_uri,
                filename=getattr(item_request, "filename", "document.txt"),
                document_id=getattr(item_request, "document_id", None),
                metadata=getattr(item_request, "metadata", {}),
                force_reindex=getattr(item_request, "force_reindex", False),
                chunking_config=getattr(item_request, "chunking_config", {}),
                embedding_config=getattr(item_request, "embedding_config", {}),
                ocr_provider=getattr(item_request, "ocr_provider", "tesseract"),
                index_name=getattr(item_request, "index_name", "default"),
                webhook_url=str(request.webhook_url)
                if request.webhook_url
                else None,
            )

            # Trigger workflow
            run_ref = ingestion_workflow.run_no_wait(workflow_input)

            item_ids.append(UUID(int=accepted))  # Placeholder
            accepted += 1

            logger.debug(f"Batch item {accepted} submitted: run_id={run_ref.workflow_run_id}")

        except Exception as e:
            logger.error(f"Failed to submit batch item: {e}")
            rejected += 1

    logger.info(
        f"Batch ingestion complete: batch_id={batch_id}, "
        f"accepted={accepted}, rejected={rejected}"
    )

    return BatchIngestResponse(
        batch_id=batch_id,
        tenant_id=request.tenant_id,
        total_items=len(request.items),
        accepted_items=accepted,
        rejected_items=rejected,
        item_ids=item_ids,
        message=f"Batch ingestion submitted: {accepted}/{len(request.items)} items accepted",
    )


# ============================================================================
# GET /status/{item_id} - Get ingestion status
# ============================================================================


@router.get(
    "/status/{item_id}",
    response_model=IngestionStatusResponse,
    summary="Get ingestion status",
    description="Query the status of an ingestion item",
)
async def get_ingestion_status(item_id: UUID) -> IngestionStatusResponse:
    """
    Get status of an ingestion item.

    Args:
        item_id: Ingestion item ID

    Returns:
        IngestionStatusResponse with current status
    """
    async with AsyncSessionLocal() as session:
        stmt = select(IngestionItem).where(IngestionItem.id == item_id)
        result = await session.execute(stmt)
        item = result.scalar_one_or_none()

        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    error="NotFound",
                    message=f"Ingestion item not found: {item_id}",
                    item_id=item_id,
                ).dict(),
            )

        return IngestionStatusResponse(
            item_id=item.id,
            tenant_id=item.tenant_id,
            source_type=str(item.source_type.value),
            source_uri=item.source_uri,
            status=str(item.status.value),
            document_id=item.document_id,
            chunk_count=item.chunk_count or 0,
            image_count=item.image_count or 0,
            table_count=item.table_count or 0,
            created_at=item.created_at,
            updated_at=item.updated_at,
            completed_at=item.completed_at,
            error=item.error,
        )


# ============================================================================
# GET /stats - Get ingestion statistics
# ============================================================================


@router.get(
    "/stats",
    response_model=IngestionStatsResponse,
    summary="Get ingestion statistics",
    description="Query ingestion statistics for a tenant or globally",
)
async def get_ingestion_stats(
    tenant_id: str | None = Query(None, description="Filter by tenant ID")
) -> IngestionStatsResponse:
    """
    Get ingestion statistics.

    Args:
        tenant_id: Optional tenant ID to filter by

    Returns:
        IngestionStatsResponse with statistics
    """
    async with AsyncSessionLocal() as session:
        # Base query
        base_stmt = select(IngestionItem)

        if tenant_id:
            base_stmt = base_stmt.where(IngestionItem.tenant_id == tenant_id)

        # Count by status
        total_items = await session.scalar(
            select(func.count()).select_from(base_stmt.subquery())
        )

        pending_items = await session.scalar(
            select(func.count()).select_from(
                base_stmt.where(IngestionItem.status == IngestionStatus.PENDING).subquery()
            )
        )

        processing_items = await session.scalar(
            select(func.count()).select_from(
                base_stmt.where(IngestionItem.status == IngestionStatus.PROCESSING).subquery()
            )
        )

        indexed_items = await session.scalar(
            select(func.count()).select_from(
                base_stmt.where(IngestionItem.status == IngestionStatus.INDEXED).subquery()
            )
        )

        failed_items = await session.scalar(
            select(func.count()).select_from(
                base_stmt.where(
                    IngestionItem.status.in_(
                        [IngestionStatus.FAILED, IngestionStatus.FAILED_PARTIAL]
                    )
                ).subquery()
            )
        )

        duplicate_items = await session.scalar(
            select(func.count()).select_from(
                base_stmt.where(IngestionItem.status == IngestionStatus.DUPLICATE).subquery()
            )
        )

        # Total chunks indexed
        total_chunks = await session.scalar(
            select(func.sum(IngestionItem.chunk_count)).select_from(
                base_stmt.where(IngestionItem.status == IngestionStatus.INDEXED).subquery()
            )
        ) or 0

        # Total bytes processed (sum of file sizes)
        total_bytes = await session.scalar(
            select(func.sum(IngestionItem.file_size)).select_from(
                base_stmt.where(IngestionItem.status == IngestionStatus.INDEXED).subquery()
            )
        ) or 0

        return IngestionStatsResponse(
            tenant_id=tenant_id,
            total_items=total_items or 0,
            pending_items=pending_items or 0,
            processing_items=processing_items or 0,
            indexed_items=indexed_items or 0,
            failed_items=failed_items or 0,
            duplicate_items=duplicate_items or 0,
            avg_processing_time_seconds=None,  # Would calculate from timestamps
            total_chunks_indexed=total_chunks,
            total_bytes_processed=total_bytes,
        )
