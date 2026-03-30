"""API schemas for ingestion endpoints."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


# ============================================================================
# Request Schemas
# ============================================================================


class IngestFileRequest(BaseModel):
    """Request to ingest a file."""

    tenant_id: str = Field(..., description="Tenant ID for multi-tenancy")
    source_uri: str = Field(..., description="File path or URL to ingest")
    filename: str = Field(..., description="Original filename")
    document_id: Optional[str] = Field(
        None, description="Optional custom document ID for explicit mapping"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    force_reindex: bool = Field(
        False, description="Force reindexing if document already exists"
    )

    # Configuration overrides
    chunking_config: dict[str, Any] = Field(
        default_factory=dict, description="Chunking configuration overrides"
    )
    embedding_config: dict[str, Any] = Field(
        default_factory=dict, description="Embedding configuration overrides"
    )
    ocr_provider: str = Field("tesseract", description="OCR provider to use")
    index_name: str = Field("default", description="Target index name")

    # Webhook callback
    webhook_url: Optional[HttpUrl] = Field(
        None, description="Webhook URL for completion/failure notifications"
    )


class IngestURLRequest(BaseModel):
    """Request to ingest a URL."""

    tenant_id: str
    source_url: HttpUrl = Field(..., description="URL to fetch and ingest")
    document_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    force_reindex: bool = False

    chunking_config: dict[str, Any] = Field(default_factory=dict)
    embedding_config: dict[str, Any] = Field(default_factory=dict)
    ocr_provider: str = "tesseract"
    index_name: str = "default"
    webhook_url: Optional[HttpUrl] = None


class IngestTextRequest(BaseModel):
    """Request to ingest raw text or markdown."""

    tenant_id: str
    text_content: str = Field(..., description="Text or markdown content to ingest")
    filename: str = Field("document.md", description="Virtual filename")
    content_type: str = Field(
        "text", description="Content type: text or markdown"
    )
    document_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    force_reindex: bool = False

    chunking_config: dict[str, Any] = Field(default_factory=dict)
    embedding_config: dict[str, Any] = Field(default_factory=dict)
    index_name: str = "default"
    webhook_url: Optional[HttpUrl] = None


class BatchIngestRequest(BaseModel):
    """Request to ingest multiple documents."""

    tenant_id: str
    items: list[IngestFileRequest | IngestURLRequest | IngestTextRequest] = Field(
        ..., description="List of documents to ingest"
    )
    webhook_url: Optional[HttpUrl] = Field(
        None, description="Webhook URL for batch completion"
    )


# ============================================================================
# Response Schemas
# ============================================================================


class IngestResponse(BaseModel):
    """Response for single ingestion request."""

    item_id: UUID = Field(..., description="Unique ingestion item ID")
    tenant_id: str
    source_type: str
    source_uri: str
    status: str = Field(..., description="Current ingestion status")
    workflow_run_id: Optional[str] = Field(
        None, description="Hatchet workflow run ID"
    )
    created_at: datetime
    message: str = Field(..., description="Status message")


class BatchIngestResponse(BaseModel):
    """Response for batch ingestion request."""

    batch_id: str = Field(..., description="Batch operation ID")
    tenant_id: str
    total_items: int
    accepted_items: int
    rejected_items: int
    item_ids: list[UUID]
    message: str


class IngestionStatusResponse(BaseModel):
    """Response for status query."""

    item_id: UUID
    tenant_id: str
    source_type: str
    source_uri: str
    status: str
    document_id: Optional[str]
    chunk_count: int
    image_count: int
    table_count: int
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    error: Optional[str]


class IngestionStatsResponse(BaseModel):
    """Response for statistics query."""

    tenant_id: Optional[str]
    total_items: int
    pending_items: int
    processing_items: int
    indexed_items: int
    failed_items: int
    duplicate_items: int

    # Performance stats
    avg_processing_time_seconds: Optional[float]
    total_chunks_indexed: int
    total_bytes_processed: int


# ============================================================================
# Error Schemas
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict[str, Any]] = Field(None, description="Additional details")
    item_id: Optional[UUID] = Field(None, description="Related item ID if applicable")
