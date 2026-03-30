"""Database models and validation schemas for ingestion and indexing."""

import enum
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Column,
    Enum,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()


# ============================================================================
# Enums
# ============================================================================


class SourceType(str, enum.Enum):
    """Source type for ingestion items.

    Note: Values must match enum values in PostgreSQL database (lowercase).
    """
    FILE = "file"
    URL = "url"
    TEXT = "text"
    MARKDOWN = "markdown"


class IngestionStatus(str, enum.Enum):
    """Status of ingestion item processing."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"
    FAILED_PARTIAL = "FAILED_PARTIAL"
    DUPLICATE = "DUPLICATE"


class OCRStatus(str, enum.Enum):
    """OCR processing status for images."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# ============================================================================
# SQLAlchemy Models
# ============================================================================


class IngestionItem(Base):
    """
    Ingestion item tracking table.

    Tracks the lifecycle of a document from raw content ingestion through
    parsing, OCR, chunking, embedding, and indexing.
    """
    __tablename__ = "ingestion_items"

    # Primary key
    id = Column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )

    # Tenant isolation
    tenant_id = Column(String(255), nullable=False, index=True)

    # Source information
    source_type = Column(
        Enum(
            SourceType,
            name="source_type_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    source_uri = Column(Text, nullable=False)  # URL or original file path
    file_name = Column(String(512), nullable=False)
    mime_type = Column(String(255), nullable=True)
    size_bytes = Column(BigInteger, nullable=False, default=0)

    # Deduplication
    content_hash = Column(
        String(64),
        nullable=False,
        index=True,
    )  # SHA-256 hash of normalized content + source_type

    # MinIO references
    content_ref = Column(Text, nullable=True)  # raw/{tenant_id}/{item_id}/{filename}
    parsed_ref = Column(Text, nullable=True)  # parsed/{tenant_id}/{item_id}/text.md

    # Processing status
    status = Column(
        Enum(IngestionStatus, name="ingestion_status_enum"),
        nullable=False,
        default=IngestionStatus.PENDING,
        index=True,
    )
    error = Column(Text, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    last_attempt_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Index reference
    document_id = Column(
        String(255),
        nullable=True,
        index=True,
    )  # Final document_id in vector index
    chunk_count = Column(Integer, nullable=True, default=0)
    image_count = Column(Integer, nullable=True, default=0)
    table_count = Column(Integer, nullable=True, default=0)

    # Metadata
    metadata_json = Column(
        JSONB,
        nullable=True,
        default=dict,
    )  # Flexible metadata (title, author, etc.)

    # Webhook callback
    webhook_url = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    completed_at = Column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )

    __table_args__ = (
        Index(
            "idx_ingestion_items_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "idx_ingestion_items_content_hash",
            "tenant_id",
            "content_hash",
        ),
        Index(
            "idx_ingestion_items_document_id",
            "document_id",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<IngestionItem(id={self.id}, tenant_id={self.tenant_id}, "
            f"status={self.status}, document_id={self.document_id})>"
        )


class IngestionImage(Base):
    """
    Optional table for tracking image-level OCR status.

    Tracks individual images extracted from documents with separate
    OCR processing status and MinIO references.
    """
    __tablename__ = "ingestion_images"

    # Primary key
    id = Column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )

    # Parent reference
    item_id = Column(
        PG_UUID(as_uuid=True),
        nullable=False,
        index=True,
    )  # References ingestion_items.id
    tenant_id = Column(String(255), nullable=False, index=True)

    # Image information
    image_id = Column(String(255), nullable=False)  # Unique within item
    page_number = Column(Integer, nullable=True)
    mime_type = Column(String(255), nullable=False)
    size_bytes = Column(BigInteger, nullable=False, default=0)

    # MinIO references
    image_ref = Column(Text, nullable=True)  # images/{tenant_id}/{item_id}/{image_id}.{ext}
    ocr_ref = Column(Text, nullable=True)  # ocr/{tenant_id}/{item_id}/{image_id}.json

    # OCR processing
    ocr_status = Column(
        Enum(OCRStatus, name="ocr_status_enum"),
        nullable=False,
        default=OCRStatus.PENDING,
    )
    ocr_text = Column(Text, nullable=True)
    ocr_confidence = Column(Integer, nullable=True)  # 0-100
    ocr_error = Column(Text, nullable=True)

    # Metadata
    caption = Column(Text, nullable=True)
    metadata_json = Column(JSONB, nullable=True, default=dict)

    # Timestamps
    created_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    __table_args__ = (
        Index(
            "idx_ingestion_images_item_id",
            "item_id",
        ),
        Index(
            "idx_ingestion_images_tenant_status",
            "tenant_id",
            "ocr_status",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<IngestionImage(id={self.id}, item_id={self.item_id}, "
            f"image_id={self.image_id}, ocr_status={self.ocr_status})>"
        )


class IngestionBatch(Base):
    """
    Optional table for tracking batch import operations.

    Groups multiple ingestion items into logical batches for bulk
    import tracking and status monitoring.
    """
    __tablename__ = "ingestion_batches"

    # Primary key
    id = Column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )

    # Tenant isolation
    tenant_id = Column(String(255), nullable=False, index=True)

    # Batch information
    batch_name = Column(String(512), nullable=True)
    total_items = Column(Integer, nullable=False, default=0)
    completed_items = Column(Integer, nullable=False, default=0)
    failed_items = Column(Integer, nullable=False, default=0)

    # Status
    status = Column(
        Enum(IngestionStatus, name="ingestion_status_enum"),
        nullable=False,
        default=IngestionStatus.PENDING,
    )

    # Metadata
    metadata_json = Column(JSONB, nullable=True, default=dict)

    # Timestamps
    created_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    __table_args__ = (
        Index(
            "idx_ingestion_batches_tenant_status",
            "tenant_id",
            "status",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<IngestionBatch(id={self.id}, tenant_id={self.tenant_id}, "
            f"total_items={self.total_items}, status={self.status})>"
        )


# ============================================================================
# Pydantic Validation Models
# ============================================================================


class IngestionItemCreate(BaseModel):
    """Request schema for creating a new ingestion item."""

    tenant_id: str = Field(..., min_length=1, max_length=255)
    source_type: SourceType
    source_uri: str = Field(..., min_length=1)
    file_name: str = Field(..., min_length=1, max_length=512)
    mime_type: Optional[str] = Field(None, max_length=255)
    size_bytes: int = Field(0, ge=0)

    # Optional explicit document_id (overrides hash-based dedup)
    document_id: Optional[str] = Field(None, max_length=255)

    # Optional metadata
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)

    # Optional webhook
    webhook_url: Optional[str] = None

    # Force reindex flag
    force_reindex: bool = False

    @field_validator("source_uri")
    @classmethod
    def validate_source_uri(cls, v: str) -> str:
        """Validate source URI is not empty."""
        if not v or not v.strip():
            raise ValueError("source_uri cannot be empty")
        return v.strip()

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        """Validate file name is not empty."""
        if not v or not v.strip():
            raise ValueError("file_name cannot be empty")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "tenant_id": "default",
                "source_type": "file",
                "source_uri": "/uploads/document.pdf",
                "file_name": "document.pdf",
                "mime_type": "application/pdf",
                "size_bytes": 1024000,
                "metadata": {
                    "title": "Sample Document",
                    "author": "John Doe",
                },
            }
        }


class IngestionItemResponse(BaseModel):
    """Response schema for ingestion item."""

    id: UUID
    tenant_id: str
    source_type: SourceType
    source_uri: str
    file_name: str
    mime_type: Optional[str]
    size_bytes: int
    content_hash: str
    content_ref: Optional[str]
    parsed_ref: Optional[str]
    status: IngestionStatus
    error: Optional[str]
    retry_count: int
    last_attempt_at: Optional[datetime]
    document_id: Optional[str]
    chunk_count: Optional[int]
    metadata_json: Optional[dict[str, Any]]
    webhook_url: Optional[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "tenant_id": "default",
                "source_type": "file",
                "source_uri": "/uploads/document.pdf",
                "file_name": "document.pdf",
                "mime_type": "application/pdf",
                "size_bytes": 1024000,
                "content_hash": "a3c7e...",
                "content_ref": "raw/default/550e8400.../document.pdf",
                "parsed_ref": "parsed/default/550e8400.../text.md",
                "status": "INDEXED",
                "error": None,
                "retry_count": 0,
                "last_attempt_at": None,
                "document_id": "doc_550e8400",
                "chunk_count": 15,
                "metadata_json": {"title": "Sample Document"},
                "webhook_url": None,
                "created_at": "2025-12-28T10:00:00Z",
                "updated_at": "2025-12-28T10:05:00Z",
                "completed_at": "2025-12-28T10:06:00Z",
            }
        }


class IngestionItemUpdate(BaseModel):
    """Schema for updating ingestion item fields."""

    status: Optional[IngestionStatus] = None
    error: Optional[str] = None
    content_ref: Optional[str] = None
    parsed_ref: Optional[str] = None
    document_id: Optional[str] = None
    chunk_count: Optional[int] = None
    metadata_json: Optional[dict[str, Any]] = None
    completed_at: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "INDEXED",
                "document_id": "doc_550e8400",
                "chunk_count": 15,
            }
        }


class IngestionImageCreate(BaseModel):
    """Request schema for creating ingestion image record."""

    item_id: UUID
    tenant_id: str = Field(..., min_length=1, max_length=255)
    image_id: str = Field(..., min_length=1, max_length=255)
    page_number: Optional[int] = Field(None, ge=0)
    mime_type: str = Field(..., max_length=255)
    size_bytes: int = Field(0, ge=0)
    caption: Optional[str] = None
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "item_id": "550e8400-e29b-41d4-a716-446655440000",
                "tenant_id": "default",
                "image_id": "img_001",
                "page_number": 1,
                "mime_type": "image/png",
                "size_bytes": 204800,
                "caption": "Figure 1: Architecture Diagram",
            }
        }


class IngestionImageResponse(BaseModel):
    """Response schema for ingestion image."""

    id: UUID
    item_id: UUID
    tenant_id: str
    image_id: str
    page_number: Optional[int]
    mime_type: str
    size_bytes: int
    image_ref: Optional[str]
    ocr_ref: Optional[str]
    ocr_status: OCRStatus
    ocr_text: Optional[str]
    ocr_confidence: Optional[int]
    ocr_error: Optional[str]
    caption: Optional[str]
    metadata_json: Optional[dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class IngestionBatchCreate(BaseModel):
    """Request schema for creating ingestion batch."""

    tenant_id: str = Field(..., min_length=1, max_length=255)
    batch_name: Optional[str] = Field(None, max_length=512)
    total_items: int = Field(0, ge=0)
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "tenant_id": "default",
                "batch_name": "Monthly Reports Import",
                "total_items": 100,
                "metadata": {"source": "s3://bucket/reports/"},
            }
        }


class IngestionBatchResponse(BaseModel):
    """Response schema for ingestion batch."""

    id: UUID
    tenant_id: str
    batch_name: Optional[str]
    total_items: int
    completed_items: int
    failed_items: int
    status: IngestionStatus
    metadata_json: Optional[dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Helper Models
# ============================================================================


class DeduplicationResult(BaseModel):
    """Result of deduplication check."""

    is_duplicate: bool
    existing_item_id: Optional[UUID] = None
    existing_document_id: Optional[str] = None
    action: str  # "create_new", "skip_duplicate", "reprocess", "update"
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "is_duplicate": True,
                "existing_item_id": "550e8400-e29b-41d4-a716-446655440000",
                "existing_document_id": "doc_550e8400",
                "action": "skip_duplicate",
                "message": "Content already indexed with document_id doc_550e8400",
            }
        }
