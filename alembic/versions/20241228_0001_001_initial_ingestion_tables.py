"""Initial ingestion tables

Revision ID: 001
Revises: None
Create Date: 2024-12-28

Creates the core ingestion tracking tables:
- ingestion_items: Document lifecycle tracking
- ingestion_images: Image-level OCR tracking
- ingestion_batches: Batch import tracking
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum types first
    source_type_enum = postgresql.ENUM(
        "file", "url", "text", "markdown",
        name="source_type_enum",
        create_type=False,
    )
    source_type_enum.create(op.get_bind(), checkfirst=True)

    ingestion_status_enum = postgresql.ENUM(
        "PENDING", "PROCESSING", "INDEXED", "FAILED", "FAILED_PARTIAL", "DUPLICATE",
        name="ingestion_status_enum",
        create_type=False,
    )
    ingestion_status_enum.create(op.get_bind(), checkfirst=True)

    ocr_status_enum = postgresql.ENUM(
        "PENDING", "PROCESSING", "COMPLETED", "FAILED", "SKIPPED",
        name="ocr_status_enum",
        create_type=False,
    )
    ocr_status_enum.create(op.get_bind(), checkfirst=True)

    # Create ingestion_items table
    op.create_table(
        "ingestion_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("tenant_id", sa.String(255), nullable=False, index=True),
        sa.Column(
            "source_type",
            postgresql.ENUM("file", "url", "text", "markdown", name="source_type_enum", create_type=False),
            nullable=False,
        ),
        sa.Column("source_uri", sa.Text, nullable=False),
        sa.Column("file_name", sa.String(512), nullable=False),
        sa.Column("mime_type", sa.String(255), nullable=True),
        sa.Column("size_bytes", sa.BigInteger, nullable=False, default=0),
        sa.Column("content_hash", sa.String(64), nullable=False, index=True),
        sa.Column("content_ref", sa.Text, nullable=True),
        sa.Column("parsed_ref", sa.Text, nullable=True),
        sa.Column(
            "status",
            postgresql.ENUM(
                "PENDING", "PROCESSING", "INDEXED", "FAILED", "FAILED_PARTIAL", "DUPLICATE",
                name="ingestion_status_enum",
                create_type=False,
            ),
            nullable=False,
            default="PENDING",
            index=True,
        ),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("retry_count", sa.Integer, nullable=False, default=0),
        sa.Column("last_attempt_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("document_id", sa.String(255), nullable=True, index=True),
        sa.Column("chunk_count", sa.Integer, nullable=True, default=0),
        sa.Column("image_count", sa.Integer, nullable=True, default=0),
        sa.Column("table_count", sa.Integer, nullable=True, default=0),
        sa.Column("metadata_json", postgresql.JSONB, nullable=True, default=dict),
        sa.Column("webhook_url", sa.Text, nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )

    # Create composite indexes for ingestion_items
    op.create_index(
        "idx_ingestion_items_tenant_status",
        "ingestion_items",
        ["tenant_id", "status"],
    )
    op.create_index(
        "idx_ingestion_items_content_hash",
        "ingestion_items",
        ["tenant_id", "content_hash"],
    )
    op.create_index(
        "idx_ingestion_items_document_id",
        "ingestion_items",
        ["document_id"],
    )

    # Create ingestion_images table
    op.create_table(
        "ingestion_images",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("item_id", postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("tenant_id", sa.String(255), nullable=False, index=True),
        sa.Column("image_id", sa.String(255), nullable=False),
        sa.Column("page_number", sa.Integer, nullable=True),
        sa.Column("mime_type", sa.String(255), nullable=False),
        sa.Column("size_bytes", sa.BigInteger, nullable=False, default=0),
        sa.Column("image_ref", sa.Text, nullable=True),
        sa.Column("ocr_ref", sa.Text, nullable=True),
        sa.Column(
            "ocr_status",
            postgresql.ENUM(
                "PENDING", "PROCESSING", "COMPLETED", "FAILED", "SKIPPED",
                name="ocr_status_enum",
                create_type=False,
            ),
            nullable=False,
            default="PENDING",
        ),
        sa.Column("ocr_text", sa.Text, nullable=True),
        sa.Column("ocr_confidence", sa.Integer, nullable=True),
        sa.Column("ocr_error", sa.Text, nullable=True),
        sa.Column("caption", sa.Text, nullable=True),
        sa.Column("metadata_json", postgresql.JSONB, nullable=True, default=dict),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # Create composite indexes for ingestion_images
    op.create_index(
        "idx_ingestion_images_item_id",
        "ingestion_images",
        ["item_id"],
    )
    op.create_index(
        "idx_ingestion_images_tenant_status",
        "ingestion_images",
        ["tenant_id", "ocr_status"],
    )

    # Create ingestion_batches table
    op.create_table(
        "ingestion_batches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("tenant_id", sa.String(255), nullable=False, index=True),
        sa.Column("batch_name", sa.String(512), nullable=True),
        sa.Column("total_items", sa.Integer, nullable=False, default=0),
        sa.Column("completed_items", sa.Integer, nullable=False, default=0),
        sa.Column("failed_items", sa.Integer, nullable=False, default=0),
        sa.Column(
            "status",
            postgresql.ENUM(
                "PENDING", "PROCESSING", "INDEXED", "FAILED", "FAILED_PARTIAL", "DUPLICATE",
                name="ingestion_status_enum",
                create_type=False,
            ),
            nullable=False,
            default="PENDING",
        ),
        sa.Column("metadata_json", postgresql.JSONB, nullable=True, default=dict),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # Create composite index for ingestion_batches
    op.create_index(
        "idx_ingestion_batches_tenant_status",
        "ingestion_batches",
        ["tenant_id", "status"],
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table("ingestion_batches")
    op.drop_table("ingestion_images")
    op.drop_table("ingestion_items")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS ocr_status_enum")
    op.execute("DROP TYPE IF EXISTS ingestion_status_enum")
    op.execute("DROP TYPE IF EXISTS source_type_enum")
