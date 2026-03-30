"""Hatchet workflow tasks for ingestion pipeline."""

from agent_rag.ingestion.workflow.tasks.indexing_tasks import (
    chunk_document_task,
    embed_chunks_task,
    index_chunks_task,
)
from agent_rag.ingestion.workflow.tasks.ingestion_tasks import (
    dedup_check_task,
    fetch_content_task,
    parse_document_task,
    store_content_task,
)
from agent_rag.ingestion.workflow.tasks.ocr_tasks import (
    extract_images_task,
    ocr_images_task,
)

__all__ = [
    # Ingestion tasks
    "fetch_content_task",
    "dedup_check_task",
    "store_content_task",
    "parse_document_task",
    # OCR tasks
    "extract_images_task",
    "ocr_images_task",
    # Indexing tasks
    "chunk_document_task",
    "embed_chunks_task",
    "index_chunks_task",
]
