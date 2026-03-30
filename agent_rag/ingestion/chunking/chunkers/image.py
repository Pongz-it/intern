"""Image-aware chunker for documents with images."""

import logging
from typing import Optional

from agent_rag.core.models import Chunk
from agent_rag.ingestion.chunking.base import (
    BaseChunker,
    count_tokens,
    truncate_to_tokens,
)
from agent_rag.ingestion.chunking.config import ChunkingConfig
from agent_rag.ingestion.models import IngestionItem
from agent_rag.ingestion.parsing.base import ParsedDocument, ParsedImage

logger = logging.getLogger(__name__)


# MIME types for image files
IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/svg+xml",
}

# File extensions for images
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".svg",
}


class ImageChunker(BaseChunker):
    """
    Specialized chunker for image-heavy documents.

    Features:
    - Creates dedicated chunks for each image with OCR text
    - Preserves image metadata (page number, caption)
    - Handles image-only documents (PDFs, scanned docs)
    - Combines OCR text with captions for searchability
    - Supports surrounding text context for images

    Use cases:
    - Scanned PDF documents
    - Image galleries
    - Documents with embedded diagrams/charts
    - Photo archives with metadata
    """

    @property
    def priority(self) -> int:
        """Medium-high priority - runs before semantic chunker but after code/table."""
        return 40

    @property
    def name(self) -> str:
        """Chunker name."""
        return "ImageChunker"

    def supports(self, source_type: str, mime_type: str, document: ParsedDocument) -> bool:
        """
        Check if this chunker supports the document.

        Args:
            source_type: Source type
            mime_type: MIME type
            document: Parsed document

        Returns:
            True if document is an image or contains significant images
        """
        # Pure image files
        if mime_type in IMAGE_MIME_TYPES:
            return True

        # Check file extension from metadata
        filename = document.metadata.get("filename", "") or document.metadata.get("file_name", "")
        if filename:
            filename_lower = filename.lower()
            for ext in IMAGE_EXTENSIONS:
                if filename_lower.endswith(ext):
                    return True

        # Documents with embedded images (e.g., PDFs with images)
        if document.images and len(document.images) > 0:
            # Use image chunker if images are significant portion of content
            # or if there's minimal text content
            text_tokens = count_tokens(document.text or "")
            image_count = len(document.images)

            # Use image chunker if:
            # 1. Multiple images with little text (likely scanned doc)
            # 2. Image-heavy document (>1 image per 500 tokens of text)
            if text_tokens < 100 and image_count > 0:
                return True
            if image_count > 0 and text_tokens / max(image_count, 1) < 500:
                return True

        return False

    def chunk(
        self,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """
        Chunk document focusing on images.

        Creates dedicated chunks for each image, combining:
        - OCR text (if available)
        - Image caption/alt text
        - Surrounding context text
        - Image metadata

        Args:
            document: Parsed document with images
            item: Ingestion item
            config: Chunking configuration

        Returns:
            List of image-focused chunks
        """
        if not config.create_image_chunks:
            logger.debug("Image chunk creation disabled in config")
            return []

        chunks = []
        chunk_id = 0

        # Build title prefix
        title = document.metadata.get("title") or item.file_name
        title_prefix = f"{title}\n\n" if title else ""

        # Get any OCR results from metadata
        ocr_results = document.metadata.get("ocr_results", {})

        # Process each image
        for image in document.images:
            chunk = self._create_image_chunk(
                chunk_id=chunk_id,
                image=image,
                ocr_results=ocr_results,
                title_prefix=title_prefix,
                document=document,
                item=item,
                config=config,
            )
            if chunk:
                chunks.append(chunk)
                chunk_id += 1

        # If document has text but also images, create text chunks for non-image content
        # This ensures we don't lose text that's not directly associated with images
        if document.text and document.text.strip():
            text_chunks = self._create_text_chunks(
                start_chunk_id=chunk_id,
                document=document,
                item=item,
                config=config,
                title_prefix=title_prefix,
            )
            chunks.extend(text_chunks)

        logger.info(
            f"ImageChunker generated {len(chunks)} chunks "
            f"({len(document.images)} images, {len(chunks) - len(document.images)} text)"
        )

        return chunks

    def _create_image_chunk(
        self,
        chunk_id: int,
        image: ParsedImage,
        ocr_results: dict,
        title_prefix: str,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> Optional[Chunk]:
        """
        Create a chunk for a single image.

        Args:
            chunk_id: Chunk ID
            image: ParsedImage object
            ocr_results: OCR results mapping image_id -> text
            title_prefix: Title prefix
            document: Parsed document
            item: Ingestion item
            config: Chunking configuration

        Returns:
            Chunk for the image, or None if no content
        """
        # Gather all text content for this image
        content_parts = []

        # Add OCR text if available
        ocr_text = ocr_results.get(image.image_id, "")
        if ocr_text:
            content_parts.append(f"[OCR Text]\n{ocr_text.strip()}")

        # Add caption if available
        if image.caption:
            content_parts.append(f"[Caption]\n{image.caption.strip()}")

        # If no content, create minimal chunk with image reference
        if not content_parts:
            content_parts.append(f"[Image: {image.image_id}]")
            if image.mime_type:
                content_parts.append(f"Type: {image.mime_type}")

        # Combine content
        content = "\n\n".join(content_parts)

        # Truncate if too long
        content = truncate_to_tokens(content, config.chunk_token_limit)

        # Build metadata
        chunk_metadata = dict(document.metadata or {})
        chunk_metadata.update({
            "image_id": image.image_id,
            "image_mime_type": image.mime_type,
            "chunk_type": "image",
        })
        if image.page_number is not None:
            chunk_metadata["page_number"] = image.page_number

        # Create blurb
        blurb = content[:config.blurb_size * 4] if content else f"Image: {image.image_id}"

        # Build semantic identifier
        semantic_id = title_prefix.strip()
        if image.page_number is not None:
            semantic_id += f" - Page {image.page_number}"

        return Chunk(
            document_id=item.document_id,
            chunk_id=chunk_id,
            content=content,
            title=item.file_name,
            source_type=str(item.source_type.value) if hasattr(item.source_type, "value") else item.source_type,
            link=item.source_uri,
            metadata=chunk_metadata,
            blurb=blurb,
            semantic_identifier=semantic_id,
            image_file_name=image.image_id,
            # Internal fields
            _title_prefix=title_prefix,
            _metadata_suffix_semantic=f"\nImage: {image.image_id}",
            _metadata_suffix_keyword=f"image {image.mime_type or ''} {image.image_id}",
        )

    def _create_text_chunks(
        self,
        start_chunk_id: int,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
        title_prefix: str,
    ) -> list[Chunk]:
        """
        Create chunks from remaining text content.

        Simple text chunking for non-image content in image-heavy documents.

        Args:
            start_chunk_id: Starting chunk ID
            document: Parsed document
            item: Ingestion item
            config: Chunking configuration
            title_prefix: Title prefix

        Returns:
            List of text chunks
        """
        text = document.text or ""
        if not text.strip():
            return []

        chunks = []
        chunk_id = start_chunk_id

        # Simple paragraph-based splitting
        paragraphs = text.split("\n\n")
        current_content = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if adding this paragraph exceeds limit
            test_content = f"{current_content}\n\n{para}".strip() if current_content else para
            if count_tokens(test_content) > config.chunk_token_limit:
                # Save current chunk if not empty
                if current_content:
                    chunk = self._create_text_chunk(
                        chunk_id=chunk_id,
                        content=current_content,
                        title_prefix=title_prefix,
                        document=document,
                        item=item,
                        config=config,
                    )
                    chunks.append(chunk)
                    chunk_id += 1

                # Start new chunk with current paragraph
                if count_tokens(para) > config.chunk_token_limit:
                    # Paragraph itself is too long, truncate
                    para = truncate_to_tokens(para, config.chunk_token_limit)
                current_content = para
            else:
                current_content = test_content

        # Add final chunk
        if current_content:
            chunk = self._create_text_chunk(
                chunk_id=chunk_id,
                content=current_content,
                title_prefix=title_prefix,
                document=document,
                item=item,
                config=config,
            )
            chunks.append(chunk)

        return chunks

    def _create_text_chunk(
        self,
        chunk_id: int,
        content: str,
        title_prefix: str,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> Chunk:
        """
        Create a text chunk.

        Args:
            chunk_id: Chunk ID
            content: Text content
            title_prefix: Title prefix
            document: Parsed document
            item: Ingestion item
            config: Chunking configuration

        Returns:
            Chunk instance
        """
        # Build metadata
        chunk_metadata = dict(document.metadata or {})
        chunk_metadata["chunk_type"] = "text"

        # Create blurb
        blurb = content[:config.blurb_size * 4]

        return Chunk(
            document_id=item.document_id,
            chunk_id=chunk_id,
            content=content,
            title=item.file_name,
            source_type=str(item.source_type.value) if hasattr(item.source_type, "value") else item.source_type,
            link=item.source_uri,
            metadata=chunk_metadata,
            blurb=blurb,
            semantic_identifier=title_prefix.strip(),
            # Internal fields
            _title_prefix=title_prefix,
        )
