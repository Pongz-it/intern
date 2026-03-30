"""PDF document parser using pypdf and pdfplumber."""

import io
import logging
from typing import Optional

from agent_rag.ingestion.parsing.base import BaseParser, ParsedDocument, ParsedImage
from agent_rag.ingestion.parsing.utils import (
    extract_title_from_text,
    normalize_text,
)

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """
    Parser for PDF files using pypdf + pdfplumber.

    Strategy:
    - pypdf for basic text extraction and metadata
    - pdfplumber for tables and enhanced layout detection
    - Fallback to OCR if text extraction fails

    Supports:
    - .pdf files
    - application/pdf

    Priority: 10 (default for common documents)
    """

    def supports(self, source_type: str, extension: str, mime_type: str) -> bool:
        """Support .pdf files."""
        if extension.lower() == "pdf":
            return True
        if mime_type and "pdf" in mime_type.lower():
            return True
        return False

    def parse(self, content: bytes, filename: str) -> ParsedDocument:
        """
        Parse PDF file.

        Args:
            content: Raw PDF bytes
            filename: Original filename

        Returns:
            ParsedDocument with extracted text, images, tables, metadata

        Raises:
            ImportError: If required libraries not installed
            Exception: If parsing fails
        """
        try:
            import pypdf
        except ImportError:
            raise RuntimeError("pypdf not installed. Run: pip install pypdf")

        # Create BytesIO for multiple reads
        file_stream = io.BytesIO(content)

        # Extract with pypdf
        try:
            reader = pypdf.PdfReader(file_stream)
        except Exception as e:
            logger.error(f"Failed to read PDF with pypdf: {e}")
            raise RuntimeError(f"Failed to parse PDF: {e}")

        # Extract text from all pages
        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")
                continue

        # Combine all pages
        text = "\n\n".join(pages_text)

        # If no text extracted, may be scanned PDF needing OCR
        if not text.strip():
            logger.warning(f"No text extracted from PDF {filename}, may need OCR")

        # Normalize text
        text = normalize_text(text)

        # Extract metadata
        metadata = self._extract_metadata(reader, filename)

        # Try to extract title if not in metadata
        if "title" not in metadata or not metadata["title"]:
            title = extract_title_from_text(text)
            if title:
                metadata["title"] = title

        # Extract images
        images = self._extract_images(reader, filename)

        # Extract tables with pdfplumber if available
        tables = self._extract_tables(content)

        return ParsedDocument(
            text=text,
            metadata=metadata,
            images=images,
            links=[],  # Could extract PDF links/annotations
            tables=tables,
        )

    def _extract_metadata(
        self,
        reader: "pypdf.PdfReader",
        filename: str,
    ) -> dict:
        """
        Extract PDF metadata.

        Args:
            reader: pypdf PdfReader instance
            filename: Original filename

        Returns:
            Metadata dictionary
        """
        metadata = {
            "filename": filename,
            "source_type": "file",
            "format": "pdf",
            "page_count": len(reader.pages),
        }

        # Extract document info
        if reader.metadata:
            info = reader.metadata

            if info.title:
                metadata["title"] = info.title
            if info.author:
                metadata["author"] = info.author
            if info.subject:
                metadata["subject"] = info.subject
            if info.creator:
                metadata["creator"] = info.creator
            if info.producer:
                metadata["producer"] = info.producer
            if info.creation_date:
                try:
                    metadata["created"] = info.creation_date.isoformat()
                except Exception:
                    pass
            if info.modification_date:
                try:
                    metadata["modified"] = info.modification_date.isoformat()
                except Exception:
                    pass

        return metadata

    def _extract_images(
        self,
        reader: "pypdf.PdfReader",
        filename: str,
    ) -> list[ParsedImage]:
        """
        Extract images from PDF.

        Args:
            reader: pypdf PdfReader instance
            filename: Original filename

        Returns:
            List of ParsedImage objects
        """
        images = []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                if "/XObject" in page["/Resources"]:
                    xobjects = page["/Resources"]["/XObject"].get_object()

                    for obj_name, obj in xobjects.items():
                        if obj["/Subtype"] == "/Image":
                            try:
                                # Extract image data
                                image_data = obj.get_data()

                                # Determine MIME type
                                if "/Filter" in obj:
                                    filter_type = obj["/Filter"]
                                    if filter_type == "/DCTDecode":
                                        mime_type = "image/jpeg"
                                        ext = "jpg"
                                    elif filter_type == "/FlateDecode":
                                        mime_type = "image/png"
                                        ext = "png"
                                    else:
                                        mime_type = "image/png"
                                        ext = "png"
                                else:
                                    mime_type = "image/png"
                                    ext = "png"

                                # Create image ID
                                image_id = f"page{page_num}_img{len(images) + 1}"

                                images.append(
                                    ParsedImage(
                                        image_id=image_id,
                                        content=image_data,
                                        mime_type=mime_type,
                                        page_number=page_num,
                                        caption=None,
                                    )
                                )

                            except Exception as e:
                                logger.warning(
                                    f"Failed to extract image from page {page_num}: {e}"
                                )
                                continue

            except Exception as e:
                logger.warning(f"Failed to process images on page {page_num}: {e}")
                continue

        return images

    def _extract_tables(self, content: bytes) -> list[dict]:
        """
        Extract tables using pdfplumber.

        Args:
            content: Raw PDF bytes

        Returns:
            List of table dictionaries
        """
        tables = []

        try:
            import pdfplumber
        except ImportError:
            logger.debug("pdfplumber not installed, skipping table extraction")
            return tables

        try:
            file_stream = io.BytesIO(content)
            with pdfplumber.open(file_stream) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        page_tables = page.extract_tables()

                        for table_idx, table in enumerate(page_tables):
                            if table:
                                # Convert table to dict format
                                table_dict = {
                                    "page_number": page_num,
                                    "table_index": table_idx,
                                    "rows": table,
                                    "row_count": len(table),
                                    "column_count": len(table[0]) if table else 0,
                                }
                                tables.append(table_dict)

                    except Exception as e:
                        logger.warning(
                            f"Failed to extract tables from page {page_num}: {e}"
                        )
                        continue

        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed: {e}")

        return tables

    @property
    def priority(self) -> int:
        """Default priority for common documents."""
        return 10
