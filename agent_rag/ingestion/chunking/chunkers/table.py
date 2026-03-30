"""Table-aware chunker for spreadsheet and tabular data files."""

import logging
import re
from typing import Any, Optional

from agent_rag.core.models import Chunk
from agent_rag.ingestion.chunking.base import (
    BaseChunker,
    ChunkCandidate,
    count_tokens,
    truncate_to_tokens,
)
from agent_rag.ingestion.chunking.config import ChunkingConfig
from agent_rag.ingestion.models import IngestionItem
from agent_rag.ingestion.parsing.base import ParsedDocument

logger = logging.getLogger(__name__)


# MIME types for table/spreadsheet files
TABLE_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # xlsx
    "application/vnd.ms-excel",  # xls
    "text/csv",
    "text/tab-separated-values",
    "application/vnd.oasis.opendocument.spreadsheet",  # ods
}

# File extensions for table files
TABLE_EXTENSIONS = {
    ".xlsx",
    ".xls",
    ".csv",
    ".tsv",
    ".ods",
}


class TableChunker(BaseChunker):
    """
    Table-aware chunker for spreadsheet and tabular data.

    Features:
    - Sheet-level chunking for multi-sheet workbooks
    - Row-based chunking within sheets
    - Preserves table headers with each chunk
    - Handles column metadata and data types
    - Markdown table formatting for readability

    Supported formats: XLSX, XLS, CSV, TSV, ODS
    """

    @property
    def priority(self) -> int:
        """High priority for table files."""
        return 50

    @property
    def name(self) -> str:
        """Chunker name."""
        return "TableChunker"

    def supports(self, source_type: str, mime_type: str, document: ParsedDocument) -> bool:
        """
        Check if this chunker supports the document.

        Args:
            source_type: Source type
            mime_type: MIME type
            document: Parsed document

        Returns:
            True if document contains tabular data
        """
        # Check MIME type
        if mime_type in TABLE_MIME_TYPES:
            return True

        # Check file extension from metadata
        filename = document.metadata.get("filename", "") or document.metadata.get("file_name", "")
        if filename:
            for ext in TABLE_EXTENSIONS:
                if filename.lower().endswith(ext):
                    return True

        # Check if document has tables extracted
        if document.tables and len(document.tables) > 0:
            return True

        # Check for markdown table patterns in text
        if document.text:
            # Match markdown table pattern: | header | header |
            if re.search(r"^\|[^|]+\|[^|]+\|", document.text, re.MULTILINE):
                return True
            # Match separator line: |---|---|
            if re.search(r"^\|[-:]+\|[-:]+\|", document.text, re.MULTILINE):
                return True

        return False

    def chunk(
        self,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """
        Chunk tabular data into semantic units.

        Args:
            document: Parsed document containing table data
            item: Ingestion item
            config: Chunking configuration

        Returns:
            List of chunks representing table sections
        """
        chunks = []
        chunk_id = 0

        # Build title prefix
        title = document.metadata.get("title") or item.file_name
        title_prefix = f"{title}\n\n" if title else ""

        # Process extracted tables from document
        if document.tables:
            for table_idx, table in enumerate(document.tables):
                table_chunks = self._chunk_table(
                    table=table,
                    table_idx=table_idx,
                    start_chunk_id=chunk_id,
                    title_prefix=title_prefix,
                    document=document,
                    item=item,
                    config=config,
                )
                chunks.extend(table_chunks)
                chunk_id += len(table_chunks)

        # Process markdown tables in text
        if document.text:
            text_table_chunks = self._chunk_markdown_tables(
                text=document.text,
                start_chunk_id=chunk_id,
                title_prefix=title_prefix,
                document=document,
                item=item,
                config=config,
            )
            chunks.extend(text_table_chunks)
            chunk_id += len(text_table_chunks)

        logger.info(
            f"TableChunker generated {len(chunks)} chunks from "
            f"{len(document.tables or [])} tables"
        )

        return chunks

    def _chunk_table(
        self,
        table: dict[str, Any],
        table_idx: int,
        start_chunk_id: int,
        title_prefix: str,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """
        Chunk a single table into row-based chunks.

        Args:
            table: Table dict with 'headers', 'rows', 'name', etc.
            table_idx: Index of the table
            start_chunk_id: Starting chunk ID
            title_prefix: Title prefix
            document: Parsed document
            item: Ingestion item
            config: Chunking configuration

        Returns:
            List of chunks for this table
        """
        chunks = []

        # Extract table structure
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        table_name = table.get("name", f"Table_{table_idx + 1}")
        sheet_name = table.get("sheet", "")

        if not rows:
            return chunks

        # Build header row for markdown
        header_text = self._format_markdown_row(headers) if headers else ""
        separator = self._format_separator_row(len(headers)) if headers else ""
        header_block = f"{header_text}\n{separator}" if header_text else ""
        header_tokens = count_tokens(header_block)

        # Calculate rows per chunk
        available_tokens = config.chunk_token_limit - header_tokens - count_tokens(title_prefix)
        sample_row = self._format_markdown_row(rows[0]) if rows else ""
        tokens_per_row = max(count_tokens(sample_row), 10)
        rows_per_chunk = max(5, int(available_tokens / tokens_per_row))

        # Build chunks with rows
        chunk_id = start_chunk_id

        for i in range(0, len(rows), rows_per_chunk):
            chunk_rows = rows[i : i + rows_per_chunk]

            # Format rows as markdown table
            row_texts = [self._format_markdown_row(row) for row in chunk_rows]
            table_content = header_block + "\n" + "\n".join(row_texts) if header_block else "\n".join(row_texts)

            # Create chunk metadata
            chunk_metadata = dict(document.metadata or {})
            chunk_metadata.update({
                "table_name": table_name,
                "sheet_name": sheet_name,
                "table_index": table_idx,
                "row_start": i,
                "row_end": i + len(chunk_rows),
                "total_rows": len(rows),
                "columns": headers,
            })

            # Truncate if still too long
            table_content = truncate_to_tokens(table_content, config.chunk_token_limit)

            chunk = Chunk(
                document_id=item.document_id,
                chunk_id=chunk_id,
                content=table_content,
                title=item.file_name,
                source_type=str(item.source_type.value) if hasattr(item.source_type, "value") else item.source_type,
                link=item.source_uri,
                metadata=chunk_metadata,
                blurb=self._create_table_blurb(table_name, sheet_name, headers, i, len(chunk_rows)),
                semantic_identifier=title_prefix.strip(),
                section_continuation=(i > 0),
                _title_prefix=title_prefix,
                _metadata_suffix_semantic=f"\nTable: {table_name}\nSheet: {sheet_name}\nColumns: {', '.join(headers)}",
                _metadata_suffix_keyword=f"{table_name} {sheet_name} {' '.join(headers)}",
            )
            chunks.append(chunk)
            chunk_id += 1

        return chunks

    def _chunk_markdown_tables(
        self,
        text: str,
        start_chunk_id: int,
        title_prefix: str,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """
        Extract and chunk markdown tables from text.

        Args:
            text: Document text containing markdown tables
            start_chunk_id: Starting chunk ID
            title_prefix: Title prefix
            document: Parsed document
            item: Ingestion item
            config: Chunking configuration

        Returns:
            List of chunks from markdown tables
        """
        chunks = []

        # Find markdown table blocks
        table_pattern = r"(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)"
        tables = re.findall(table_pattern, text)

        chunk_id = start_chunk_id

        for table_idx, table_text in enumerate(tables):
            # Check if table fits in one chunk
            table_tokens = count_tokens(table_text)

            if table_tokens <= config.chunk_token_limit:
                chunk = self._create_markdown_table_chunk(
                    chunk_id=chunk_id,
                    content=table_text,
                    table_idx=table_idx,
                    title_prefix=title_prefix,
                    document=document,
                    item=item,
                    config=config,
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # Split large markdown table
                sub_chunks = self._split_markdown_table(
                    table_text=table_text,
                    table_idx=table_idx,
                    start_chunk_id=chunk_id,
                    title_prefix=title_prefix,
                    document=document,
                    item=item,
                    config=config,
                )
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)

        return chunks

    def _split_markdown_table(
        self,
        table_text: str,
        table_idx: int,
        start_chunk_id: int,
        title_prefix: str,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """
        Split a large markdown table into smaller chunks.

        Args:
            table_text: Full markdown table text
            table_idx: Table index
            start_chunk_id: Starting chunk ID
            title_prefix: Title prefix
            document: Parsed document
            item: Ingestion item
            config: Chunking configuration

        Returns:
            List of chunks from split table
        """
        chunks = []
        lines = table_text.strip().split("\n")

        if len(lines) < 3:
            return chunks

        # Extract header and separator
        header_line = lines[0]
        separator_line = lines[1]
        data_lines = lines[2:]

        header_block = f"{header_line}\n{separator_line}"
        header_tokens = count_tokens(header_block)

        # Calculate lines per chunk
        available_tokens = config.chunk_token_limit - header_tokens - count_tokens(title_prefix)
        sample_line = data_lines[0] if data_lines else "|cell|"
        tokens_per_line = max(count_tokens(sample_line), 5)
        lines_per_chunk = max(3, int(available_tokens / tokens_per_line))

        chunk_id = start_chunk_id

        for i in range(0, len(data_lines), lines_per_chunk):
            chunk_lines = data_lines[i : i + lines_per_chunk]
            chunk_content = f"{header_block}\n" + "\n".join(chunk_lines)

            chunk = self._create_markdown_table_chunk(
                chunk_id=chunk_id,
                content=chunk_content,
                table_idx=table_idx,
                title_prefix=title_prefix,
                document=document,
                item=item,
                config=config,
                is_continuation=(i > 0),
            )
            chunks.append(chunk)
            chunk_id += 1

        return chunks

    def _create_markdown_table_chunk(
        self,
        chunk_id: int,
        content: str,
        table_idx: int,
        title_prefix: str,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
        is_continuation: bool = False,
    ) -> Chunk:
        """
        Create a chunk from markdown table content.

        Args:
            chunk_id: Chunk ID
            content: Table content
            table_idx: Table index
            title_prefix: Title prefix
            document: Parsed document
            item: Ingestion item
            config: Chunking configuration
            is_continuation: Whether this is a continuation

        Returns:
            Chunk instance
        """
        # Extract headers from content
        lines = content.split("\n")
        headers = []
        if lines and lines[0].startswith("|"):
            headers = [h.strip() for h in lines[0].split("|")[1:-1]]

        # Build metadata
        chunk_metadata = dict(document.metadata or {})
        chunk_metadata.update({
            "table_index": table_idx,
            "columns": headers,
            "is_markdown_table": True,
        })

        # Create blurb
        blurb = f"Table {table_idx + 1}: " + ", ".join(headers[:5])
        if len(headers) > 5:
            blurb += f" (+{len(headers) - 5} more columns)"

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
            section_continuation=is_continuation,
            _title_prefix=title_prefix,
            _metadata_suffix_semantic=f"\nTable {table_idx + 1}\nColumns: {', '.join(headers)}",
            _metadata_suffix_keyword=f"table {' '.join(headers)}",
        )

    def _format_markdown_row(self, row: list[Any]) -> str:
        """
        Format a row as markdown table row.

        Args:
            row: List of cell values

        Returns:
            Markdown formatted row
        """
        cells = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        return "| " + " | ".join(cells) + " |"

    def _format_separator_row(self, num_cols: int) -> str:
        """
        Format markdown table separator row.

        Args:
            num_cols: Number of columns

        Returns:
            Separator row
        """
        return "| " + " | ".join(["---"] * max(num_cols, 1)) + " |"

    def _create_table_blurb(
        self,
        table_name: str,
        sheet_name: str,
        headers: list[str],
        row_start: int,
        row_count: int,
    ) -> str:
        """
        Create a descriptive blurb for table chunk.

        Args:
            table_name: Name of the table
            sheet_name: Name of the sheet
            headers: Column headers
            row_start: Starting row index
            row_count: Number of rows in chunk

        Returns:
            Descriptive blurb
        """
        parts = []

        if table_name:
            parts.append(table_name)
        if sheet_name:
            parts.append(f"Sheet: {sheet_name}")

        parts.append(f"Rows {row_start + 1}-{row_start + row_count}")

        if headers:
            header_preview = ", ".join(headers[:3])
            if len(headers) > 3:
                header_preview += f" (+{len(headers) - 3} more)"
            parts.append(f"Columns: {header_preview}")

        return " | ".join(parts)
