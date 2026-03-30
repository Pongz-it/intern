# Ingestion and Indexing Design (Agent RAG)

## Goals

- Add a standalone ingestion module to `agent_rag` that can parse and index content from files and URLs.
- Support at least: Word (with images), PDF (with images), plain text, Markdown, and web URLs.
- Provide a unified ingestion API that persists inputs to Postgres and triggers asynchronous indexing.
- Store raw and derived artifacts in MinIO.
- Implement asynchronous indexing with Hatchet workflows (not Celery).
- Keep parsing and indexing capabilities at least as strong as Onyx.
- Ensure indexed data aligns with `agent_rag.core.models.Chunk` and existing retrieval flows.
- Provide extension points for parsers, OCR, summarization, and embeddings.

## Non-goals

- Replace existing connectors or retrieval logic in `agent_rag`.
- Build full connector parity with Onyx connectors.
- Introduce a new vector store; reuse existing `DocumentIndex` implementations.

## Current Baseline (Onyx)

Onyx provides:

- File parsing: `backend/onyx/file_processing/extract_file_text.py` (docx, pdf, pptx, xlsx, eml, epub, msg, text)
- HTML parsing: `backend/onyx/file_processing/html_utils.py` (trafilatura + bs4 fallback)
- Extraction fallback via Unstructured API if configured
- Indexing pipeline: `backend/onyx/indexing/indexing_pipeline.py` (chunk -> embed -> index -> DB update)
- Chunking: `backend/onyx/indexing/chunker.py` (SentenceChunker with configurable token limits)
- Async orchestration: Celery docfetching/docprocessing

The new module must not be weaker than these capabilities.

## Proposed Architecture

### Module Layout

```
agent_rag/ingestion/
  __init__.py
  api.py                    # FastAPI endpoints
  models.py                 # Pydantic models & DB models
  storage.py                # MinIO storage adapter
  dedup.py                  # Content deduplication logic
  parsing/
    __init__.py
    base.py                 # Parser interface
    registry.py             # Parser registration & priority
    parsers/
      __init__.py
      docx.py               # Word documents (markitdown)
      pdf.py                # PDF (pypdf + pdfplumber for tables)
      pptx.py               # PowerPoint (pptx2md)        [P0]
      xlsx.py               # Excel (openpyxl)            [P0]
      text.py               # Plain text
      markdown.py           # Markdown files
      html.py               # HTML/URL (trafilatura + bs4)
      eml.py                # Email files                 [P1]
      epub.py               # EPUB ebooks                 [P1]
      msg.py                # Outlook MSG files           [P1]
      archive.py            # ZIP/TAR archives            [P1]
  ocr/
    __init__.py
    base.py                 # OCR provider interface
    providers/
      __init__.py
      llm_ocr.py            # LLM-based OCR
  indexing/
    __init__.py
    workflow.py             # Hatchet workflow definition
    tasks.py                # Individual task implementations
  chunking/
    __init__.py
    base.py                 # BaseChunker interface          [P1]
    registry.py             # ChunkerRegistry                [P1]
    config.py               # ChunkingConfig dataclass
    chunker.py              # SemanticChunker (default)
    chunkers/               # Specialized chunkers           [P2]
      __init__.py
      code.py               # CodeChunker (syntax-aware)
      slide.py              # SlideChunker (pptx)
      table.py              # TableChunker (xlsx/csv)
      email.py              # EmailChunker (eml/msg)
  embeddings/
    __init__.py
    config.py               # EmbeddingConfig dataclass       [P1]
    models.py               # ChunkEmbedding, IndexChunk      [P1]
    embedder.py             # DefaultIndexingEmbedder         [P1]
    failure_handler.py      # embed_chunks_with_failure_handling [P1]
  callbacks/
    __init__.py
    webhook.py              # Webhook notification         [P2]
    progress.py             # Progress tracking            [P2]
```

### High-level Flow

1) Client calls `POST /ingest` with file/url/text.
2) API computes `content_hash` and checks for duplicates.
3) If not duplicate: stores raw content in MinIO, writes row in `ingestion_items` (status=PENDING).
4) Hatchet workflow `content-indexing` is started with `item_id`.
5) Workflow loads input, parses content, extracts images, optionally OCRs images, chunks, embeds, indexes, and updates status.
6) Optional: sends webhook notification on completion/failure.

## Storage Design

### MinIO

- Bucket: `agent-rag-ingestion` (configurable)
- Keys:
  - `raw/{tenant_id}/{item_id}/{filename}`
  - `parsed/{tenant_id}/{item_id}/text.md`
  - `images/{tenant_id}/{item_id}/{image_id}.{ext}`
  - `ocr/{tenant_id}/{item_id}/{image_id}.json`
  - `derived/{tenant_id}/{item_id}/doc_summary.txt`
  - `derived/{tenant_id}/{item_id}/chunk_context.json`

### Postgres: `ingestion_items`

Required fields:

- `id` (uuid, primary key)
- `tenant_id` (varchar, indexed)
- `source_type` (enum: file|url|text|markdown)
- `source_uri` (text, url or original path)
- `file_name` (varchar)
- `mime_type` (varchar)
- `size_bytes` (bigint)
- `content_hash` (varchar(64), indexed, for deduplication)
- `content_ref` (text, MinIO key to raw)
- `parsed_ref` (text, MinIO key to parsed text)
- `status` (enum: PENDING|PROCESSING|INDEXED|FAILED|DUPLICATE)
- `error` (text)
- `retry_count` (int, default 0)
- `last_attempt_at` (timestamp)
- `document_id` (varchar, used by index, indexed)
- `chunk_count` (int)
- `metadata_json` (jsonb)
- `webhook_url` (text, optional callback URL)
- `created_at` (timestamp)
- `updated_at` (timestamp)

Indexes:
- `idx_ingestion_items_tenant_status` on (tenant_id, status)
- `idx_ingestion_items_content_hash` on (tenant_id, content_hash)
- `idx_ingestion_items_document_id` on (document_id)

Optional tables:

- `ingestion_images` (image-level OCR status and links)
- `ingestion_batches` (for batch import tracking)

## Content Deduplication [P1]

### Deduplication Strategy

1) Compute `content_hash` (SHA-256) from normalized content bytes plus `source_type`.
2) Deduplication is tenant-scoped by default: key is `(tenant_id, content_hash)`.
3) If found and status is INDEXED:
   - Return existing `document_id` with status=DUPLICATE.
   - Skip workflow trigger unless `force_reindex=true`.
4) If found and status is PROCESSING:
   - Return existing `item_id` and do not start a new workflow.
5) If found and status is FAILED:
   - Create a new attempt only if `AGENT_RAG_DEDUP_REPROCESS_FAILED=true`.

### Explicit document_id

- If the caller supplies `document_id`, it overrides hash-based IDs.
- If the supplied `document_id` already exists with a different `content_hash`, treat it as an update:
  - Delete existing chunks for that `document_id` first.
  - Index new content and update stored hash and metadata.

### Cross-tenant behavior

- No cross-tenant dedup by default.
- Optional global dedup is allowed only behind a feature flag and must keep index isolation by tenant.

### Dedup Configuration

```python
AGENT_RAG_DEDUP_ENABLED = True           # Enable/disable deduplication
AGENT_RAG_DEDUP_REPROCESS_FAILED = True  # Allow reprocessing failed items
AGENT_RAG_DEDUP_CROSS_TENANT = False     # Optional global dedup
```

## Parsing Design

### Parser Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional, Any

@dataclass
class ParsedImage:
    image_id: str
    content: bytes
    mime_type: str
    page_number: Optional[int] = None
    caption: Optional[str] = None

@dataclass
class ParsedDocument:
    text: str                              # Normalized text content
    images: list[ParsedImage]              # Extracted images
    metadata: dict[str, Any]               # title, author, timestamps, etc.
    links: list[str]                       # Extracted URLs
    tables: list[dict[str, Any]]           # Extracted tables (for xlsx/pdf)

class BaseParser(ABC):
    @abstractmethod
    def supports(self, source_type: str, extension: str, mime_type: str) -> bool:
        """Check if parser supports this file type."""
        pass

    @abstractmethod
    def parse(self, content: bytes, filename: str) -> ParsedDocument:
        """Parse content and return structured document."""
        pass

    @property
    def priority(self) -> int:
        """Higher priority parsers are tried first. Default: 0."""
        return 0
```

### ParsedDocument normalization rules

Required fields:
- `text`
- `metadata`

Optional fields:
- `images`, `links`, `tables`

Normalization:
- Strip control characters except `\\n` and `\\t`.
- Collapse repeated whitespace and excessive blank lines.
- Enforce maximum length by tokens or chars with head+tail truncation.
- Record original length in `metadata` (e.g., `original_char_count`, `original_token_count`).
- Preserve source hints in `metadata` (`title`, `source_url`, `author`).

### Default Parsers (Onyx Parity + Extensions)

| Format | Parser | Library | Priority | Status |
|--------|--------|---------|----------|--------|
| .docx | DocxParser | markitdown + zipfile | 0 | Core |
| .pdf | PdfParser | pypdf + pdfplumber | 0 | Core |
| .pptx | PptxParser | pptx2md | 0 | **P0** |
| .xlsx | XlsxParser | openpyxl | 0 | **P0** |
| .txt | TextParser | chardet + read | 0 | Core |
| .md | MarkdownParser | direct read | 0 | Core |
| .html | HtmlParser | trafilatura + bs4 | 0 | Core |
| .eml | EmlParser | email.parser | 0 | **P1** |
| .epub | EpubParser | ebooklib | 0 | **P1** |
| .msg | MsgParser | extract-msg | 0 | **P1** |
| .zip/.tar | ArchiveParser | zipfile/tarfile | -1 | **P1** |
| * | UnstructuredParser | unstructured-api | -10 | Fallback |

### Parser Implementations

#### PptxParser [P0]

```python
class PptxParser(BaseParser):
    """PowerPoint parser using pptx2md."""

    def supports(self, source_type: str, extension: str, mime_type: str) -> bool:
        return extension in ['.pptx', '.ppt'] or \
               mime_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation'

    def parse(self, content: bytes, filename: str) -> ParsedDocument:
        # Use pptx2md for text extraction
        # Extract images from pptx zip structure
        # Extract speaker notes as additional content
        pass
```

#### XlsxParser [P0]

```python
class XlsxParser(BaseParser):
    """Excel parser using openpyxl."""

    def supports(self, source_type: str, extension: str, mime_type: str) -> bool:
        return extension in ['.xlsx', '.xls'] or \
               mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            'application/vnd.ms-excel']

    def parse(self, content: bytes, filename: str) -> ParsedDocument:
        # Parse each sheet as separate section
        # Convert tables to markdown format
        # Extract cell values with structure preservation
        pass
```

#### PdfParser (Enhanced) [P2]

```python
class PdfParser(BaseParser):
    """PDF parser with table extraction support."""

    def parse(self, content: bytes, filename: str) -> ParsedDocument:
        # Primary: pypdf for text extraction
        text = self._extract_with_pypdf(content)

        # Enhanced: pdfplumber for table extraction
        if self.enable_table_extraction:
            tables = self._extract_tables_with_pdfplumber(content)

        # Fallback: Unstructured API for complex PDFs
        if not text.strip() and self.unstructured_enabled:
            return self._parse_with_unstructured(content)

        return ParsedDocument(text=text, tables=tables, ...)
```

### Unstructured Fallback

- If `UNSTRUCTURED_API_KEY` is configured, attempt Unstructured parsing for complex files.
- Fallback to local parsers on API failure or timeout.
- Log which files required fallback for monitoring.

### Extensibility

- `ParserRegistry` allows additional parser plugins and custom priority.
- Custom parsers can be registered at runtime.

```python
from agent_rag.ingestion.parsing import parser_registry

@parser_registry.register(priority=10)
class CustomParser(BaseParser):
    def supports(self, source_type, extension, mime_type) -> bool:
        return extension == '.custom'

    def parse(self, content: bytes, filename: str) -> ParsedDocument:
        # Custom parsing logic
        pass
```

## Chunking Design [P0/P1/P2]

Based on analysis of Onyx's `backend/onyx/indexing/chunker.py`, implement comprehensive chunking with full feature parity and extensibility.

### Chunking Configuration

```python
from dataclasses import dataclass
from typing import Optional
from chonkie import SentenceChunker

@dataclass
class ChunkingConfig:
    """Chunking configuration parameters."""
    # Core settings
    chunk_token_limit: int = 512          # Max tokens per chunk
    chunk_overlap: int = 0                 # Token overlap between chunks
    blurb_size: int = 128                  # Size of excerpt for display
    mini_chunk_size: int = 64              # For multipass embedding
    large_chunk_ratio: int = 4             # Combine N chunks for large chunks
    max_metadata_percentage: float = 0.25  # Max metadata portion of chunk
    chunk_min_content: int = 256           # Min content tokens after metadata
    strict_chunk_token_limit: bool = True  # Enforce strict token limits [P1]

    # Feature flags
    enable_multipass: bool = False         # Enable mini-chunk embeddings [P0]
    enable_large_chunks: bool = False      # Enable combined large chunks [P0]
    enable_contextual_rag: bool = False    # Enable doc_summary/chunk_context [P1]
    include_metadata: bool = True          # Include metadata in chunk

    # Contextual RAG settings [P1]
    use_doc_summary: bool = True           # Include document summary in chunks
    use_chunk_context: bool = True         # Include chunk context
    max_context_tokens: int = 512          # Max tokens for contextual RAG

    # Section separators
    section_separator: str = "\n\n---\n\n"
    return_separator: str = "\n\n"
```

### Chunker Interface and Registry [P1]

```python
from abc import ABC, abstractmethod
from typing import Optional

class BaseChunker(ABC):
    """Base chunker interface for extensibility."""

    @abstractmethod
    def supports(self, source_type: str, mime_type: str) -> bool:
        """Check if chunker supports this document type."""
        pass

    @abstractmethod
    def chunk(
        self,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """Chunk document into pieces."""
        pass

    @property
    def priority(self) -> int:
        """Higher priority chunkers are tried first. Default: 0."""
        return 0


class ChunkerRegistry:
    """Registry for document-type-specific chunkers."""

    def __init__(self):
        self._chunkers: list[BaseChunker] = []
        self._default_chunker: Optional[BaseChunker] = None

    def register(self, chunker: BaseChunker) -> None:
        """Register a chunker with priority ordering."""
        self._chunkers.append(chunker)
        self._chunkers.sort(key=lambda c: c.priority, reverse=True)

    def set_default(self, chunker: BaseChunker) -> None:
        """Set default chunker for unmatched types."""
        self._default_chunker = chunker

    def get_chunker(self, source_type: str, mime_type: str) -> BaseChunker:
        """Get appropriate chunker for document type."""
        for chunker in self._chunkers:
            if chunker.supports(source_type, mime_type):
                return chunker
        if self._default_chunker:
            return self._default_chunker
        raise ValueError(f"No chunker for {source_type}/{mime_type}")


# Global registry instance
chunker_registry = ChunkerRegistry()
```

### Dual Metadata Suffix [P1]

```python
def _get_metadata_suffix(
    metadata: dict[str, str | list[str]],
    include_separator: bool = False,
) -> tuple[str, str]:
    """
    Generate dual metadata suffixes for hybrid search optimization.

    Returns:
        tuple: (metadata_suffix_semantic, metadata_suffix_keyword)
        - semantic: Natural language with keys for embedding
        - keyword: Values only for keyword/BM25 search
    """
    if not metadata:
        return "", ""

    METADATA_KEYS_TO_IGNORE = {"internal_id", "access_control", "permissions"}

    metadata_str = "Metadata:\n"
    metadata_values = []

    for key, value in metadata.items():
        if key in METADATA_KEYS_TO_IGNORE:
            continue

        value_str = ", ".join(value) if isinstance(value, list) else value

        if isinstance(value, list):
            metadata_values.extend(value)
        else:
            metadata_values.append(value)

        metadata_str += f"\t{key} - {value_str}\n"

    metadata_semantic = metadata_str.strip()
    metadata_keyword = " ".join(metadata_values)

    if include_separator:
        sep = "\n\n"
        return sep + metadata_semantic, sep + metadata_keyword

    return metadata_semantic, metadata_keyword
```

### Semantic Chunker Implementation [P0]

```python
from typing import cast

SECTION_SEPARATOR = "\n\n---\n\n"
RETURN_SEPARATOR = "\n\n"


class SemanticChunker(BaseChunker):
    """Semantic-aware document chunker with full Onyx parity."""

    def __init__(self, config: ChunkingConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # Token counter function
        def token_counter(text: str) -> int:
            return len(tokenizer.encode(text))

        # Sentence-based chunking for semantic coherence
        self.chunk_splitter = SentenceChunker(
            tokenizer_or_token_counter=token_counter,
            chunk_size=config.chunk_token_limit,
            chunk_overlap=config.chunk_overlap,
            return_type="texts",
        )

        self.blurb_splitter = SentenceChunker(
            tokenizer_or_token_counter=token_counter,
            chunk_size=config.blurb_size,
            chunk_overlap=0,
            return_type="texts",
        )

        # Mini-chunk splitter for multipass mode [P0]
        self.mini_chunk_splitter = (
            SentenceChunker(
                tokenizer_or_token_counter=token_counter,
                chunk_size=config.mini_chunk_size,
                chunk_overlap=0,
                return_type="texts",
            )
            if config.enable_multipass
            else None
        )

    def supports(self, source_type: str, mime_type: str) -> bool:
        """Default chunker supports all types."""
        return True

    @property
    def priority(self) -> int:
        return -100  # Lowest priority, used as fallback

    def chunk(
        self,
        document: ParsedDocument,
        item: IngestionItem,
        config: Optional[ChunkingConfig] = None,
    ) -> list[Chunk]:
        """Split document into chunks with full feature support."""
        cfg = config or self.config
        chunks: list[Chunk] = []

        # Prepare title prefix
        title = self._extract_blurb(
            document.metadata.get("title", item.file_name) or ""
        )
        title_prefix = title + RETURN_SEPARATOR if title else ""
        title_tokens = len(self.tokenizer.encode(title_prefix))

        # Prepare dual metadata suffixes [P1]
        metadata_suffix_semantic = ""
        metadata_suffix_keyword = ""
        metadata_tokens = 0

        if cfg.include_metadata:
            metadata_suffix_semantic, metadata_suffix_keyword = (
                _get_metadata_suffix(document.metadata, include_separator=True)
            )
            metadata_tokens = len(self.tokenizer.encode(metadata_suffix_semantic))

        # Skip metadata if too large
        if metadata_tokens >= cfg.chunk_token_limit * cfg.max_metadata_percentage:
            metadata_suffix_semantic = ""
            metadata_suffix_keyword = ""
            metadata_tokens = 0

        # Calculate contextual RAG token reservation [P1]
        context_size = 0
        if cfg.enable_contextual_rag:
            doc_content = document.text
            doc_token_count = len(self.tokenizer.encode(doc_content))
            single_chunk_fits = (
                doc_token_count + title_tokens + metadata_tokens
                <= cfg.chunk_token_limit
            )

            if not single_chunk_fits:
                context_size = cfg.max_context_tokens * (
                    int(cfg.use_chunk_context) + int(cfg.use_doc_summary)
                )

        # Calculate available content space
        content_token_limit = (
            cfg.chunk_token_limit - title_tokens - metadata_tokens - context_size
        )

        # Fallback if not enough space for content
        if content_token_limit <= cfg.chunk_min_content:
            context_size = 0
            content_token_limit = cfg.chunk_token_limit - title_tokens - metadata_tokens

        if content_token_limit <= cfg.chunk_min_content:
            content_token_limit = cfg.chunk_token_limit
            title_prefix = ""
            metadata_suffix_semantic = ""

        # Process sections including images [P0]
        chunks = self._chunk_with_sections(
            document=document,
            item=item,
            title_prefix=title_prefix,
            metadata_suffix_semantic=metadata_suffix_semantic,
            metadata_suffix_keyword=metadata_suffix_keyword,
            content_token_limit=content_token_limit,
            context_size=context_size,
        )

        # Generate large chunks if enabled [P0]
        if cfg.enable_multipass and cfg.enable_large_chunks:
            large_chunks = self._generate_large_chunks(chunks)
            chunks.extend(large_chunks)

        return chunks

    def _chunk_with_sections(
        self,
        document: ParsedDocument,
        item: IngestionItem,
        title_prefix: str,
        metadata_suffix_semantic: str,
        metadata_suffix_keyword: str,
        content_token_limit: int,
        context_size: int,
    ) -> list[Chunk]:
        """Process document sections including image handling [P0]."""
        chunks: list[Chunk] = []
        source_links: dict[int, str] = {}  # [P2] Offset-based link mapping
        chunk_text = ""

        # Parse document into sections (text sections + image sections)
        sections = self._parse_sections(document)

        for section_idx, section in enumerate(sections):
            section_text = section.get("text", "").strip()
            section_link = section.get("link", "")
            image_file_id = section.get("image_file_id")

            # Skip empty sections
            if not section_text and section_idx > 0:
                continue

            # CASE 1: Image section - create dedicated chunk [P0]
            if image_file_id:
                # Finalize any pending text chunk
                if chunk_text.strip():
                    self._create_chunk(
                        chunks=chunks,
                        item=item,
                        text=chunk_text,
                        source_links=source_links,
                        title_prefix=title_prefix,
                        metadata_suffix_semantic=metadata_suffix_semantic,
                        metadata_suffix_keyword=metadata_suffix_keyword,
                        section_continuation=False,
                        context_size=context_size,
                    )
                    chunk_text = ""
                    source_links = {}

                # Create dedicated image chunk
                self._create_chunk(
                    chunks=chunks,
                    item=item,
                    text=section_text,  # OCR text or caption
                    source_links={0: section_link} if section_link else {},
                    title_prefix=title_prefix,
                    metadata_suffix_semantic=metadata_suffix_semantic,
                    metadata_suffix_keyword=metadata_suffix_keyword,
                    image_file_id=image_file_id,
                    section_continuation=False,
                    context_size=context_size,
                )
                continue

            # CASE 2: Normal text section
            section_token_count = len(self.tokenizer.encode(section_text))

            # Large section - split separately
            if section_token_count > content_token_limit:
                if chunk_text.strip():
                    self._create_chunk(
                        chunks=chunks,
                        item=item,
                        text=chunk_text,
                        source_links=source_links,
                        title_prefix=title_prefix,
                        metadata_suffix_semantic=metadata_suffix_semantic,
                        metadata_suffix_keyword=metadata_suffix_keyword,
                        section_continuation=False,
                        context_size=context_size,
                    )
                    chunk_text = ""
                    source_links = {}

                # Split oversized section
                split_texts = cast(list[str], self.chunk_splitter.chunk(section_text))

                for i, split_text in enumerate(split_texts):
                    # Further split if still too large [P1]
                    if (
                        self.config.strict_chunk_token_limit
                        and len(self.tokenizer.encode(split_text)) > content_token_limit
                    ):
                        smaller_chunks = self._split_oversized_chunk(
                            split_text, content_token_limit
                        )
                        for j, small_chunk in enumerate(smaller_chunks):
                            self._create_chunk(
                                chunks=chunks,
                                item=item,
                                text=small_chunk,
                                source_links={0: section_link},
                                title_prefix=title_prefix,
                                metadata_suffix_semantic=metadata_suffix_semantic,
                                metadata_suffix_keyword=metadata_suffix_keyword,
                                section_continuation=(j != 0),
                                context_size=context_size,
                            )
                    else:
                        self._create_chunk(
                            chunks=chunks,
                            item=item,
                            text=split_text,
                            source_links={0: section_link},
                            title_prefix=title_prefix,
                            metadata_suffix_semantic=metadata_suffix_semantic,
                            metadata_suffix_keyword=metadata_suffix_keyword,
                            section_continuation=(i != 0),
                            context_size=context_size,
                        )
                continue

            # Accumulate sections into chunk
            current_token_count = len(self.tokenizer.encode(chunk_text))
            current_offset = len(chunk_text)  # [P2] Track offset for source links
            next_section_tokens = (
                len(self.tokenizer.encode(SECTION_SEPARATOR)) + section_token_count
            )

            if next_section_tokens + current_token_count <= content_token_limit:
                if chunk_text:
                    chunk_text += SECTION_SEPARATOR
                chunk_text += section_text
                source_links[current_offset] = section_link  # [P2]
            else:
                # Finalize current chunk
                self._create_chunk(
                    chunks=chunks,
                    item=item,
                    text=chunk_text,
                    source_links=source_links,
                    title_prefix=title_prefix,
                    metadata_suffix_semantic=metadata_suffix_semantic,
                    metadata_suffix_keyword=metadata_suffix_keyword,
                    section_continuation=False,
                    context_size=context_size,
                )
                # Start new chunk
                source_links = {0: section_link}
                chunk_text = section_text

        # Finalize remaining text
        if chunk_text.strip() or not chunks:
            self._create_chunk(
                chunks=chunks,
                item=item,
                text=chunk_text,
                source_links=source_links or {0: ""},
                title_prefix=title_prefix,
                metadata_suffix_semantic=metadata_suffix_semantic,
                metadata_suffix_keyword=metadata_suffix_keyword,
                section_continuation=False,
                context_size=context_size,
            )

        return chunks

    def _create_chunk(
        self,
        chunks: list[Chunk],
        item: IngestionItem,
        text: str,
        source_links: dict[int, str],
        title_prefix: str,
        metadata_suffix_semantic: str,
        metadata_suffix_keyword: str,
        section_continuation: bool = False,  # [P2]
        image_file_id: Optional[str] = None,  # [P0]
        context_size: int = 0,
    ) -> None:
        """Create a chunk with all metadata."""
        chunk_id = len(chunks)

        # Generate mini-chunks for multipass [P0]
        mini_chunk_texts = None
        if self.mini_chunk_splitter and text.strip():
            mini_chunk_texts = cast(
                list[str], self.mini_chunk_splitter.chunk(text)
            )

        chunk = Chunk(
            document_id=item.document_id,
            chunk_id=chunk_id,
            content=text,
            title=item.file_name,
            source_type=item.source_type,
            link=item.source_uri,
            blurb=self._extract_blurb(text),
            metadata=item.metadata_json or {},
            tenant_id=item.tenant_id,
            # Enhanced fields
            section_continuation=section_continuation,  # [P2]
            image_file_name=image_file_id,  # [P0]
            # Store for embedding generation
            _title_prefix=title_prefix,
            _metadata_suffix_semantic=metadata_suffix_semantic,
            _metadata_suffix_keyword=metadata_suffix_keyword,
            _mini_chunk_texts=mini_chunk_texts,  # [P0]
            _source_links=source_links,  # [P2]
            _contextual_rag_reserved_tokens=context_size,  # [P1]
        )
        chunks.append(chunk)

    def _split_oversized_chunk(self, text: str, content_token_limit: int) -> list[str]:
        """Token-based splitting for oversized chunks [P1]."""
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        start = 0
        total_tokens = len(tokens)

        while start < total_tokens:
            end = min(start + content_token_limit, total_tokens)
            token_chunk = tokens[start:end]
            chunk_text = " ".join(token_chunk)
            chunks.append(chunk_text)
            start = end

        return chunks

    def _extract_blurb(self, text: str) -> str:
        """Extract short excerpt from text."""
        if not text:
            return ""
        texts = cast(list[str], self.blurb_splitter.chunk(text))
        return texts[0] if texts else ""

    def _parse_sections(self, document: ParsedDocument) -> list[dict]:
        """Parse document into text and image sections."""
        sections = []

        # Add main text section
        if document.text:
            sections.append({"text": document.text, "link": ""})

        # Add image sections [P0]
        for image in document.images:
            sections.append({
                "text": image.caption or "",
                "link": "",
                "image_file_id": image.image_id,
            })

        return sections

    def _generate_large_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Combine chunks into larger chunks for multipass [P0]."""
        large_chunks = []
        ratio = self.config.large_chunk_ratio

        for idx, i in enumerate(range(0, len(chunks), ratio)):
            chunk_group = chunks[i : i + ratio]
            if len(chunk_group) > 1:
                large_chunk = self._combine_chunks(chunk_group, idx)
                large_chunks.append(large_chunk)

        return large_chunks

    def _combine_chunks(self, chunks: list[Chunk], large_chunk_id: int) -> Chunk:
        """Combine multiple chunks into one large chunk [P0]."""
        combined_content = SECTION_SEPARATOR.join(c.content for c in chunks)

        return Chunk(
            document_id=chunks[0].document_id,
            chunk_id=chunks[0].chunk_id,
            content=combined_content,
            title=chunks[0].title,
            source_type=chunks[0].source_type,
            link=chunks[0].link,
            blurb=chunks[0].blurb,
            metadata=chunks[0].metadata,
            tenant_id=chunks[0].tenant_id,
            large_chunk_reference_ids=[c.chunk_id for c in chunks],
            section_continuation=(chunks[0].chunk_id > 0),
        )
```

### Specialized Chunkers [P2]

```python
class SlideChunker(BaseChunker):
    """Slide-based chunking for presentations (pptx)."""

    def supports(self, source_type: str, mime_type: str) -> bool:
        return (
            source_type in ["pptx", "ppt"]
            or "presentation" in mime_type
        )

    @property
    def priority(self) -> int:
        return 10  # Higher priority than default

    def chunk(
        self,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """One chunk per slide, preserving speaker notes."""
        chunks = []
        # Parse slides from document structure
        # Each slide becomes one chunk
        # Speaker notes appended to slide content
        return chunks


class TableChunker(BaseChunker):
    """Row-based chunking for spreadsheets (xlsx, csv)."""

    def supports(self, source_type: str, mime_type: str) -> bool:
        return (
            source_type in ["xlsx", "xls", "csv"]
            or "spreadsheet" in mime_type
        )

    @property
    def priority(self) -> int:
        return 10

    def chunk(
        self,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """Chunk by row groups, preserving headers."""
        chunks = []
        # Group rows into chunks of N rows
        # Repeat headers in each chunk
        # Maintain table structure in markdown format
        return chunks


class CodeChunker(BaseChunker):
    """Syntax-aware chunking for source code files."""

    def supports(self, source_type: str, mime_type: str) -> bool:
        return mime_type in [
            "text/x-python",
            "application/javascript",
            "text/x-java",
            "text/x-c",
            "text/x-typescript",
        ]

    @property
    def priority(self) -> int:
        return 10

    def chunk(
        self,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """Preserve function/class boundaries using AST parsing."""
        chunks = []
        # Use tree-sitter or ast to find function/class boundaries
        # Create one chunk per function/class
        # Include imports and context
        return chunks


class EmailChunker(BaseChunker):
    """Thread-aware chunking for emails (eml, msg)."""

    def supports(self, source_type: str, mime_type: str) -> bool:
        return source_type in ["eml", "msg"] or "message" in mime_type

    @property
    def priority(self) -> int:
        return 10

    def chunk(
        self,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """Separate headers from body, handle threads."""
        chunks = []
        # Extract headers as metadata
        # Chunk body content
        # Handle email threads/replies
        return chunks


# Register specialized chunkers
chunker_registry.register(SlideChunker())
chunker_registry.register(TableChunker())
chunker_registry.register(CodeChunker())
chunker_registry.register(EmailChunker())
chunker_registry.set_default(SemanticChunker(ChunkingConfig(), tokenizer))
```

### Chunking Configuration

```python
# Environment variables
AGENT_RAG_CHUNK_SIZE = 512                # Default chunk size in tokens
AGENT_RAG_CHUNK_OVERLAP = 0               # Overlap between chunks
AGENT_RAG_BLURB_SIZE = 128                # Blurb/excerpt size
AGENT_RAG_MINI_CHUNK_SIZE = 64            # Mini-chunk size for multipass
AGENT_RAG_LARGE_CHUNK_RATIO = 4           # Chunks to combine for large chunks
AGENT_RAG_ENABLE_MULTIPASS = False        # Enable mini-chunk embeddings [P0]
AGENT_RAG_ENABLE_LARGE_CHUNKS = False     # Enable combined large chunks [P0]
AGENT_RAG_ENABLE_CONTEXTUAL_RAG = False   # Enable doc summary/chunk context [P1]
AGENT_RAG_STRICT_TOKEN_LIMIT = True       # Enforce strict token limits [P1]
AGENT_RAG_MAX_CONTEXT_TOKENS = 512        # Max tokens for contextual RAG [P1]
```

### Chunking behavior and retrieval alignment

- Token counting uses the same tokenizer as the embedding provider.
- Enforce `MAX_DOCUMENT_TOKENS` or `MAX_DOCUMENT_CHARS` before chunking.
- `doc_updated_at` should be sourced from upstream metadata if available, otherwise ingestion time.
- `blurb` is generated from the first `AGENT_RAG_BLURB_SIZE` tokens of each chunk.

## Embedding Design [P1]

Based on analysis of Onyx's `backend/onyx/indexing/embedder.py`, implement comprehensive embedding with failure handling and caching.

### Embedding Models [P1]

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ChunkEmbedding:
    """Embedding structure for a chunk (matches Onyx ChunkEmbedding)."""
    full_embedding: list[float]
    mini_chunk_embeddings: list[list[float]] = field(default_factory=list)


@dataclass
class EmbeddingConfig:
    """Embedding configuration parameters."""
    # Model settings
    model_name: str = "text-embedding-ada-002"
    normalize: bool = True
    query_prefix: Optional[str] = None
    passage_prefix: Optional[str] = None
    reduced_dimension: Optional[int] = None  # For dimensionality reduction

    # Provider settings
    provider_type: Optional[str] = None  # openai, azure, local, etc.
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None  # For Azure

    # Performance settings
    batch_size: int = 100  # Texts per API call
    max_retries: int = 3
    retry_delay: float = 2.0  # seconds

    # Feature flags
    enable_title_embedding: bool = True
    cache_title_embeddings: bool = True  # [P1] Cache to avoid recalculation
    average_summary_embeddings: bool = False  # [P2] Average doc_summary + chunk_context
```

### Embedder Interface [P1]

```python
from abc import ABC, abstractmethod
from typing import Protocol, Any

class EmbeddingModel(Protocol):
    """Protocol for embedding model implementations."""

    def encode(
        self,
        texts: list[str],
        text_type: str = "passage",  # "passage" or "query"
        tenant_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> list[list[float]]:
        """Encode texts into embeddings."""
        ...


class IndexingEmbedder(ABC):
    """Base embedder interface for indexing."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._title_embed_cache: dict[str, list[float]] = {}  # [P1] Title cache

    @abstractmethod
    def embed_chunks(
        self,
        chunks: list[Chunk],
        tenant_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> list[IndexChunk]:
        """Add embeddings to chunks."""
        raise NotImplementedError
```

### Default Embedder Implementation [P1]

```python
@dataclass
class IndexChunk:
    """Chunk with embeddings ready for indexing."""
    chunk: Chunk
    embeddings: ChunkEmbedding
    title_embedding: Optional[list[float]] = None


class DefaultIndexingEmbedder(IndexingEmbedder):
    """Default embedder with full feature support."""

    def __init__(self, config: EmbeddingConfig, model: EmbeddingModel):
        super().__init__(config)
        self.model = model

    def embed_chunks(
        self,
        chunks: list[Chunk],
        tenant_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> list[IndexChunk]:
        """
        Add embeddings to chunks with:
        - Full chunk embedding
        - Mini-chunk embeddings (if multipass enabled)
        - Title embedding with caching [P1]
        """
        # Prepare texts for embedding
        flat_texts: list[str] = []
        large_chunks_present = False

        for chunk in chunks:
            if chunk.large_chunk_reference_ids:
                large_chunks_present = True

            # Combine chunk content with title prefix, metadata suffix, context
            chunk_text = (
                f"{chunk._title_prefix or ''}"
                f"{chunk.doc_summary or ''}"
                f"{chunk.content}"
                f"{chunk.chunk_context or ''}"
                f"{chunk._metadata_suffix_semantic or ''}"
            ).strip()

            if not chunk_text:
                chunk_text = chunk.title or ""

            flat_texts.append(chunk_text)

            # Add mini-chunk texts if present [P0]
            if chunk._mini_chunk_texts:
                if chunk.large_chunk_reference_ids:
                    raise RuntimeError("Large chunk should not contain mini chunks")
                flat_texts.extend(chunk._mini_chunk_texts)

        # Batch embed all texts
        embeddings = self.model.encode(
            texts=flat_texts,
            text_type="passage",
            tenant_id=tenant_id,
            request_id=request_id,
        )

        # Cache title embeddings [P1]
        title_embed_dict: dict[str, list[float]] = {}
        if self.config.enable_title_embedding:
            unique_titles = {c.title for c in chunks if c.title}
            if unique_titles:
                # Check cache first
                titles_to_embed = []
                for title in unique_titles:
                    if title in self._title_embed_cache:
                        title_embed_dict[title] = self._title_embed_cache[title]
                    else:
                        titles_to_embed.append(title)

                # Embed missing titles
                if titles_to_embed:
                    title_embeddings = self.model.encode(
                        texts=titles_to_embed,
                        text_type="passage",
                        tenant_id=tenant_id,
                        request_id=request_id,
                    )
                    for title, embedding in zip(titles_to_embed, title_embeddings):
                        title_embed_dict[title] = embedding
                        if self.config.cache_title_embeddings:
                            self._title_embed_cache[title] = embedding

        # Map embeddings back to chunks
        embedded_chunks: list[IndexChunk] = []
        embedding_idx = 0

        for chunk in chunks:
            # Count embeddings for this chunk
            num_embeddings = 1 + (len(chunk._mini_chunk_texts) if chunk._mini_chunk_texts else 0)
            chunk_embeddings = embeddings[embedding_idx : embedding_idx + num_embeddings]

            # Build ChunkEmbedding
            chunk_embedding = ChunkEmbedding(
                full_embedding=chunk_embeddings[0],
                mini_chunk_embeddings=chunk_embeddings[1:] if len(chunk_embeddings) > 1 else [],
            )

            # Get title embedding from cache
            title_embedding = title_embed_dict.get(chunk.title) if chunk.title else None

            embedded_chunks.append(IndexChunk(
                chunk=chunk,
                embeddings=chunk_embedding,
                title_embedding=title_embedding,
            ))

            embedding_idx += num_embeddings

        return embedded_chunks
```

### Embedding Failure Handling [P1]

```python
from dataclasses import dataclass

@dataclass
class FailedDocument:
    """Failed document during embedding."""
    document_id: str
    document_link: Optional[str]


@dataclass
class EmbeddingFailure:
    """Embedding failure record."""
    failed_document: FailedDocument
    failure_message: str
    exception: Optional[Exception] = None


async def embed_chunks_with_failure_handling(
    chunks: list[Chunk],
    embedder: IndexingEmbedder,
    tenant_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> tuple[list[IndexChunk], list[EmbeddingFailure]]:
    """
    Embed chunks with per-document failure isolation.

    First attempts batch embedding. If batch fails, falls back to
    per-document embedding to isolate failures.

    Returns:
        tuple: (successfully_embedded_chunks, failures)
    """
    import time
    from collections import defaultdict

    # First try batch embedding
    try:
        embedded = embedder.embed_chunks(
            chunks=chunks,
            tenant_id=tenant_id,
            request_id=request_id,
        )
        return embedded, []
    except Exception as e:
        logger.exception("Batch embedding failed, trying per-document")
        time.sleep(2)  # Brief delay before retry

    # Group chunks by document
    chunks_by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk.document_id].append(chunk)

    # Embed each document separately
    embedded_chunks: list[IndexChunk] = []
    failures: list[EmbeddingFailure] = []

    for doc_id, doc_chunks in chunks_by_doc.items():
        try:
            doc_embedded = embedder.embed_chunks(
                chunks=doc_chunks,
                tenant_id=tenant_id,
                request_id=request_id,
            )
            embedded_chunks.extend(doc_embedded)
        except Exception as e:
            logger.exception(f"Failed to embed document {doc_id}")
            failures.append(EmbeddingFailure(
                failed_document=FailedDocument(
                    document_id=doc_id,
                    document_link=doc_chunks[0].link if doc_chunks else None,
                ),
                failure_message=str(e),
                exception=e,
            ))

    return embedded_chunks, failures
```

### Embedding Configuration

```python
# Environment variables
AGENT_RAG_EMBEDDING_MODEL = "text-embedding-ada-002"
AGENT_RAG_EMBEDDING_PROVIDER = "openai"  # openai, azure, local
AGENT_RAG_EMBEDDING_NORMALIZE = True
AGENT_RAG_EMBEDDING_BATCH_SIZE = 100
AGENT_RAG_EMBEDDING_MAX_RETRIES = 3
AGENT_RAG_EMBEDDING_CACHE_TITLES = True  # [P1] Enable title caching
AGENT_RAG_AVERAGE_SUMMARY_EMBEDDINGS = False  # [P2] Average summary embeddings
```

## OCR Design (LLM-based, configurable)

- OCR is optional and handled as a separate workflow step.
- `OCRProvider` interface allows swapping LLM OCR with other OCR solutions.
- OCR output can be appended to parsed text and/or stored as structured JSON in MinIO.
- OCR configuration:
  - `AGENT_RAG_OCR_ENABLED`
  - `AGENT_RAG_OCR_PROVIDER`
  - `AGENT_RAG_OCR_LLM_MODEL`
  - `AGENT_RAG_OCR_LLM_API_KEY`
  - `AGENT_RAG_OCR_LLM_API_BASE`

## Indexing and Retrieval Alignment

### Chunk fields mapping

`agent_rag.core.models.Chunk` should be populated as follows:

| ingestion_items field | Chunk field | Notes |
|-----------------------|-------------|-------|
| id | - | Internal, not mapped |
| tenant_id | tenant_id | For multi-tenant filters |
| document_id | document_id | Stable ID for retrieval |
| - | chunk_id | Sequential within document |
| - | content | Chunked text |
| file_name | title | Fallback if no HTML title |
| source_type | source_type | file\|url\|text\|markdown |
| source_uri | link | URL or download link |
| metadata_json | metadata | Merged metadata |
| - | metadata_list | Optional array-based filtering |
| - | doc_summary | Optional LLM enhancement |
| - | chunk_context | Optional LLM enhancement |
| - | image_file_name | For OCR image references |
| updated_at | doc_updated_at | Unix timestamp for recency |

### Search filters

`SearchFilters` alignment:

- `source_types` mapped from `source_type`
- `metadata` stored in `Chunk.metadata` and optionally duplicated into `metadata_list`
- `document_ids` uses `document_id`

## Hatchet Workflow Design [P0/P2]

### Workflow: `content-indexing` (DAG Mode) [P2]

Using Hatchet DAG for optimized parallel execution:

```python
from datetime import timedelta
from hatchet_sdk import Context, EmptyModel
from hatchet_sdk.rate_limit import RateLimit, RateLimitDuration
from pydantic import BaseModel

from .hatchet_client import hatchet

class IndexingInput(BaseModel):
    item_id: str
    tenant_id: str

class ParseOutput(BaseModel):
    text: str
    has_images: bool
    image_count: int

# Define workflow with concurrency control
indexing_workflow = hatchet.workflow(
    name="content-indexing",
    on_events=["ingestion:start"],
    input_validator=IndexingInput,
)

# Task 1: Load input (entry point)
@indexing_workflow.task(
    name="load_input",
    retries=3,
    backoff_factor=2.0,
    backoff_max_seconds=30,
    execution_timeout=timedelta(seconds=60),
)
def load_input(input: IndexingInput, ctx: Context) -> dict:
    """Fetch raw content from MinIO or URL."""
    ctx.log(f"Loading input for item {input.item_id}")
    # Implementation: fetch from MinIO, update status to PROCESSING
    return {"content": content, "filename": filename, "mime_type": mime_type}

# Task 2: Parse content (depends on load_input)
@indexing_workflow.task(
    name="parse_content",
    parents=[load_input],
    retries=2,
    execution_timeout=timedelta(minutes=5),
)
def parse_content(input: IndexingInput, ctx: Context) -> ParseOutput:
    """Run parser and store parsed text in MinIO."""
    load_result = ctx.task_output(load_input)
    # Implementation: select parser, parse, store result
    return ParseOutput(text=text, has_images=has_images, image_count=len(images))

# Task 3: Extract images (depends on parse_content, parallel with normalize_text)
@indexing_workflow.task(
    name="extract_images",
    parents=[parse_content],
    execution_timeout=timedelta(minutes=2),
)
def extract_images(input: IndexingInput, ctx: Context) -> dict:
    """Store images in MinIO (if supported)."""
    parse_result = ctx.task_output(parse_content)
    if not parse_result.has_images:
        return {"skipped": True, "image_ids": []}
    # Implementation: extract and store images
    return {"skipped": False, "image_ids": image_ids}

# Task 4: OCR images (depends on extract_images, optional)
@indexing_workflow.task(
    name="ocr_images",
    parents=[extract_images],
    retries=2,
    backoff_factor=2.0,
    execution_timeout=timedelta(minutes=10),
    rate_limits=[
        RateLimit(static_key="ocr-api-limit", units=1),
        RateLimit(
            dynamic_key="input.tenant_id",
            units=1,
            limit=5,
            duration=RateLimitDuration.MINUTE,
        ),
    ],
)
def ocr_images(input: IndexingInput, ctx: Context) -> dict:
    """Optional OCR step using LLM."""
    extract_result = ctx.task_output(extract_images)
    if extract_result["skipped"] or not OCR_ENABLED:
        return {"skipped": True, "ocr_text": ""}
    # Implementation: OCR each image
    return {"skipped": False, "ocr_text": ocr_text}

# Task 5: Normalize text (depends on parse_content and ocr_images)
@indexing_workflow.task(
    name="normalize_text",
    parents=[parse_content, ocr_images],
    execution_timeout=timedelta(seconds=30),
)
def normalize_text(input: IndexingInput, ctx: Context) -> dict:
    """Merge OCR text, clean text."""
    parse_result = ctx.task_output(parse_content)
    ocr_result = ctx.task_output(ocr_images)
    # Implementation: merge and normalize
    return {"normalized_text": normalized_text}

# Task 6: Chunk content (depends on normalize_text)
@indexing_workflow.task(
    name="chunk_content",
    parents=[normalize_text],
    execution_timeout=timedelta(minutes=2),
)
def chunk_content(input: IndexingInput, ctx: Context) -> dict:
    """Split into chunks using semantic chunking."""
    normalize_result = ctx.task_output(normalize_text)
    # Implementation: run SemanticChunker
    return {"chunk_count": len(chunks), "chunks": chunks}

# Task 7: Embed chunks (depends on chunk_content)
@indexing_workflow.task(
    name="embed_chunks",
    parents=[chunk_content],
    retries=3,
    backoff_factor=2.0,
    execution_timeout=timedelta(minutes=10),
    rate_limits=[
        RateLimit(static_key="embedding-api-limit", units=1),
        RateLimit(
            dynamic_key="input.tenant_id",
            units=1,
            limit=10,
            duration=RateLimitDuration.MINUTE,
        ),
    ],
)
def embed_chunks(input: IndexingInput, ctx: Context) -> dict:
    """Compute embeddings for all chunks."""
    chunk_result = ctx.task_output(chunk_content)
    # Implementation: batch embed
    return {"embedded_count": len(embeddings)}

# Task 8: Index chunks (depends on embed_chunks)
@indexing_workflow.task(
    name="index_chunks",
    parents=[embed_chunks],
    retries=3,
    backoff_factor=2.0,
    execution_timeout=timedelta(minutes=5),
)
def index_chunks(input: IndexingInput, ctx: Context) -> dict:
    """Call DocumentIndex.index_chunks."""
    # Implementation: index to vector store
    return {"indexed_count": indexed_count}

# Task 9: Update status (depends on index_chunks)
@indexing_workflow.task(
    name="update_status",
    parents=[index_chunks],
    execution_timeout=timedelta(seconds=30),
)
def update_status(input: IndexingInput, ctx: Context) -> dict:
    """Update PG status and chunk_count."""
    index_result = ctx.task_output(index_chunks)
    # Implementation: update DB, send webhook if configured
    return {"status": "INDEXED", "chunk_count": index_result["indexed_count"]}

# On-failure handler
@indexing_workflow.on_failure_task()
def handle_failure(input: IndexingInput, ctx: Context) -> dict:
    """Handle workflow failure."""
    errors = ctx.task_run_errors
    ctx.log(f"Workflow failed for item {input.item_id}: {errors}")

    # Update status to FAILED
    # Send failure webhook if configured
    # Optionally schedule retry based on error type

    return {"handled": True, "errors": str(errors)}
```

### Rate Limit Configuration [P0]

```python
# Initialize global rate limits at worker startup
from hatchet_sdk.rate_limit import RateLimitDuration

def init_rate_limits(hatchet):
    """Initialize global rate limits."""
    # Embedding API: 100 requests per minute
    hatchet.rate_limits.put("embedding-api-limit", 100, RateLimitDuration.MINUTE)

    # OCR API: 20 requests per minute
    hatchet.rate_limits.put("ocr-api-limit", 20, RateLimitDuration.MINUTE)

    # Unstructured API: 10 requests per minute
    hatchet.rate_limits.put("unstructured-api-limit", 10, RateLimitDuration.MINUTE)
```

### Retry Configuration [P0]

| Task | Retries | Backoff Factor | Max Backoff | Timeout |
|------|---------|----------------|-------------|---------|
| load_input | 3 | 2.0 | 30s | 60s |
| parse_content | 2 | 2.0 | 60s | 5min |
| extract_images | 1 | - | - | 2min |
| ocr_images | 2 | 2.0 | 120s | 10min |
| normalize_text | 0 | - | - | 30s |
| chunk_content | 0 | - | - | 2min |
| embed_chunks | 3 | 2.0 | 60s | 10min |
| index_chunks | 3 | 2.0 | 60s | 5min |
| update_status | 2 | 2.0 | 30s | 30s |

### Error Handling

```python
from hatchet_sdk import NonRetryableException

class UnsupportedFormatError(NonRetryableException):
    """Raised for unsupported file formats."""
    pass

class CorruptedFileError(NonRetryableException):
    """Raised for corrupted or unreadable files."""
    pass

class QuotaExceededError(NonRetryableException):
    """Raised when tenant quota is exceeded."""
    pass

# In task implementation
@indexing_workflow.task(name="parse_content", ...)
def parse_content(input: IndexingInput, ctx: Context):
    try:
        parser = registry.get_parser(mime_type, extension)
        if not parser:
            raise UnsupportedFormatError(f"No parser for {mime_type}")
        return parser.parse(content, filename)
    except UnsupportedFormatError:
        raise  # Will not retry
    except IOError as e:
        raise  # Will retry based on config
```

### Concurrency Control

```python
from hatchet_sdk import ConcurrencyExpression, ConcurrencyLimitStrategy

# Limit concurrent workflows per tenant
indexing_workflow = hatchet.workflow(
    name="content-indexing",
    concurrency=ConcurrencyExpression(
        expression="input.tenant_id",
        max_runs=10,  # Max 10 concurrent per tenant
        limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
    ),
)
```

### Indexing consistency and rollback

- Indexing is complete only if all chunks are written successfully.
- If `index_chunks` fails after partial writes:
  - Best-effort delete all chunks for `document_id`.
  - Mark ingestion item as `FAILED` with error details.
  - If rollback fails, mark `FAILED_PARTIAL` and allow a cleanup job.
- `force_reindex=true` must delete existing chunks before indexing new content.
- If deletion fails, abort indexing to avoid mixed versions.

### Hatchet runtime model

Worker roles:
- `ingestion-worker`: parse, normalize, chunk
- `ocr-worker`: OCR tasks only
- `indexing-worker`: embed + index

Queue separation:
- OCR and embedding/indexing should be isolated for scaling and to protect latency.

Suggested concurrency:
- OCR: 2-5 concurrent runs per tenant
- Embedding: 5-10 concurrent runs per tenant
- Indexing: up to 5 concurrent runs per tenant

## API Design [P0/P1]

### `POST /ingest` (Single Item)

Input:
```json
{
  "source_type": "file|url|text|markdown",
  "file": "<binary>",           // for file upload
  "url": "https://...",         // for URL
  "text": "...",                // for text/markdown
  "metadata": {},               // optional
  "webhook_url": "https://..."  // optional callback [P2]
}
```

Output:
```json
{
  "item_id": "uuid",
  "status": "PENDING|DUPLICATE",
  "document_id": "...",         // if duplicate
  "message": "..."
}
```

### `POST /ingest/batch` [P1]

Input:
```json
{
  "items": [
    {"source_type": "file", "file": "..."},
    {"source_type": "url", "url": "https://..."}
  ],
  "metadata": {},               // applied to all items
  "webhook_url": "https://..."  // callback when batch completes
}
```

Output:
```json
{
  "batch_id": "uuid",
  "items": [
    {"item_id": "uuid", "status": "PENDING"},
    {"item_id": "uuid", "status": "DUPLICATE", "document_id": "..."}
  ],
  "total": 2,
  "pending": 1,
  "duplicate": 1
}
```

### `GET /ingest/{id}`

Returns:
```json
{
  "item_id": "uuid",
  "status": "PENDING|PROCESSING|INDEXED|FAILED|DUPLICATE",
  "error": "...",               // if failed
  "document_id": "...",         // if indexed
  "chunk_count": 42,            // if indexed
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:01:00Z",
  "processing_time_ms": 1234    // if indexed
}
```

### `GET /ingest/batch/{batch_id}` [P1]

Returns:
```json
{
  "batch_id": "uuid",
  "status": "PROCESSING|COMPLETED|PARTIAL_FAILURE",
  "items": [...],
  "total": 10,
  "indexed": 8,
  "failed": 2,
  "created_at": "...",
  "completed_at": "..."
}
```

### `DELETE /ingest/{id}` [P1]

Deletes ingestion item and associated indexed chunks.

Response:
```json
{
  "deleted": true,
  "document_id": "...",
  "chunks_deleted": 42
}
```

### `POST /ingest/{id}/reindex` [P1]

Re-triggers indexing workflow for an existing item.

Response:
```json
{
  "item_id": "uuid",
  "status": "PENDING",
  "message": "Reindexing triggered"
}
```

## Webhook Notifications [P2]

### Webhook Payload

```json
{
  "event": "ingestion.completed|ingestion.failed",
  "item_id": "uuid",
  "batch_id": "uuid",           // if part of batch
  "tenant_id": "...",
  "status": "INDEXED|FAILED",
  "document_id": "...",         // if successful
  "chunk_count": 42,            // if successful
  "error": "...",               // if failed
  "processing_time_ms": 1234,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Webhook Configuration

```python
AGENT_RAG_WEBHOOK_ENABLED = True
AGENT_RAG_WEBHOOK_TIMEOUT = 30          # seconds
AGENT_RAG_WEBHOOK_RETRY_COUNT = 3
AGENT_RAG_WEBHOOK_RETRY_DELAY = 5       # seconds
```

## Configuration

### Storage

- `MINIO_ENDPOINT`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_BUCKET`
- `MINIO_SECURE`

### OCR

- `AGENT_RAG_OCR_ENABLED`
- `AGENT_RAG_OCR_PROVIDER`
- `AGENT_RAG_OCR_LLM_MODEL`
- `AGENT_RAG_OCR_LLM_API_KEY`
- `AGENT_RAG_OCR_LLM_API_BASE`

### Parsing

- `AGENT_RAG_UNSTRUCTURED_API_KEY` (or `UNSTRUCTURED_API_KEY`)
- `AGENT_RAG_PDF_TABLE_EXTRACTION` (enable pdfplumber)

### Chunking [P0/P1/P2]

Core Settings:
- `AGENT_RAG_CHUNK_SIZE` (default: 512) - Max tokens per chunk
- `AGENT_RAG_CHUNK_OVERLAP` (default: 0) - Token overlap between chunks
- `AGENT_RAG_BLURB_SIZE` (default: 128) - Blurb/excerpt size
- `AGENT_RAG_MINI_CHUNK_SIZE` (default: 64) - Mini-chunk size for multipass [P0]
- `AGENT_RAG_LARGE_CHUNK_RATIO` (default: 4) - Chunks to combine [P0]

Feature Flags:
- `AGENT_RAG_ENABLE_MULTIPASS` (default: false) - Mini-chunk embeddings [P0]
- `AGENT_RAG_ENABLE_LARGE_CHUNKS` (default: false) - Combined large chunks [P0]
- `AGENT_RAG_ENABLE_CONTEXTUAL_RAG` (default: false) - Doc summary/chunk context [P1]
- `AGENT_RAG_STRICT_TOKEN_LIMIT` (default: true) - Enforce strict limits [P1]
- `AGENT_RAG_MAX_CONTEXT_TOKENS` (default: 512) - Contextual RAG budget [P1]

### RAG Enhancement

- `ENABLE_DOC_SUMMARY`
- `ENABLE_CHUNK_CONTEXT`

### Deduplication [P1]

- `AGENT_RAG_DEDUP_ENABLED` (default: true)
- `AGENT_RAG_DEDUP_REPROCESS_FAILED` (default: true)

### Webhook [P2]

- `AGENT_RAG_WEBHOOK_ENABLED` (default: false)
- `AGENT_RAG_WEBHOOK_TIMEOUT` (default: 30)
- `AGENT_RAG_WEBHOOK_RETRY_COUNT` (default: 3)

### Hatchet [P0]

- `HATCHET_CLIENT_TOKEN`
- `HATCHET_CLIENT_TLS_STRATEGY` (none for self-hosted)

## Security and Compliance

- Store raw inputs and derived outputs under tenant-specific namespaces.
- Enforce per-tenant access in all read/write operations and index calls.
- Avoid storing secrets or auth headers in metadata.
- Validate webhook URLs before storing (no internal IPs, valid HTTPS).
- Rate limit API endpoints per tenant.

## Testing Strategy

### Parser Tests
- Parser unit tests for each file type (docx, pdf, pptx, xlsx, eml, epub, msg, html, md, txt)
- OCR provider tests with stubbed LLM responses
- Image extraction tests for docx, pdf, pptx

### Chunking Tests [P0/P1/P2]
- SemanticChunker unit tests:
  - Basic text chunking with token limits
  - Title prefix and metadata suffix handling
  - Image section handling (dedicated chunks) [P0]
  - Multipass mode (mini-chunk generation) [P0]
  - Large chunk generation (_combine_chunks) [P0]
  - Section continuation tracking [P2]
  - Dual metadata suffix (semantic + keyword) [P1]
  - Contextual RAG token reservation [P1]
  - Oversized chunk fallback (strict limits) [P1]
  - Source link offset mapping [P2]

- ChunkerRegistry tests [P1]:
  - Chunker registration and priority ordering
  - Chunker selection by source_type and mime_type
  - Default chunker fallback

- Specialized chunker tests [P2]:
  - SlideChunker: one chunk per slide
  - TableChunker: row-based with header preservation
  - CodeChunker: function/class boundary detection
  - EmailChunker: header/body separation

### Integration Tests
- Deduplication unit tests
- End-to-end ingest for file, URL, and markdown
- Indexing test to validate `Chunk` fields and retrieval alignment
- Hatchet workflow integration tests with mock tasks
- Batch API tests
- Webhook delivery tests

## Rollout Plan

### Phase 1: Core Infrastructure
1) Build `ingestion_items` model and MinIO storage adapter
2) Implement core parsers (docx, pdf, txt, md, html)
3) Implement basic chunking with ChunkingConfig
4) Implement Hatchet workflow with retry/rate limit configs

### Phase 2: Onyx Parity - Parsing (P0)
5) Add PPTX parser
6) Add XLSX parser
7) Complete Hatchet task configurations
8) Implement single-item API endpoints

### Phase 3: Onyx Parity - Chunking (P0)
9) Add image section handling (dedicated image chunks)
10) Implement multipass mode (mini-chunk generation)
11) Implement large chunks (_combine_chunks, _generate_large_chunks)
12) Add section continuation tracking

### Phase 4: Enhanced Features (P1)
13) Add EML, EPUB, MSG parsers
14) Implement content deduplication
15) Add batch import API
16) Add delete/reindex APIs
17) Add dual metadata suffix (semantic + keyword)
18) Add contextual RAG token reservation
19) Add oversized chunk fallback (strict token limits)
20) Add ChunkerRegistry for extensibility
21) Add ChunkEmbedding and IndexChunk models
22) Add title embedding caching in embedder
23) Add embed_chunks_with_failure_handling for per-document failure isolation

### Phase 5: Optimizations (P2)
24) Implement DAG workflow optimization
25) Add pdfplumber for PDF table extraction
26) Add webhook notifications
27) Add progress tracking API
28) Add source link offset mapping
29) Implement specialized chunkers:
    - SlideChunker (pptx - one chunk per slide)
    - TableChunker (xlsx/csv - row-based with headers)
    - CodeChunker (syntax-aware with AST)
    - EmailChunker (thread-aware, header separation)
30) Add AVERAGE_SUMMARY_EMBEDDINGS option

### Phase 6: Production Readiness
31) Add comprehensive integration tests
32) Add chunker unit tests for all strategies
33) Add embedding unit tests with failure handling
34) Add monitoring and metrics
35) Add documentation and configuration guide
36) Performance testing and optimization
