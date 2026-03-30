"""Base parser interface and data models for document parsing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ParsedImage:
    """
    Represents an image extracted from a document.

    Attributes:
        image_id: Unique identifier for this image
        content: Raw image bytes
        mime_type: Image MIME type (e.g., 'image/png', 'image/jpeg')
        page_number: Page number where image was found (optional)
        caption: Image caption or alt text (optional)
    """
    image_id: str
    content: bytes
    mime_type: str
    page_number: Optional[int] = None
    caption: Optional[str] = None


@dataclass
class ParsedDocument:
    """
    Represents a parsed document with extracted content.

    Required fields:
    - text: Normalized text content
    - metadata: Document metadata dict

    Optional fields:
    - images: Extracted images
    - links: Extracted URLs
    - tables: Extracted tables (for xlsx/pdf)

    Normalization rules:
    - Strip control characters except \\n and \\t
    - Collapse repeated whitespace and excessive blank lines
    - Extract and store structured metadata
    """
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    images: list[ParsedImage] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate required fields."""
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dict")


class BaseParser(ABC):
    """
    Base parser interface for document parsing.

    All parsers must implement:
    - supports(): Check if parser can handle file type
    - parse(): Parse content and return ParsedDocument

    Parsers are selected by ParserRegistry based on:
    - File extension match
    - MIME type match
    - Priority (higher priority parsers tried first)
    """

    @abstractmethod
    def supports(self, source_type: str, extension: str, mime_type: str) -> bool:
        """
        Check if this parser supports the given file type.

        Args:
            source_type: Source type (file, url, text, markdown)
            extension: File extension (e.g., 'pdf', 'docx', 'xlsx')
            mime_type: MIME type (e.g., 'application/pdf')

        Returns:
            True if parser can handle this file type
        """
        pass

    @abstractmethod
    def parse(self, content: bytes, filename: str) -> ParsedDocument:
        """
        Parse document content and return structured result.

        Args:
            content: Raw document bytes
            filename: Original filename (for extension detection)

        Returns:
            ParsedDocument with text, images, metadata, etc.

        Raises:
            Exception: If parsing fails
        """
        pass

    @property
    def priority(self) -> int:
        """
        Parser priority for selection.

        Higher priority parsers are tried first when multiple
        parsers support the same file type.

        Default: 0
        Recommended range: -100 to 100

        Returns:
            Priority value (higher = tried first)
        """
        return 0

    @property
    def name(self) -> str:
        """
        Parser name for logging and debugging.

        Default: Class name

        Returns:
            Parser name
        """
        return self.__class__.__name__


class PlainTextParser(BaseParser):
    """
    Plain text parser for .txt files and fallback parsing.

    Supports:
    - .txt files
    - Any UTF-8 text content as fallback
    - markdown source_type

    Priority: -100 (lowest, used as fallback)
    """

    def supports(self, source_type: str, extension: str, mime_type: str) -> bool:
        """Support .txt files and markdown source type."""
        if source_type == "markdown":
            return True
        if extension.lower() in ["txt", "text", "md", "markdown"]:
            return True
        if mime_type and "text/plain" in mime_type:
            return True
        return False

    def parse(self, content: bytes, filename: str) -> ParsedDocument:
        """
        Parse plain text content.

        Args:
            content: Raw text bytes
            filename: Original filename

        Returns:
            ParsedDocument with text content
        """
        try:
            # Try UTF-8 first
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            # Fall back to latin-1 for binary-safe decoding
            text = content.decode("latin-1")

        # Basic metadata
        metadata = {
            "filename": filename,
            "source_type": "text",
        }

        return ParsedDocument(
            text=text,
            metadata=metadata,
            images=[],
            links=[],
            tables=[],
        )

    @property
    def priority(self) -> int:
        """Lowest priority - used as fallback."""
        return -100


class MarkdownParser(BaseParser):
    """
    Markdown parser for .md files.

    Supports:
    - .md, .markdown files
    - markdown source_type

    Priority: 0 (default)
    """

    def supports(self, source_type: str, extension: str, mime_type: str) -> bool:
        """Support markdown files."""
        if source_type == "markdown":
            return True
        if extension.lower() in ["md", "markdown"]:
            return True
        if mime_type and "text/markdown" in mime_type:
            return True
        return False

    def parse(self, content: bytes, filename: str) -> ParsedDocument:
        """
        Parse markdown content.

        Args:
            content: Raw markdown bytes
            filename: Original filename

        Returns:
            ParsedDocument with markdown text
        """
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

        # Extract links (basic pattern matching)
        import re
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = [match[1] for match in re.findall(link_pattern, text)]

        # Basic metadata
        metadata = {
            "filename": filename,
            "source_type": "markdown",
            "format": "markdown",
        }

        return ParsedDocument(
            text=text,
            metadata=metadata,
            images=[],
            links=links,
            tables=[],
        )

    @property
    def priority(self) -> int:
        """Default priority."""
        return 0


class URLParser(BaseParser):
    """
    URL content parser for web pages.

    Supports:
    - source_type = "url"

    Uses trafilatura for content extraction.

    Priority: 0 (default)
    """

    def supports(self, source_type: str, extension: str, mime_type: str) -> bool:
        """Support URL source type."""
        return source_type == "url"

    def parse(self, content: bytes, filename: str) -> ParsedDocument:
        """
        Parse HTML content from URL.

        Args:
            content: Raw HTML bytes
            filename: URL (used as source_uri)

        Returns:
            ParsedDocument with extracted text
        """
        try:
            import trafilatura
        except ImportError:
            raise RuntimeError(
                "trafilatura not installed. Run: pip install trafilatura"
            )

        # Decode HTML
        try:
            html = content.decode("utf-8")
        except UnicodeDecodeError:
            html = content.decode("latin-1")

        # Extract main content
        text = trafilatura.extract(
            html,
            include_links=True,
            include_tables=True,
            include_images=False,  # We'll handle images separately
        )

        if not text:
            # Fallback to basic text extraction
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)

        # Extract links
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        links = [a.get("href") for a in soup.find_all("a", href=True)]

        # Extract metadata
        metadata = {
            "source_uri": filename,  # URL
            "source_type": "url",
        }

        # Try to extract title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)

        # Try to extract meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            metadata["description"] = meta_desc.get("content")

        return ParsedDocument(
            text=text or "",
            metadata=metadata,
            images=[],
            links=links,
            tables=[],
        )

    @property
    def priority(self) -> int:
        """Default priority."""
        return 0
