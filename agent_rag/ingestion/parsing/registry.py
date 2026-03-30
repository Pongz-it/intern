"""Parser registry for automatic parser selection."""

import logging
from pathlib import Path
from typing import Optional

from agent_rag.ingestion.parsing.base import (
    BaseParser,
    MarkdownParser,
    ParsedDocument,
    PlainTextParser,
    URLParser,
)
from agent_rag.ingestion.parsing.parsers.docx_parser import DOCXParser
from agent_rag.ingestion.parsing.parsers.pdf_parser import PDFParser
from agent_rag.ingestion.parsing.parsers.pptx_parser import PPTXParser
from agent_rag.ingestion.parsing.parsers.xlsx_parser import XLSXParser
from agent_rag.ingestion.parsing.unstructured import get_unstructured_api_key, unstructured_to_text
from agent_rag.ingestion.parsing.utils import normalize_text

logger = logging.getLogger(__name__)


class ParserRegistry:
    """
    Registry for document parsers.

    Manages parser registration and selection based on:
    - Source type
    - File extension
    - MIME type
    - Parser priority

    Usage:
        registry = ParserRegistry()
        registry.register(MyCustomParser())

        parser = registry.get_parser("file", ".pdf", "application/pdf")
        document = parser.parse(content, "file.pdf")
    """

    def __init__(self):
        """Initialize parser registry with built-in parsers."""
        self._parsers: list[BaseParser] = []

        # Register built-in parsers
        self._register_builtin_parsers()

    def _register_builtin_parsers(self):
        """Register built-in parsers with default priority order."""
        # Register in reverse priority order (lowest first)
        # PlainTextParser has lowest priority (-100) and acts as fallback
        self.register(PlainTextParser())
        self.register(MarkdownParser())
        self.register(URLParser())

        # Register document parsers (higher priority than PlainText)
        self.register(PDFParser())
        self.register(DOCXParser())
        self.register(PPTXParser())
        self.register(XLSXParser())

    def register(self, parser: BaseParser) -> None:
        """
        Register a new parser.

        Parsers are sorted by priority (highest first) after registration.

        Args:
            parser: Parser instance to register
        """
        if not isinstance(parser, BaseParser):
            raise TypeError(f"Parser must be instance of BaseParser, got {type(parser)}")

        logger.info(
            f"Registering parser: {parser.name} (priority={parser.priority})"
        )

        self._parsers.append(parser)

        # Sort parsers by priority (highest first)
        self._parsers.sort(key=lambda p: p.priority, reverse=True)

    def get_parser(
        self,
        source_type: str,
        extension: str,
        mime_type: Optional[str] = None,
    ) -> BaseParser:
        """
        Get appropriate parser for file type.

        Selection process:
        1. Normalize extension (remove leading dot, lowercase)
        2. Iterate parsers in priority order
        3. Return first parser that supports the file type
        4. Raise error if no parser found

        Args:
            source_type: Source type (file, url, text, markdown)
            extension: File extension (with or without leading dot)
            mime_type: MIME type (optional)

        Returns:
            Parser instance that supports the file type

        Raises:
            ValueError: If no suitable parser found
        """
        # Normalize extension
        if extension.startswith("."):
            extension = extension[1:]
        extension = extension.lower()

        # Try to find matching parser
        for parser in self._parsers:
            try:
                if parser.supports(source_type, extension, mime_type or ""):
                    logger.debug(
                        f"Selected parser: {parser.name} for "
                        f"source_type={source_type}, ext={extension}, "
                        f"mime={mime_type}"
                    )
                    return parser
            except Exception as e:
                logger.warning(
                    f"Parser {parser.name} raised error in supports(): {e}"
                )
                continue

        # No parser found
        raise ValueError(
            f"No parser found for source_type={source_type}, "
            f"extension={extension}, mime_type={mime_type}"
        )

    def parse(
        self,
        content: bytes,
        filename: str,
        source_type: str = "file",
        mime_type: Optional[str] = None,
    ) -> ParsedDocument:
        """
        Parse content using automatic parser selection.

        Convenience method that combines get_parser() and parse().

        Args:
            content: Raw document bytes
            filename: Original filename
            source_type: Source type (file, url, text, markdown)
            mime_type: MIME type (optional)

        Returns:
            ParsedDocument with extracted content

        Raises:
            ValueError: If no suitable parser found
            Exception: If parsing fails
        """
        # Extract extension from filename
        extension = Path(filename).suffix or ""

        # Get appropriate parser
        try:
            parser = self.get_parser(source_type, extension, mime_type)
        except ValueError as e:
            if get_unstructured_api_key():
                logger.info(f"No parser found, using Unstructured for {filename}")
                text = normalize_text(unstructured_to_text(content, filename))
                return ParsedDocument(
                    text=text,
                    metadata={"filename": filename, "parser": "unstructured"},
                    images=[],
                    links=[],
                    tables=[],
                )
            raise

        # Parse content
        logger.info(f"Parsing {filename} with {parser.name}")
        try:
            document = parser.parse(content, filename)
        except Exception as e:
            logger.error(f"Failed to parse {filename} with {parser.name}: {e}")
            if get_unstructured_api_key():
                logger.info(f"Falling back to Unstructured for {filename}")
                text = normalize_text(unstructured_to_text(content, filename))
                return ParsedDocument(
                    text=text,
                    metadata={"filename": filename, "parser": "unstructured"},
                    images=[],
                    links=[],
                    tables=[],
                )
            raise

        logger.info(
            f"Successfully parsed {filename}: "
            f"{len(document.text)} chars, "
            f"{len(document.images)} images, "
            f"{len(document.tables)} tables"
        )
        return document

    def list_supported_extensions(self) -> dict[str, list[str]]:
        """
        Get list of supported extensions by parser.

        Returns:
            Dict mapping parser names to supported extensions
        """
        # This is a simple implementation that asks each parser
        # what it supports. For better UX, parsers should declare
        # their supported extensions explicitly.
        supported = {}
        common_extensions = [
            "txt", "md", "pdf", "docx", "doc", "pptx", "ppt",
            "xlsx", "xls", "csv", "html", "htm",
        ]

        for parser in self._parsers:
            parser_extensions = []
            for ext in common_extensions:
                try:
                    if parser.supports("file", ext, ""):
                        parser_extensions.append(ext)
                except Exception:
                    pass

            if parser_extensions:
                supported[parser.name] = parser_extensions

        return supported

    def __repr__(self) -> str:
        """String representation."""
        parser_names = [p.name for p in self._parsers]
        return f"<ParserRegistry with {len(self._parsers)} parsers: {parser_names}>"


# Singleton registry instance
_parser_registry: Optional[ParserRegistry] = None


def get_parser_registry() -> ParserRegistry:
    """
    Get singleton ParserRegistry instance.

    Returns:
        ParserRegistry instance
    """
    global _parser_registry
    if _parser_registry is None:
        _parser_registry = ParserRegistry()
    return _parser_registry


def register_parser(parser: BaseParser) -> None:
    """
    Register a parser with the global registry.

    Convenience function for adding custom parsers.

    Args:
        parser: Parser instance to register
    """
    registry = get_parser_registry()
    registry.register(parser)
