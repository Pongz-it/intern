"""Chunker registry for automatic chunker selection."""

import logging
from typing import Optional

from agent_rag.core.models import Chunk
from agent_rag.ingestion.chunking.base import BaseChunker
from agent_rag.ingestion.chunking.config import ChunkingConfig
from agent_rag.ingestion.models import IngestionItem
from agent_rag.ingestion.parsing.base import ParsedDocument

logger = logging.getLogger(__name__)


class ChunkerRegistry:
    """
    Registry for document chunkers.

    Manages chunker registration and selection based on:
    - Source type
    - MIME type
    - Document characteristics
    - Chunker priority

    Usage:
        registry = ChunkerRegistry()
        registry.register(MyCustomChunker())

        chunker = registry.get_chunker("file", "application/pdf", parsed_doc)
        chunks = chunker.chunk(parsed_doc, item, config)
    """

    def __init__(self):
        """Initialize chunker registry."""
        self._chunkers: list[BaseChunker] = []
        self._default_chunker: Optional[BaseChunker] = None

        # Will register default chunker after SemanticChunker is imported
        self._register_default_chunker()

    def _register_default_chunker(self):
        """Register default semantic chunker."""
        # Import here to avoid circular dependency
        try:
            from agent_rag.ingestion.chunking.chunker import SemanticChunker

            self._default_chunker = SemanticChunker()
            logger.info("Registered default SemanticChunker")
        except ImportError:
            logger.warning("SemanticChunker not available, no default chunker")

    def register(self, chunker: BaseChunker) -> None:
        """
        Register a new chunker.

        Chunkers are sorted by priority (highest first) after registration.

        Args:
            chunker: Chunker instance to register
        """
        if not isinstance(chunker, BaseChunker):
            raise TypeError(
                f"Chunker must be instance of BaseChunker, got {type(chunker)}"
            )

        logger.info(
            f"Registering chunker: {chunker.name} (priority={chunker.priority})"
        )

        self._chunkers.append(chunker)

        # Sort chunkers by priority (highest first)
        self._chunkers.sort(key=lambda c: c.priority, reverse=True)

    def get_chunker(
        self,
        source_type: str,
        mime_type: str,
        document: ParsedDocument,
    ) -> BaseChunker:
        """
        Get appropriate chunker for document type.

        Selection process:
        1. Iterate chunkers in priority order
        2. Return first chunker that supports the document
        3. Fall back to default chunker if no match

        Args:
            source_type: Source type (file, url, text, markdown)
            mime_type: MIME type
            document: Parsed document to analyze

        Returns:
            Chunker instance that supports the document type
        """
        # Try to find matching chunker
        for chunker in self._chunkers:
            try:
                if chunker.supports(source_type, mime_type, document):
                    logger.debug(
                        f"Selected chunker: {chunker.name} for "
                        f"source_type={source_type}, mime={mime_type}"
                    )
                    return chunker
            except Exception as e:
                logger.warning(
                    f"Chunker {chunker.name} raised error in supports(): {e}"
                )
                continue

        # No specific chunker found, use default
        if self._default_chunker:
            logger.debug(
                f"Using default chunker: {self._default_chunker.name} for "
                f"source_type={source_type}, mime={mime_type}"
            )
            return self._default_chunker

        # No default chunker available
        raise ValueError(
            f"No chunker found for source_type={source_type}, mime_type={mime_type}, "
            f"and no default chunker is registered"
        )

    def chunk(
        self,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
        source_type: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> list[Chunk]:
        """
        Chunk document using automatic chunker selection.

        Convenience method that combines get_chunker() and chunk().

        Args:
            document: Parsed document
            item: Ingestion item
            config: Chunking configuration
            source_type: Source type (defaults to item.source_type)
            mime_type: MIME type (defaults to item.mime_type)

        Returns:
            List of Chunk objects

        Raises:
            ValueError: If no suitable chunker found
            Exception: If chunking fails
        """
        # Use item fields as defaults
        source_type = source_type or str(item.source_type.value)
        mime_type = mime_type or item.mime_type or ""

        # Get appropriate chunker
        chunker = self.get_chunker(source_type, mime_type, document)

        # Chunk document
        logger.info(
            f"Chunking document {item.document_id} with {chunker.name}, "
            f"config: multipass={config.enable_multipass}, "
            f"large={config.enable_large_chunks}"
        )

        try:
            chunks = chunker.chunk(document, item, config)

            logger.info(
                f"Successfully chunked document {item.document_id}: "
                f"{len(chunks)} chunks generated"
            )

            return chunks

        except Exception as e:
            logger.error(
                f"Failed to chunk document {item.document_id} with {chunker.name}: {e}"
            )
            raise

    def list_chunkers(self) -> dict[str, dict[str, any]]:
        """
        Get list of registered chunkers with their info.

        Returns:
            Dict mapping chunker names to their properties
        """
        chunkers_info = {}

        for chunker in self._chunkers:
            chunkers_info[chunker.name] = {
                "priority": chunker.priority,
                "class": chunker.__class__.__name__,
            }

        if self._default_chunker:
            chunkers_info["_default"] = {
                "name": self._default_chunker.name,
                "priority": self._default_chunker.priority,
            }

        return chunkers_info

    def __repr__(self) -> str:
        """String representation."""
        chunker_names = [c.name for c in self._chunkers]
        default_name = self._default_chunker.name if self._default_chunker else "None"
        return (
            f"<ChunkerRegistry with {len(self._chunkers)} chunkers: {chunker_names}, "
            f"default={default_name}>"
        )


# Singleton registry instance
_chunker_registry: Optional[ChunkerRegistry] = None


def get_chunker_registry() -> ChunkerRegistry:
    """
    Get singleton ChunkerRegistry instance.

    Returns:
        ChunkerRegistry instance
    """
    global _chunker_registry
    if _chunker_registry is None:
        _chunker_registry = ChunkerRegistry()
    return _chunker_registry


def register_chunker(chunker: BaseChunker) -> None:
    """
    Register a chunker with the global registry.

    Convenience function for adding custom chunkers.

    Args:
        chunker: Chunker instance to register
    """
    registry = get_chunker_registry()
    registry.register(chunker)
