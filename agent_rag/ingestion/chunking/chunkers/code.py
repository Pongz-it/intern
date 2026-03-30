"""Code-aware chunker for source code files."""

import logging
import re
from typing import Optional

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


# Language-specific syntax patterns for code structure detection
CODE_PATTERNS = {
    "python": {
        "function": r"^(?:async\s+)?def\s+(\w+)",
        "class": r"^class\s+(\w+)",
        "method": r"^\s{4}(?:async\s+)?def\s+(\w+)",
        "decorator": r"^@\w+",
        "import": r"^(?:import|from)\s+",
        "comment_block": r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'',
    },
    "javascript": {
        "function": r"^(?:async\s+)?function\s+(\w+)|^const\s+(\w+)\s*=\s*(?:async\s+)?\(|^(?:export\s+)?(?:async\s+)?function\s*\*?\s*(\w+)",
        "class": r"^(?:export\s+)?class\s+(\w+)",
        "method": r"^\s{2,}(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{",
        "import": r"^import\s+|^const\s+\{[^}]+\}\s*=\s*require",
        "comment_block": r"/\*[\s\S]*?\*/",
    },
    "typescript": {
        "function": r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)|^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(",
        "class": r"^(?:export\s+)?class\s+(\w+)",
        "interface": r"^(?:export\s+)?interface\s+(\w+)",
        "type": r"^(?:export\s+)?type\s+(\w+)",
        "method": r"^\s{2,}(?:async\s+)?(\w+)\s*\([^)]*\)",
        "import": r"^import\s+",
        "comment_block": r"/\*[\s\S]*?\*/",
    },
    "java": {
        "class": r"^(?:public\s+)?(?:abstract\s+)?class\s+(\w+)",
        "interface": r"^(?:public\s+)?interface\s+(\w+)",
        "method": r"^\s{4}(?:public|private|protected)\s+(?:static\s+)?(?:\w+)\s+(\w+)\s*\(",
        "import": r"^import\s+",
        "comment_block": r"/\*[\s\S]*?\*/",
    },
    "go": {
        "function": r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)",
        "struct": r"^type\s+(\w+)\s+struct",
        "interface": r"^type\s+(\w+)\s+interface",
        "import": r"^import\s+",
        "comment_block": r"/\*[\s\S]*?\*/",
    },
    "rust": {
        "function": r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)",
        "struct": r"^(?:pub\s+)?struct\s+(\w+)",
        "impl": r"^impl(?:<[^>]+>)?\s+(?:\w+\s+for\s+)?(\w+)",
        "trait": r"^(?:pub\s+)?trait\s+(\w+)",
        "mod": r"^(?:pub\s+)?mod\s+(\w+)",
        "use": r"^use\s+",
        "comment_block": r"/\*[\s\S]*?\*/",
    },
}

# File extension to language mapping
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}

# MIME types for code files
CODE_MIME_TYPES = {
    "text/x-python",
    "application/x-python-code",
    "text/javascript",
    "application/javascript",
    "text/typescript",
    "application/typescript",
    "text/x-java-source",
    "text/x-go",
    "text/x-rust",
}


class CodeChunker(BaseChunker):
    """
    Syntax-aware code chunker.

    Features:
    - Language detection from extension/MIME type
    - Function/class boundary detection
    - Preserves logical code units
    - Handles imports, decorators, docstrings
    - Falls back to line-based splitting for unsupported languages

    Supported languages: Python, JavaScript, TypeScript, Java, Go, Rust
    """

    @property
    def priority(self) -> int:
        """High priority for code files."""
        return 50

    @property
    def name(self) -> str:
        """Chunker name."""
        return "CodeChunker"

    def supports(self, source_type: str, mime_type: str, document: ParsedDocument) -> bool:
        """
        Check if this chunker supports the document.

        Args:
            source_type: Source type
            mime_type: MIME type
            document: Parsed document

        Returns:
            True if document appears to be source code
        """
        # Check MIME type
        if mime_type in CODE_MIME_TYPES:
            return True

        # Check file extension from metadata
        filename = document.metadata.get("filename", "") or document.metadata.get("file_name", "")
        if filename:
            for ext in EXTENSION_TO_LANGUAGE:
                if filename.lower().endswith(ext):
                    return True

        # Check if text contains code-like patterns
        if document.text:
            # Simple heuristic: check for common code indicators
            code_indicators = [
                r"^def\s+\w+\s*\(",  # Python function
                r"^class\s+\w+",  # Class definition
                r"^import\s+",  # Import statement
                r"^from\s+\w+\s+import",  # Python import
                r"^function\s+\w+\s*\(",  # JavaScript function
                r"^const\s+\w+\s*=",  # JavaScript const
                r"^pub\s+fn\s+\w+",  # Rust function
                r"^func\s+\w+\s*\(",  # Go function
            ]
            for pattern in code_indicators:
                if re.search(pattern, document.text, re.MULTILINE):
                    return True

        return False

    def chunk(
        self,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
    ) -> list[Chunk]:
        """
        Chunk source code into semantic units.

        Args:
            document: Parsed document containing source code
            item: Ingestion item
            config: Chunking configuration

        Returns:
            List of chunks representing code units
        """
        # Detect programming language
        language = self._detect_language(document, item)
        logger.debug(f"Detected language: {language}")

        # Get text content
        text = document.text or ""
        if not text.strip():
            return []

        # Build title prefix
        title = document.metadata.get("title") or item.file_name
        title_prefix = f"{title}\n\n" if title else ""

        # Get patterns for language
        patterns = CODE_PATTERNS.get(language, {})

        # Parse code into logical units
        code_units = self._parse_code_units(text, patterns, language)

        # Build chunks from code units
        chunks = []
        chunk_id = 0

        for unit in code_units:
            unit_text = unit["text"]
            unit_type = unit["type"]
            unit_name = unit.get("name", "")

            # Check if unit fits in token limit
            unit_tokens = count_tokens(unit_text)

            if unit_tokens <= config.chunk_token_limit:
                # Single chunk for this unit
                chunk = self._create_code_chunk(
                    chunk_id=chunk_id,
                    content=unit_text,
                    unit_type=unit_type,
                    unit_name=unit_name,
                    title_prefix=title_prefix,
                    document=document,
                    item=item,
                    config=config,
                    language=language,
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # Split large unit into smaller chunks
                sub_chunks = self._split_large_unit(
                    unit_text=unit_text,
                    unit_type=unit_type,
                    unit_name=unit_name,
                    start_chunk_id=chunk_id,
                    title_prefix=title_prefix,
                    document=document,
                    item=item,
                    config=config,
                    language=language,
                )
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)

        logger.info(
            f"CodeChunker generated {len(chunks)} chunks from {len(code_units)} code units"
        )

        return chunks

    def _detect_language(self, document: ParsedDocument, item: IngestionItem) -> str:
        """
        Detect programming language from file extension or content.

        Args:
            document: Parsed document
            item: Ingestion item

        Returns:
            Language identifier (e.g., 'python', 'javascript')
        """
        # Check filename
        filename = (
            document.metadata.get("filename")
            or document.metadata.get("file_name")
            or item.file_name
            or ""
        )

        for ext, lang in EXTENSION_TO_LANGUAGE.items():
            if filename.lower().endswith(ext):
                return lang

        # Heuristic detection from content
        text = (document.text or "")[:1000]  # Check first 1000 chars

        if re.search(r"^def\s+\w+\s*\(|^class\s+\w+:|^import\s+\w+", text, re.MULTILINE):
            return "python"
        if re.search(r"^(?:const|let|var)\s+\w+|^function\s+\w+|^import\s+\{", text, re.MULTILINE):
            return "javascript"
        if re.search(r"^(?:pub\s+)?fn\s+\w+|^use\s+\w+::", text, re.MULTILINE):
            return "rust"
        if re.search(r"^func\s+\w+\s*\(|^package\s+\w+", text, re.MULTILINE):
            return "go"
        if re.search(r"^(?:public|private)\s+class\s+\w+", text, re.MULTILINE):
            return "java"

        return "unknown"

    def _parse_code_units(
        self,
        text: str,
        patterns: dict[str, str],
        language: str,
    ) -> list[dict]:
        """
        Parse code into logical units (functions, classes, etc.).

        Args:
            text: Source code text
            patterns: Regex patterns for this language
            language: Language identifier

        Returns:
            List of code units with type, name, and text
        """
        if not patterns:
            # No patterns for this language, fall back to line-based
            return self._parse_by_lines(text)

        lines = text.split("\n")
        units = []
        current_unit = {
            "type": "header",
            "name": "",
            "lines": [],
        }

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for class/function/struct definitions
            unit_detected = False

            for unit_type, pattern in patterns.items():
                if unit_type in ("import", "comment_block"):
                    continue  # Handle separately

                match = re.match(pattern, line)
                if match:
                    # Save current unit if non-empty
                    if current_unit["lines"]:
                        current_unit["text"] = "\n".join(current_unit["lines"])
                        units.append(current_unit)

                    # Get name from match groups
                    name_groups = [g for g in match.groups() if g]
                    unit_name = name_groups[0] if name_groups else ""

                    # Start new unit
                    current_unit = {
                        "type": unit_type,
                        "name": unit_name,
                        "lines": [line],
                    }
                    unit_detected = True
                    break

            if not unit_detected:
                current_unit["lines"].append(line)

            i += 1

        # Add final unit
        if current_unit["lines"]:
            current_unit["text"] = "\n".join(current_unit["lines"])
            units.append(current_unit)

        return units

    def _parse_by_lines(self, text: str, lines_per_chunk: int = 50) -> list[dict]:
        """
        Fall back to line-based parsing for unsupported languages.

        Args:
            text: Source code text
            lines_per_chunk: Number of lines per chunk

        Returns:
            List of line-based units
        """
        lines = text.split("\n")
        units = []

        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i : i + lines_per_chunk]
            units.append({
                "type": "lines",
                "name": f"lines_{i + 1}-{min(i + lines_per_chunk, len(lines))}",
                "text": "\n".join(chunk_lines),
            })

        return units

    def _split_large_unit(
        self,
        unit_text: str,
        unit_type: str,
        unit_name: str,
        start_chunk_id: int,
        title_prefix: str,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
        language: str,
    ) -> list[Chunk]:
        """
        Split a large code unit into smaller chunks.

        Args:
            unit_text: The code text to split
            unit_type: Type of code unit
            unit_name: Name of the code unit
            start_chunk_id: Starting chunk ID
            title_prefix: Title prefix for chunks
            document: Parsed document
            item: Ingestion item
            config: Chunking configuration
            language: Programming language

        Returns:
            List of chunks
        """
        chunks = []
        lines = unit_text.split("\n")

        # Calculate lines per chunk based on token limit
        avg_tokens_per_line = count_tokens(unit_text) / max(len(lines), 1)
        lines_per_chunk = max(10, int(config.chunk_token_limit / max(avg_tokens_per_line, 1)))

        chunk_id = start_chunk_id
        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i : i + lines_per_chunk]
            chunk_text = "\n".join(chunk_lines)

            # Truncate if still too long
            chunk_text = truncate_to_tokens(chunk_text, config.chunk_token_limit)

            chunk = self._create_code_chunk(
                chunk_id=chunk_id,
                content=chunk_text,
                unit_type=unit_type,
                unit_name=unit_name,
                title_prefix=title_prefix,
                document=document,
                item=item,
                config=config,
                language=language,
                is_continuation=(i > 0),
            )
            chunks.append(chunk)
            chunk_id += 1

        return chunks

    def _create_code_chunk(
        self,
        chunk_id: int,
        content: str,
        unit_type: str,
        unit_name: str,
        title_prefix: str,
        document: ParsedDocument,
        item: IngestionItem,
        config: ChunkingConfig,
        language: str,
        is_continuation: bool = False,
    ) -> Chunk:
        """
        Create a chunk from code content.

        Args:
            chunk_id: Chunk ID
            content: Code content
            unit_type: Type of code unit
            unit_name: Name of code unit
            title_prefix: Title prefix
            document: Parsed document
            item: Ingestion item
            config: Chunking configuration
            language: Programming language
            is_continuation: Whether this is a continuation chunk

        Returns:
            Chunk instance
        """
        # Build metadata
        chunk_metadata = dict(document.metadata or {})
        chunk_metadata.update({
            "language": language,
            "code_unit_type": unit_type,
            "code_unit_name": unit_name,
        })

        # Create blurb from first few lines
        lines = content.split("\n")
        blurb = "\n".join(lines[:5])
        if len(blurb) > config.blurb_size * 4:  # Approximate character limit
            blurb = blurb[: config.blurb_size * 4]

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
            # Internal fields
            _title_prefix=title_prefix,
            _metadata_suffix_semantic=f"\nLanguage: {language}\nUnit: {unit_type} {unit_name}",
            _metadata_suffix_keyword=f"{language} {unit_type} {unit_name}",
        )
