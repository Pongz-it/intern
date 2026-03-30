"""Chunking configuration and parameters."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkingConfig:
    """
    Chunking configuration parameters.

    Controls all aspects of document chunking including:
    - Basic chunk sizing
    - Multipass mode (mini-chunks)
    - Large chunk generation
    - Contextual RAG features
    - Metadata handling
    """

    # ========================================================================
    # Basic Chunking Parameters
    # ========================================================================

    chunk_token_limit: int = 512
    """Maximum tokens per chunk (target size)."""

    chunk_overlap: int = 0
    """Token overlap between consecutive chunks."""

    blurb_size: int = 128
    """Size in tokens for chunk blurb (preview text)."""

    chunk_min_content: int = 256
    """Minimum content tokens (excluding title/metadata)."""

    strict_chunk_token_limit: bool = True
    """If True, enforce strict token limits with fallback splitting."""

    # ========================================================================
    # Multipass Indexing [P0]
    # ========================================================================

    enable_multipass: bool = False
    """Enable mini-chunk generation for multipass indexing."""

    mini_chunk_size: int = 64
    """Size in tokens for mini-chunks (when multipass enabled)."""

    # ========================================================================
    # Large Chunks [P0]
    # ========================================================================

    enable_large_chunks: bool = False
    """Enable generation of combined large chunks."""

    large_chunk_ratio: int = 4
    """Number of regular chunks to combine into one large chunk."""

    # ========================================================================
    # Contextual RAG [P1]
    # ========================================================================

    enable_contextual_rag: bool = False
    """Enable document summary and chunk context generation."""

    use_doc_summary: bool = True
    """Include document summary in contextual RAG."""

    use_chunk_context: bool = True
    """Include chunk-specific context in contextual RAG."""

    max_context_tokens: int = 512
    """Maximum tokens reserved for contextual RAG (doc_summary + chunk_context)."""

    contextual_rag_llm_name: Optional[str] = None
    """LLM model name for generating contextual RAG content."""

    contextual_rag_llm_provider: Optional[str] = None
    """LLM provider for contextual RAG (openai, anthropic, etc.)."""

    # ========================================================================
    # Metadata Handling [P1]
    # ========================================================================

    include_metadata: bool = True
    """Include metadata suffix in chunks."""

    max_metadata_percentage: float = 0.25
    """Maximum percentage of chunk that can be metadata (0.0-1.0)."""

    # ========================================================================
    # Image Handling [P0]
    # ========================================================================

    create_image_chunks: bool = True
    """Create dedicated chunks for images with OCR text."""

    # ========================================================================
    # Advanced Options
    # ========================================================================

    preserve_section_boundaries: bool = True
    """Try to avoid breaking chunks in the middle of sections."""

    track_section_continuation: bool = True
    """Track whether chunk starts mid-section."""

    preserve_source_links: bool = True
    """Track source link offsets in chunks."""

    @property
    def contextual_rag_reserved_tokens(self) -> int:
        """
        Calculate reserved tokens for contextual RAG.

        Based on enabled features (doc_summary, chunk_context).

        Returns:
            Number of tokens to reserve for contextual RAG content
        """
        if not self.enable_contextual_rag:
            return 0

        tokens_per_feature = self.max_context_tokens
        enabled_count = int(self.use_doc_summary) + int(self.use_chunk_context)

        return tokens_per_feature * enabled_count

    @property
    def effective_chunk_token_limit(self) -> int:
        """
        Calculate effective chunk token limit.

        Accounts for contextual RAG reserved tokens.

        Returns:
            Effective token limit for chunk content
        """
        return self.chunk_token_limit - self.contextual_rag_reserved_tokens

    @property
    def max_metadata_tokens(self) -> int:
        """
        Calculate maximum tokens allowed for metadata suffix.

        Returns:
            Maximum metadata tokens based on percentage
        """
        return int(self.chunk_token_limit * self.max_metadata_percentage)

    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        """
        Create ChunkingConfig from environment variables.

        Returns:
            ChunkingConfig instance with env values
        """

        def _get_env_int(key: str, default: int) -> int:
            value = os.environ.get(key)
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                return default

        def _get_env_bool(key: str, default: bool = False) -> bool:
            value = os.environ.get(key, str(default)).lower()
            return value in ("true", "1", "yes", "on")

        return cls(
            chunk_token_limit=_get_env_int("AGENT_RAG_CHUNK_TOKEN_LIMIT", 512),
            chunk_overlap=_get_env_int("AGENT_RAG_CHUNK_OVERLAP", 0),
            blurb_size=_get_env_int("AGENT_RAG_BLURB_SIZE", 128),
            chunk_min_content=_get_env_int("AGENT_RAG_CHUNK_MIN_CONTENT", 256),
            strict_chunk_token_limit=_get_env_bool(
                "AGENT_RAG_STRICT_CHUNK_TOKEN_LIMIT", True
            ),
            enable_multipass=_get_env_bool("AGENT_RAG_ENABLE_MULTIPASS", False),
            mini_chunk_size=_get_env_int("AGENT_RAG_MINI_CHUNK_SIZE", 64),
            enable_large_chunks=_get_env_bool("AGENT_RAG_ENABLE_LARGE_CHUNKS", False),
            large_chunk_ratio=_get_env_int("AGENT_RAG_LARGE_CHUNK_RATIO", 4),
            enable_contextual_rag=_get_env_bool(
                "AGENT_RAG_ENABLE_CONTEXTUAL_RAG", False
            ),
            use_doc_summary=_get_env_bool("AGENT_RAG_USE_DOC_SUMMARY", True),
            use_chunk_context=_get_env_bool("AGENT_RAG_USE_CHUNK_CONTEXT", True),
            max_context_tokens=_get_env_int("AGENT_RAG_MAX_CONTEXT_TOKENS", 512),
            contextual_rag_llm_name=os.environ.get("AGENT_RAG_CONTEXTUAL_RAG_LLM_NAME"),
            contextual_rag_llm_provider=os.environ.get(
                "AGENT_RAG_CONTEXTUAL_RAG_LLM_PROVIDER"
            ),
            include_metadata=True,
            max_metadata_percentage=0.25,
            create_image_chunks=True,
            preserve_section_boundaries=True,
            track_section_continuation=True,
            preserve_source_links=True,
        )

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.chunk_token_limit <= 0:
            raise ValueError("chunk_token_limit must be positive")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if self.chunk_overlap >= self.chunk_token_limit:
            raise ValueError("chunk_overlap must be less than chunk_token_limit")

        if self.blurb_size <= 0:
            raise ValueError("blurb_size must be positive")

        if self.mini_chunk_size <= 0:
            raise ValueError("mini_chunk_size must be positive")

        if self.large_chunk_ratio <= 0:
            raise ValueError("large_chunk_ratio must be positive")

        if not 0.0 <= self.max_metadata_percentage <= 1.0:
            raise ValueError("max_metadata_percentage must be between 0.0 and 1.0")

        if self.max_context_tokens < 0:
            raise ValueError("max_context_tokens cannot be negative")

        # Check effective limits
        if self.effective_chunk_token_limit <= 0:
            raise ValueError(
                f"effective_chunk_token_limit is {self.effective_chunk_token_limit}, "
                f"contextual RAG reserves too many tokens"
            )

        if self.effective_chunk_token_limit < self.chunk_min_content:
            raise ValueError(
                f"effective_chunk_token_limit ({self.effective_chunk_token_limit}) "
                f"is less than chunk_min_content ({self.chunk_min_content})"
            )


# Default configuration instance
DEFAULT_CHUNKING_CONFIG = ChunkingConfig()
