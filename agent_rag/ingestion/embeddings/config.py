"""Embedding configuration and parameters."""

from dataclasses import dataclass, field
from typing import Optional

from agent_rag.core.env_config import ingestion_config


@dataclass
class EmbeddingConfig:
    """
    Embedding configuration parameters.

    Controls all aspects of chunk embedding including:
    - Model settings
    - Provider configuration
    - Performance tuning
    - Feature flags
    """

    # ========================================================================
    # Model Settings
    # ========================================================================

    model_name: str = "text-embedding-ada-002"
    """Embedding model name (e.g., text-embedding-ada-002, text-embedding-3-small)."""

    model_dimension: int = 1536
    """Embedding vector dimension."""

    normalize_embeddings: bool = True
    """Whether to normalize embeddings to unit length."""

    # ========================================================================
    # Provider Settings
    # ========================================================================

    provider_type: str = "openai"
    """Embedding provider type (openai, cohere, voyage, local)."""

    api_key: Optional[str] = None
    """API key for embedding provider."""

    api_url: Optional[str] = None
    """Custom API URL for provider."""

    # ========================================================================
    # Text Assembly
    # ========================================================================

    title_prefix_enabled: bool = True
    """Whether to include title prefix in embedding text."""

    metadata_suffix_enabled: bool = True
    """Whether to include metadata suffix in embedding text."""

    use_semantic_metadata: bool = True
    """Use semantic metadata suffix (with keys) vs keyword-only suffix."""

    # ========================================================================
    # Performance Settings
    # ========================================================================

    batch_size: int = 32
    """Maximum batch size for embedding API calls."""

    max_retries: int = 3
    """Maximum retry attempts for failed embedding calls."""

    retry_delay: float = 1.0
    """Initial delay in seconds between retries (exponential backoff)."""

    timeout_seconds: float = 30.0
    """Timeout for embedding API calls."""

    # ========================================================================
    # Title Embedding Optimization [P1]
    # ========================================================================

    enable_title_embedding: bool = True
    """Whether to embed title separately for caching."""

    cache_title_embeddings: bool = True
    """Cache title embeddings to avoid recomputation across chunks."""

    title_embed_cache_size: int = 1000
    """Maximum number of title embeddings to cache."""

    # ========================================================================
    # Summary Embedding Options [P1]
    # ========================================================================

    average_summary_embeddings: bool = False
    """
    Average doc_summary and chunk_context embeddings into full_embedding.

    If True:
        full_embedding = average(content_embedding, doc_summary_embedding, chunk_context_embedding)

    If False:
        full_embedding = content_embedding only
        (doc_summary and chunk_context are embedded separately for retrieval augmentation)
    """

    # ========================================================================
    # Failure Handling [P1]
    # ========================================================================

    fail_on_batch_error: bool = False
    """
    If True, fail entire batch if any chunk embedding fails.
    If False, isolate failures per document and continue with successful chunks.
    """

    retry_failed_documents: bool = True
    """Whether to retry failed documents in isolation."""

    max_document_retries: int = 2
    """Maximum retry attempts for individual documents."""

    # ========================================================================
    # Advanced Options
    # ========================================================================

    truncate_long_texts: bool = True
    """Truncate texts exceeding model's max input length."""

    max_input_tokens: int = 8191
    """Maximum input tokens for embedding model (default: text-embedding-ada-002 limit)."""

    embedding_prefix: str = ""
    """Optional prefix to add before all texts (e.g., 'search_document: ')."""

    # ========================================================================
    # Class Methods
    # ========================================================================

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """
        Create EmbeddingConfig from environment variables.

        Returns:
            EmbeddingConfig instance with env values
        """
        return cls(
            model_name=env_config.embedding_model_name,
            model_dimension=env_config.embedding_dimension,
            normalize_embeddings=env_config.normalize_embeddings,
            provider_type=env_config.embedding_provider_type,
            api_key=getattr(env_config, "embedding_api_key", None),
            api_url=getattr(env_config, "embedding_api_url", None),
            title_prefix_enabled=getattr(env_config, "title_prefix_enabled", True),
            metadata_suffix_enabled=getattr(
                env_config, "metadata_suffix_enabled", True
            ),
            use_semantic_metadata=getattr(env_config, "use_semantic_metadata", True),
            batch_size=getattr(env_config, "embedding_batch_size", 32),
            max_retries=getattr(env_config, "embedding_max_retries", 3),
            retry_delay=getattr(env_config, "embedding_retry_delay", 1.0),
            timeout_seconds=getattr(env_config, "embedding_timeout_seconds", 30.0),
            enable_title_embedding=getattr(env_config, "enable_title_embedding", True),
            cache_title_embeddings=getattr(env_config, "cache_title_embeddings", True),
            title_embed_cache_size=getattr(
                env_config, "title_embed_cache_size", 1000
            ),
            average_summary_embeddings=getattr(
                env_config, "average_summary_embeddings", False
            ),
            fail_on_batch_error=getattr(env_config, "fail_on_batch_error", False),
            retry_failed_documents=getattr(env_config, "retry_failed_documents", True),
            max_document_retries=getattr(env_config, "max_document_retries", 2),
            truncate_long_texts=getattr(env_config, "truncate_long_texts", True),
            max_input_tokens=getattr(env_config, "max_input_tokens", 8191),
            embedding_prefix=getattr(env_config, "embedding_prefix", ""),
        )

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.model_dimension <= 0:
            raise ValueError("model_dimension must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        if self.retry_delay < 0:
            raise ValueError("retry_delay cannot be negative")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if self.title_embed_cache_size < 0:
            raise ValueError("title_embed_cache_size cannot be negative")

        if self.max_document_retries < 0:
            raise ValueError("max_document_retries cannot be negative")

        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")

        if not self.provider_type:
            raise ValueError("provider_type cannot be empty")

        if self.provider_type != "local" and not self.api_key:
            raise ValueError(
                f"api_key required for provider_type={self.provider_type}"
            )


# Default configuration instance
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()
