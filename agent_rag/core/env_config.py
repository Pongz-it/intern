"""Environment-based configuration loader for Agent RAG."""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from agent_rag.core.config import (
    AgentConfig,
    AgentMode,
    AgentRAGConfig,
    DeepResearchConfig,
    DocumentIndexConfig,
    EmbeddingConfig,
    LLMConfig,
    SearchConfig,
)
from agent_rag.document_index.vespa.schema_config import VespaSchemaConfig


def _get_env(key: str, default: Any = None) -> Optional[str]:
    """Get environment variable with optional default."""
    return os.environ.get(key, default)


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    value = os.environ.get(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_list(key: str, default: list[str] | None = None) -> list[str]:
    """Get list from comma-separated environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default or []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_dotenv(env_path: Optional[Path] = None) -> None:
    """
    Load environment variables from .env file.

    Args:
        env_path: Path to .env file. If None, looks for .env in:
            1. Current working directory
            2. agent_rag package directory
    """
    if env_path is None:
        # Try current directory first
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            env_path = cwd_env
        else:
            # Try package directory
            pkg_env = Path(__file__).parent.parent.parent / ".env"
            if pkg_env.exists():
                env_path = pkg_env

    if env_path is None or not env_path.exists():
        return

    # Simple .env parser (no external dependencies)
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Handle key=value pairs
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value and value[0] in ("'", '"') and value[-1] == value[0]:
                    value = value[1:-1]

                # Only set if not already in environment
                if key not in os.environ:
                    os.environ[key] = value


def get_llm_config_from_env() -> LLMConfig:
    """Create LLMConfig from environment variables."""
    return LLMConfig(
        model=_get_env("AGENT_RAG_LLM_MODEL", "gpt-4o"),
        provider=_get_env("AGENT_RAG_LLM_PROVIDER", "litellm"),
        api_key=_get_env("AGENT_RAG_LLM_API_KEY"),
        api_base=_get_env("AGENT_RAG_LLM_API_BASE"),
        max_tokens=_get_env_int("AGENT_RAG_LLM_MAX_TOKENS", 4096),
        max_input_tokens=_get_env_int("AGENT_RAG_LLM_MAX_INPUT_TOKENS", 128000),
        temperature=_get_env_float("AGENT_RAG_LLM_TEMPERATURE", 0.0),
        timeout=_get_env_int("AGENT_RAG_LLM_TIMEOUT", 120),
        is_reasoning_model=_get_env_bool("AGENT_RAG_LLM_IS_REASONING_MODEL", False),
        reasoning_effort=_get_env("AGENT_RAG_LLM_REASONING_EFFORT", "medium"),
    )


def get_embedding_config_from_env() -> EmbeddingConfig:
    """Create EmbeddingConfig from environment variables."""
    return EmbeddingConfig(
        model=_get_env("AGENT_RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
        provider=_get_env("AGENT_RAG_EMBEDDING_PROVIDER", "litellm"),
        api_key=_get_env("AGENT_RAG_EMBEDDING_API_KEY"),
        api_base=_get_env("AGENT_RAG_EMBEDDING_API_BASE"),
        dimension=_get_env_int("AGENT_RAG_EMBEDDING_DIMENSION", 1536),
        batch_size=_get_env_int("AGENT_RAG_EMBEDDING_BATCH_SIZE", 32),
    )


def get_document_index_config_from_env() -> DocumentIndexConfig:
    """Create DocumentIndexConfig from environment variables."""
    decay_default = 0.5
    legacy_decay = _get_env("AGENT_RAG_VESPA_DEFAULT_DECAY_FACTOR")
    if legacy_decay is not None:
        try:
            decay_default = float(legacy_decay)
        except ValueError:
            decay_default = 0.5

    return DocumentIndexConfig(
        type=_get_env("AGENT_RAG_INDEX_TYPE", "memory"),
        vespa_host=_get_env("AGENT_RAG_VESPA_HOST", "localhost"),
        vespa_port=_get_env_int("AGENT_RAG_VESPA_PORT", 8080),
        vespa_app_name=_get_env("AGENT_RAG_VESPA_APP_NAME", "agent_rag"),
        vespa_timeout=_get_env_int("AGENT_RAG_VESPA_TIMEOUT", 30),
        vespa_schema_name=_get_env("AGENT_RAG_VESPA_SCHEMA_NAME", "agent_rag_chunk"),
        vespa_title_content_ratio=_get_env_float("AGENT_RAG_VESPA_TITLE_CONTENT_RATIO", 0.2),
        vespa_decay_factor=_get_env_float("AGENT_RAG_VESPA_DECAY_FACTOR", decay_default),
        memory_persist_path=_get_env("AGENT_RAG_MEMORY_PERSIST_PATH"),
    )


def get_vespa_schema_config_from_env() -> VespaSchemaConfig:
    """Create VespaSchemaConfig from environment variables."""
    return VespaSchemaConfig(
        schema_name=_get_env("AGENT_RAG_VESPA_SCHEMA_NAME", "agent_rag_chunk"),
        dim=_get_env_int("AGENT_RAG_VESPA_SCHEMA_DIM", 1536),
        embedding_precision=_get_env("AGENT_RAG_VESPA_EMBEDDING_PRECISION", "float"),
        multi_tenant=_get_env_bool("AGENT_RAG_VESPA_MULTI_TENANT", False),
        enable_title_embedding=_get_env_bool("AGENT_RAG_VESPA_ENABLE_TITLE_EMBEDDING", True),
        enable_large_chunks=_get_env_bool("AGENT_RAG_VESPA_ENABLE_LARGE_CHUNKS", True),
        enable_knowledge_graph=_get_env_bool("AGENT_RAG_VESPA_ENABLE_KNOWLEDGE_GRAPH", True),
        enable_access_control=_get_env_bool("AGENT_RAG_VESPA_ENABLE_ACCESS_CONTROL", False),
        default_decay_factor=_get_env_float("AGENT_RAG_VESPA_DEFAULT_DECAY_FACTOR", 0.5),
        rerank_count=_get_env_int("AGENT_RAG_VESPA_RERANK_COUNT", 1000),
        redundancy=_get_env_int("AGENT_RAG_VESPA_REDUNDANCY", 1),
        searchable_copies=_get_env_int("AGENT_RAG_VESPA_SEARCHABLE_COPIES", 1),
        search_threads=_get_env_int("AGENT_RAG_VESPA_SEARCH_THREADS", 4),
        summary_threads=_get_env_int("AGENT_RAG_VESPA_SUMMARY_THREADS", 2),
    )


def get_search_config_from_env() -> SearchConfig:
    """Create SearchConfig from environment variables."""
    keyword_alpha_default = 0.2
    legacy_keyword_alpha = _get_env("AGENT_RAG_SEARCH_KEYWORD_QUERY_HYBRID_ALPHA")
    if legacy_keyword_alpha is not None:
        try:
            keyword_alpha_default = float(legacy_keyword_alpha)
        except ValueError:
            keyword_alpha_default = 0.2

    return SearchConfig(
        default_hybrid_alpha=_get_env_float("AGENT_RAG_SEARCH_DEFAULT_HYBRID_ALPHA", 0.5),
        keyword_query_hybrid_alpha=_get_env_float(
            "AGENT_RAG_SEARCH_KEYWORD_HYBRID_ALPHA",
            keyword_alpha_default
        ),
        num_results=_get_env_int("AGENT_RAG_SEARCH_NUM_RESULTS", 10),
        max_chunks_per_response=_get_env_int("AGENT_RAG_SEARCH_MAX_CHUNKS_PER_RESPONSE", 15),
        enable_query_expansion=_get_env_bool("AGENT_RAG_SEARCH_ENABLE_QUERY_EXPANSION", True),
        max_expanded_queries=_get_env_int("AGENT_RAG_SEARCH_MAX_EXPANDED_QUERIES", 3),
        enable_document_selection=_get_env_bool("AGENT_RAG_SEARCH_ENABLE_DOCUMENT_SELECTION", True),
        max_documents_to_select=_get_env_int("AGENT_RAG_SEARCH_MAX_DOCUMENTS_TO_SELECT", 10),
        max_chunks_for_relevance=_get_env_int("AGENT_RAG_SEARCH_MAX_CHUNKS_FOR_RELEVANCE", 3),
        enable_context_expansion=_get_env_bool("AGENT_RAG_SEARCH_ENABLE_CONTEXT_EXPANSION", True),
        context_expansion_chunks=_get_env_int("AGENT_RAG_SEARCH_CONTEXT_EXPANSION_CHUNKS", 2),
        max_context_tokens=_get_env_int("AGENT_RAG_SEARCH_MAX_CONTEXT_TOKENS", 4000),
        max_full_document_chunks=_get_env_int("AGENT_RAG_SEARCH_MAX_FULL_DOCUMENT_CHUNKS", 5),
        max_content_chars_per_chunk=_get_env_int("AGENT_RAG_SEARCH_MAX_CONTENT_CHARS_PER_CHUNK", 800),
        enable_reranking=_get_env_bool("AGENT_RAG_SEARCH_ENABLE_RERANKING", False),
        rerank_model=_get_env("AGENT_RAG_SEARCH_RERANK_MODEL"),
        original_query_weight=_get_env_float("AGENT_RAG_SEARCH_ORIGINAL_QUERY_WEIGHT", 0.5),
        llm_semantic_query_weight=_get_env_float("AGENT_RAG_SEARCH_LLM_SEMANTIC_QUERY_WEIGHT", 1.3),
        llm_keyword_query_weight=_get_env_float("AGENT_RAG_SEARCH_LLM_KEYWORD_QUERY_WEIGHT", 1.0),
        rrf_k_value=_get_env_int("AGENT_RAG_SEARCH_RRF_K_VALUE", 50),
        # Concurrency settings
        query_expansion_workers=_get_env_int("AGENT_RAG_SEARCH_QUERY_EXPANSION_WORKERS", 2),
        multi_query_search_workers=_get_env_int("AGENT_RAG_SEARCH_MULTI_QUERY_WORKERS", 10),
        section_expansion_workers=_get_env_int("AGENT_RAG_SEARCH_SECTION_EXPANSION_WORKERS", 5),
    )


def get_deep_research_config_from_env() -> DeepResearchConfig:
    """Create DeepResearchConfig from environment variables."""
    return DeepResearchConfig(
        max_orchestrator_cycles=_get_env_int("AGENT_RAG_DR_MAX_ORCHESTRATOR_CYCLES", 8),
        max_research_cycles=_get_env_int("AGENT_RAG_DR_MAX_RESEARCH_CYCLES", 3),
        max_research_agents=_get_env_int("AGENT_RAG_DR_MAX_RESEARCH_AGENTS", 5),
        num_research_agents=_get_env_int("AGENT_RAG_DR_NUM_RESEARCH_AGENTS", 0) or None,
        max_agent_cycles=_get_env_int("AGENT_RAG_DR_MAX_AGENT_CYCLES", 0) or None,
        skip_clarification=_get_env_bool("AGENT_RAG_DR_SKIP_CLARIFICATION", False),
        enable_think_tool=_get_env_bool("AGENT_RAG_DR_ENABLE_THINK_TOOL", True),
        max_intermediate_report_tokens=_get_env_int("AGENT_RAG_DR_MAX_INTERMEDIATE_REPORT_TOKENS", 10000),
        max_final_report_tokens=_get_env_int("AGENT_RAG_DR_MAX_FINAL_REPORT_TOKENS", 20000),
    )


def get_agent_config_from_env() -> AgentConfig:
    """Create AgentConfig from environment variables."""
    mode_str = _get_env("AGENT_RAG_AGENT_MODE", "chat")
    mode = AgentMode.DEEP_RESEARCH if mode_str == "deep_research" else AgentMode.CHAT

    return AgentConfig(
        mode=mode,
        max_cycles=_get_env_int("AGENT_RAG_AGENT_MAX_CYCLES", 6),
        max_steps=_get_env_int("AGENT_RAG_AGENT_MAX_STEPS", 0) or None,
        max_tokens=_get_env_int("AGENT_RAG_AGENT_MAX_TOKENS", 0) or None,
        enabled_tools=_get_env_list(
            "AGENT_RAG_AGENT_ENABLED_TOOLS",
            ["internal_search", "web_search", "open_url"]
        ),
        enable_citations=_get_env_bool("AGENT_RAG_AGENT_ENABLE_CITATIONS", True),
        deep_research=get_deep_research_config_from_env(),
        search=get_search_config_from_env(),
    )


def get_config_from_env(load_env_file: bool = True) -> AgentRAGConfig:
    """
    Create complete AgentRAGConfig from environment variables.

    Args:
        load_env_file: Whether to load .env file before reading env vars

    Returns:
        Fully configured AgentRAGConfig
    """
    if load_env_file:
        load_dotenv()

    return AgentRAGConfig(
        llm=get_llm_config_from_env(),
        embedding=get_embedding_config_from_env(),
        document_index=get_document_index_config_from_env(),
        agent=get_agent_config_from_env(),
    )


# Convenience function for getting Vespa indexing constants
def get_vespa_indexing_constants_from_env() -> dict[str, int]:
    """Get Vespa indexing constants from environment variables."""
    return {
        "num_threads": _get_env_int("AGENT_RAG_VESPA_NUM_THREADS", 32),
        "batch_size": _get_env_int("AGENT_RAG_VESPA_BATCH_SIZE", 128),
        "max_retry_attempts": _get_env_int("AGENT_RAG_VESPA_MAX_RETRY_ATTEMPTS", 5),
    }


def get_log_config_from_env() -> dict[str, str]:
    """Get logging configuration from environment variables."""
    return {
        "level": _get_env("AGENT_RAG_LOG_LEVEL", "INFO"),
        "format": _get_env("AGENT_RAG_LOG_FORMAT", "text"),
    }


def get_ingestion_config_from_env() -> dict[str, Any]:
    """Get ingestion configuration from environment variables."""
    return {
        # Document parsing limits
        "max_document_chars": _get_env_int("AGENT_RAG_MAX_DOCUMENT_CHARS", 500000),
        "max_document_bytes": _get_env_int("AGENT_RAG_MAX_DOCUMENT_BYTES", 10 * 1024 * 1024),  # 10MB
        "unstructured_api_key": _get_env("AGENT_RAG_UNSTRUCTURED_API_KEY")
        or _get_env("UNSTRUCTURED_API_KEY"),
        # URL fetching
        "url_fetch_timeout": _get_env_int("AGENT_RAG_URL_FETCH_TIMEOUT", 30),
        "url_user_agent": _get_env(
            "AGENT_RAG_URL_USER_AGENT",
            "Mozilla/5.0 (compatible; AgentRAG/1.0)"
        ),
        # OCR LLM
        "ocr_enabled": _get_env_bool("AGENT_RAG_OCR_ENABLED", True),
        "ocr_llm_model": _get_env("AGENT_RAG_OCR_LLM_MODEL", "gpt-4o-mini"),
        "ocr_llm_api_key": _get_env("AGENT_RAG_OCR_LLM_API_KEY"),
        "ocr_llm_api_base": _get_env("AGENT_RAG_OCR_LLM_API_BASE"),
        # Deduplication
        "dedup_reprocess_failed": _get_env_bool("AGENT_RAG_DEDUP_REPROCESS_FAILED", True),
        "dedup_cross_tenant": _get_env_bool("AGENT_RAG_DEDUP_CROSS_TENANT", False),
        # MinIO storage configuration
        "minio_endpoint": _get_env("MINIO_ENDPOINT", "localhost:9000"),
        "minio_access_key": _get_env("MINIO_ACCESS_KEY", "minioadmin"),
        "minio_secret_key": _get_env("MINIO_SECRET_KEY", "minioadmin"),
        "minio_ingestion_bucket": _get_env("MINIO_BUCKET", "agent-rag-ingestion"),
        "minio_secure": _get_env_bool("MINIO_SECURE", False),
    }


# Convenience class for ingestion config access
class IngestionEnvConfig:
    """Environment-based ingestion configuration singleton."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = get_ingestion_config_from_env()
        return cls._instance

    @property
    def max_document_chars(self) -> int:
        return self._config["max_document_chars"]

    @property
    def max_document_bytes(self) -> int:
        return self._config["max_document_bytes"]

    @property
    def url_fetch_timeout(self) -> int:
        return self._config["url_fetch_timeout"]

    @property
    def url_user_agent(self) -> str:
        return self._config["url_user_agent"]

    @property
    def ocr_enabled(self) -> bool:
        return self._config["ocr_enabled"]

    @property
    def unstructured_api_key(self) -> Optional[str]:
        return self._config["unstructured_api_key"]

    @property
    def ocr_llm_model(self) -> str:
        return self._config["ocr_llm_model"]

    @property
    def ocr_llm_api_key(self) -> Optional[str]:
        return self._config["ocr_llm_api_key"]

    @property
    def ocr_llm_api_base(self) -> Optional[str]:
        return self._config["ocr_llm_api_base"]

    @property
    def dedup_reprocess_failed(self) -> bool:
        return self._config["dedup_reprocess_failed"]

    @property
    def dedup_cross_tenant(self) -> bool:
        return self._config["dedup_cross_tenant"]

    @property
    def minio_endpoint(self) -> str:
        return self._config["minio_endpoint"]

    @property
    def minio_access_key(self) -> str:
        return self._config["minio_access_key"]

    @property
    def minio_secret_key(self) -> str:
        return self._config["minio_secret_key"]

    @property
    def minio_ingestion_bucket(self) -> str:
        return self._config["minio_ingestion_bucket"]

    @property
    def minio_secure(self) -> bool:
        return self._config["minio_secure"]


def get_database_config_from_env() -> dict[str, Any]:
    """Get database configuration from environment variables.
    
    Now using external database configuration (AGENT_RAG_API_EXTERNAL_DB_*).
    """
    return {
        "postgres_host": _get_env("AGENT_RAG_API_EXTERNAL_DB_HOST", "localhost"),
        "postgres_port": _get_env_int("AGENT_RAG_API_EXTERNAL_DB_PORT", 5432),
        "postgres_user": _get_env("AGENT_RAG_API_EXTERNAL_DB_USERNAME", "postgres"),
        "postgres_password": _get_env("AGENT_RAG_API_EXTERNAL_DB_PASSWORD", "password"),
        "postgres_db": _get_env("AGENT_RAG_API_EXTERNAL_DB_DATABASE", "agent_rag"),
        # Connection pool settings
        "pool_size": _get_env_int("AGENT_RAG_DB_POOL_SIZE", 10),
        "max_overflow": _get_env_int("AGENT_RAG_DB_MAX_OVERFLOW", 20),
        "pool_timeout": _get_env_int("AGENT_RAG_DB_POOL_TIMEOUT", 30),
        "pool_recycle": _get_env_int("AGENT_RAG_DB_POOL_RECYCLE", 3600),
        "echo": _get_env_bool("AGENT_RAG_DB_ECHO", False),
    }


class DatabaseEnvConfig:
    """Environment-based database configuration singleton."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = get_database_config_from_env()
        return cls._instance

    @property
    def postgres_host(self) -> str:
        return self._config["postgres_host"]

    @property
    def postgres_port(self) -> int:
        return self._config["postgres_port"]

    @property
    def postgres_user(self) -> str:
        return self._config["postgres_user"]

    @property
    def postgres_password(self) -> str:
        return self._config["postgres_password"]

    @property
    def postgres_db(self) -> str:
        return self._config["postgres_db"]

    @property
    def pool_size(self) -> int:
        return self._config["pool_size"]

    @property
    def max_overflow(self) -> int:
        return self._config["max_overflow"]

    @property
    def pool_timeout(self) -> int:
        return self._config["pool_timeout"]

    @property
    def pool_recycle(self) -> int:
        return self._config["pool_recycle"]

    @property
    def echo(self) -> bool:
        return self._config["echo"]

    @property
    def async_database_url(self) -> str:
        """Build async PostgreSQL URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Build sync PostgreSQL URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Global config instances
ingestion_config = IngestionEnvConfig()
database_config = DatabaseEnvConfig()
vespa_schema_config = get_vespa_schema_config_from_env()
vespa_config = get_document_index_config_from_env()
