"""Text-to-SQL configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextToSQLConfig:
    """Configuration for Text-to-SQL functionality."""

    enabled: bool = True

    enable_schema_caching: bool = True
    schema_cache_ttl: int = 3600

    max_query_results: int = 1000
    default_limit: int = 100

    enable_intent_analysis: bool = True
    intent_confidence_threshold: float = 0.7

    sql_generation_max_tokens: int = 1024
    sql_generation_temperature: float = 0.0

    allowed_tables: list[str] = field(default_factory=list)
    blocked_tables: list[str] = field(default_factory=list)
    allowed_columns: dict[str, list[str]] = field(default_factory=dict)

    enable_audit_log: bool = True

    @classmethod
    def from_env(cls) -> "TextToSQLConfig":
        """Create config from environment variables."""
        import os
        from agent_rag.core.env_config import _get_env_bool, _get_env_int, _get_env_list

        return cls(
            enabled=_get_env_bool("AGENT_RAG_TEXT_TO_SQL_ENABLED", True),
            enable_schema_caching=_get_env_bool(
                "AGENT_RAG_TEXT_TO_SQL_SCHEMA_CACHING", True
            ),
            schema_cache_ttl=_get_env_int("AGENT_RAG_TEXT_TO_SQL_SCHEMA_CACHE_TTL", 3600),
            max_query_results=_get_env_int("AGENT_RAG_TEXT_TO_SQL_MAX_RESULTS", 1000),
            default_limit=_get_env_int("AGENT_RAG_TEXT_TO_SQL_DEFAULT_LIMIT", 100),
            enable_intent_analysis=_get_env_bool(
                "AGENT_RAG_TEXT_TO_SQL_INTENT_ANALYSIS", True
            ),
            intent_confidence_threshold=_get_env_float(
                "AGENT_RAG_TEXT_TO_SQL_INTENT_CONFIDENCE", 0.7
            ),
            sql_generation_max_tokens=_get_env_int(
                "AGENT_RAG_TEXT_TO_SQL_MAX_TOKENS", 1024
            ),
            sql_generation_temperature=_get_env_float(
                "AGENT_RAG_TEXT_TO_SQL_TEMPERATURE", 0.0
            ),
            allowed_tables=_get_env_list("AGENT_RAG_TEXT_TO_SQL_ALLOWED_TABLES"),
            blocked_tables=_get_env_list("AGENT_RAG_TEXT_TO_SQL_BLOCKED_TABLES"),
            enable_audit_log=_get_env_bool("AGENT_RAG_TEXT_TO_SQL_AUDIT_LOG", True),
        )


def _get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    import os
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default
