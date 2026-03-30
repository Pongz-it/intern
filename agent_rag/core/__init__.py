"""Backward compatibility imports for optional core modules."""

__all__: list[str] = []

try:
    from agent_rag.core.external_database_config import (
        ExternalDatabaseConfig,
        ExternalPostgresConfig,
    )
    from agent_rag.core.external_database_connector import (
        ExternalDatabaseConnector,
        ExternalDatabaseQueryResult,
        ExternalPostgresConnector,
        ExternalPostgresQueryResult,
    )

    __all__.extend(
        [
            "ExternalDatabaseConfig",
            "ExternalDatabaseConnector",
            "ExternalDatabaseQueryResult",
            "ExternalPostgresConfig",
            "ExternalPostgresConnector",
            "ExternalPostgresQueryResult",
        ]
    )
except ModuleNotFoundError:
    # Database extras are optional for lightweight/test installs.
    pass

try:
    from agent_rag.core.session_memory_models import (
        ConversationMessage,
        ConversationSession,
        MessageType,
        MemoryType,
        UserMemory,
    )
    from agent_rag.core.session_manager import SessionManager
    from agent_rag.core.memory_store import MemoryStore
    from agent_rag.core.memory_extractor import MemoryExtractor, MemoryRetriever

    __all__.extend(
        [
            "ConversationMessage",
            "ConversationSession",
            "MessageType",
            "MemoryType",
            "UserMemory",
            "SessionManager",
            "MemoryStore",
            "MemoryExtractor",
            "MemoryRetriever",
        ]
    )
except ModuleNotFoundError:
    # Session/memory extras depend on optional database/vector packages.
    pass
