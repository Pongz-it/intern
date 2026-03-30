"""Vespa-based document index."""

try:
    from agent_rag.document_index.vespa.vespa_index import VespaIndex
    from agent_rag.document_index.vespa.enhanced_vespa_index import (
        EnhancedVespaIndex,
        IndexingResult,
        VisitResult,
    )
    from agent_rag.document_index.vespa.schema_config import (
        VespaSchemaConfig,
        VespaSchemaRenderer,
        SCHEMA_PRESETS,
        get_schema_preset,
    )

    __all__ = [
        # Index implementations
        "VespaIndex",
        "EnhancedVespaIndex",
        # Indexing utilities
        "IndexingResult",
        "VisitResult",
        # Schema configuration
        "VespaSchemaConfig",
        "VespaSchemaRenderer",
        "SCHEMA_PRESETS",
        "get_schema_preset",
    ]
except ImportError:
    __all__ = []
