"""Vespa schema configuration and template rendering."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader

from agent_rag.document_index.vespa.app_config import SCHEMAS_DIR, TEMPLATES_DIR


@dataclass
class VespaSchemaConfig:
    """Configuration for Vespa schema generation."""

    # Schema identity
    schema_name: str = "agent_rag_chunk"

    # Embedding configuration
    dim: int = 1536  # Embedding dimension
    embedding_precision: str = "float"  # float or bfloat16

    # Feature toggles
    multi_tenant: bool = False
    enable_title_embedding: bool = True
    enable_large_chunks: bool = True
    enable_knowledge_graph: bool = True
    enable_access_control: bool = False

    # Ranking configuration
    default_decay_factor: float = 0.5  # Time decay factor for recency bias
    rerank_count: int = 1000  # Number of hits to rerank in global phase

    # Service configuration
    redundancy: int = 1
    searchable_copies: int = 1
    search_threads: int = 4
    summary_threads: int = 2

    def to_template_vars(self) -> dict[str, Any]:
        """Convert config to template variables."""
        return {
            "schema_name": self.schema_name,
            "dim": self.dim,
            "embedding_precision": self.embedding_precision,
            "multi_tenant": self.multi_tenant,
            "enable_title_embedding": self.enable_title_embedding,
            "enable_large_chunks": self.enable_large_chunks,
            "enable_knowledge_graph": self.enable_knowledge_graph,
            "enable_access_control": self.enable_access_control,
            "default_decay_factor": self.default_decay_factor,
            "rerank_count": self.rerank_count,
            "redundancy": self.redundancy,
            "searchable_copies": self.searchable_copies,
            "search_threads": self.search_threads,
            "summary_threads": self.summary_threads,
        }


class VespaSchemaRenderer:
    """Renders Vespa schema templates using Jinja2."""

    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self.schemas_dir = self.templates_dir / "schemas"

        # Initialize Jinja2 environment
        self._env = Environment(
            loader=FileSystemLoader([str(self.templates_dir), str(self.schemas_dir)]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render_schema(
        self,
        config: VespaSchemaConfig,
        template_name: str = "agent_rag_chunk.sd.jinja",
    ) -> str:
        """Render schema template with configuration."""
        template = self._env.get_template(template_name)
        return template.render(**config.to_template_vars())

    def render_services(
        self,
        config: VespaSchemaConfig,
        template_name: str = "services.xml.jinja",
    ) -> str:
        """Render services.xml template."""
        template = self._env.get_template(template_name)
        return template.render(**config.to_template_vars())

    def generate_application_package(
        self,
        config: VespaSchemaConfig,
        output_dir: Path,
    ) -> dict[str, Path]:
        """
        Generate complete Vespa application package.

        Returns:
            Dict mapping file type to output path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        schemas_dir = output_dir / "schemas"
        schemas_dir.mkdir(exist_ok=True)

        # Render and write schema
        schema_content = self.render_schema(config)
        schema_path = schemas_dir / f"{config.schema_name}.sd"
        schema_path.write_text(schema_content)

        # Render and write services.xml
        services_content = self.render_services(config)
        services_path = output_dir / "services.xml"
        services_path.write_text(services_content)

        return {
            "schema": schema_path,
            "services": services_path,
        }


# Predefined configurations for common use cases
SCHEMA_PRESETS: dict[str, VespaSchemaConfig] = {
    "minimal": VespaSchemaConfig(
        enable_title_embedding=False,
        enable_large_chunks=False,
        enable_knowledge_graph=False,
        enable_access_control=False,
    ),
    "standard": VespaSchemaConfig(
        enable_title_embedding=True,
        enable_large_chunks=True,
        enable_knowledge_graph=False,
        enable_access_control=False,
    ),
    "enterprise": VespaSchemaConfig(
        multi_tenant=True,
        enable_title_embedding=True,
        enable_large_chunks=True,
        enable_knowledge_graph=True,
        enable_access_control=True,
        rerank_count=2000,
    ),
}


def get_schema_preset(name: str) -> VespaSchemaConfig:
    """Get a predefined schema configuration."""
    if name not in SCHEMA_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(SCHEMA_PRESETS.keys())}")
    return SCHEMA_PRESETS[name]
