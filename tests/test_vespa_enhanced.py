"""Tests for enhanced Vespa features."""

import pytest
from pathlib import Path
import tempfile

from agent_rag.core.models import Chunk, KGRelationship, SearchFilters
from agent_rag.document_index.vespa.schema_config import (
    VespaSchemaConfig,
    VespaSchemaRenderer,
    SCHEMA_PRESETS,
    get_schema_preset,
)


class TestVespaSchemaConfig:
    """Tests for VespaSchemaConfig."""

    def test_default_config(self):
        config = VespaSchemaConfig()
        assert config.schema_name == "agent_rag_chunk"
        assert config.dim == 1536
        assert config.embedding_precision == "float"
        assert config.enable_title_embedding is True
        assert config.enable_knowledge_graph is True

    def test_custom_config(self):
        config = VespaSchemaConfig(
            schema_name="custom_chunk",
            dim=768,
            multi_tenant=True,
            enable_access_control=True,
        )
        assert config.schema_name == "custom_chunk"
        assert config.dim == 768
        assert config.multi_tenant is True
        assert config.enable_access_control is True

    def test_to_template_vars(self):
        config = VespaSchemaConfig(dim=1024)
        vars = config.to_template_vars()
        assert vars["dim"] == 1024
        assert "schema_name" in vars
        assert "enable_title_embedding" in vars


class TestSchemaPresets:
    """Tests for schema presets."""

    def test_minimal_preset(self):
        config = get_schema_preset("minimal")
        assert config.enable_title_embedding is False
        assert config.enable_knowledge_graph is False
        assert config.enable_access_control is False

    def test_standard_preset(self):
        config = get_schema_preset("standard")
        assert config.enable_title_embedding is True
        assert config.enable_large_chunks is True
        assert config.enable_knowledge_graph is False

    def test_enterprise_preset(self):
        config = get_schema_preset("enterprise")
        assert config.multi_tenant is True
        assert config.enable_knowledge_graph is True
        assert config.enable_access_control is True
        assert config.rerank_count == 2000

    def test_invalid_preset(self):
        with pytest.raises(ValueError):
            get_schema_preset("nonexistent")


class TestVespaSchemaRenderer:
    """Tests for VespaSchemaRenderer."""

    def test_render_schema_basic(self):
        renderer = VespaSchemaRenderer()
        config = VespaSchemaConfig(
            schema_name="test_chunk",
            dim=768,
            enable_title_embedding=False,
            enable_knowledge_graph=False,
        )
        schema = renderer.render_schema(config)

        assert "schema test_chunk" in schema
        assert "document test_chunk" in schema
        assert "field document_id" in schema
        assert "field content" in schema
        assert "field embeddings" in schema

    def test_render_schema_with_title_embedding(self):
        renderer = VespaSchemaRenderer()
        config = VespaSchemaConfig(
            enable_title_embedding=True,
            dim=1536,
        )
        schema = renderer.render_schema(config)

        assert "field title_embedding" in schema
        assert "tensor<float>(x[1536])" in schema

    def test_render_schema_with_kg(self):
        renderer = VespaSchemaRenderer()
        config = VespaSchemaConfig(enable_knowledge_graph=True)
        schema = renderer.render_schema(config)

        assert "field kg_entities" in schema
        assert "field kg_relationships" in schema
        assert "field kg_terms" in schema
        assert "struct kg_relationship" in schema

    def test_render_schema_multi_tenant(self):
        renderer = VespaSchemaRenderer()
        config = VespaSchemaConfig(multi_tenant=True)
        schema = renderer.render_schema(config)

        assert "field tenant_id" in schema

    def test_render_services(self):
        renderer = VespaSchemaRenderer()
        config = VespaSchemaConfig(
            schema_name="test_chunk",
            redundancy=2,
            search_threads=8,
        )
        services = renderer.render_services(config)

        assert "<content id=" in services
        assert "test_chunk" in services
        assert "<redundancy>2</redundancy>" in services
        assert "<search>8</search>" in services

    def test_generate_application_package(self):
        renderer = VespaSchemaRenderer()
        config = VespaSchemaConfig(schema_name="test_app")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "vespa_app"
            result = renderer.generate_application_package(config, output_dir)

            assert "schema" in result
            assert "services" in result
            assert result["schema"].exists()
            assert result["services"].exists()

            # Verify content
            schema_content = result["schema"].read_text()
            assert "schema test_app" in schema_content

            services_content = result["services"].read_text()
            assert "test_app" in services_content


class TestEnhancedChunk:
    """Tests for enhanced Chunk model."""

    def test_chunk_with_kg_fields(self):
        chunk = Chunk(
            document_id="doc1",
            chunk_id=0,
            content="Test content",
            kg_entities=["Entity1", "Entity2"],
            kg_terms=["term1", "term2"],
            kg_relationships=[
                KGRelationship(source="A", rel_type="relates_to", target="B"),
            ],
        )

        assert len(chunk.kg_entities) == 2
        assert len(chunk.kg_relationships) == 1
        assert chunk.kg_relationships[0].source == "A"

    def test_chunk_with_boost(self):
        chunk = Chunk(
            document_id="doc1",
            chunk_id=0,
            content="Test",
            boost=1.5,
            aggregated_chunk_boost_factor=0.8,
        )

        assert chunk.boost == 1.5
        assert chunk.aggregated_chunk_boost_factor == 0.8

    def test_chunk_with_multi_embedding(self):
        chunk = Chunk(
            document_id="doc1",
            chunk_id=0,
            content="Test",
            title_embedding=[0.1, 0.2, 0.3],
            embeddings={"0": [0.1, 0.2], "1": [0.3, 0.4]},
        )

        assert chunk.title_embedding is not None
        assert len(chunk.embeddings) == 2

    def test_chunk_with_rag_context(self):
        chunk = Chunk(
            document_id="doc1",
            chunk_id=0,
            content="Main content",
            doc_summary="Document summary",
            chunk_context="Surrounding context",
            metadata_suffix="Author: John, Date: 2024",
        )

        assert chunk.doc_summary == "Document summary"
        assert chunk.chunk_context == "Surrounding context"
        assert chunk.metadata_suffix == "Author: John, Date: 2024"

    def test_chunk_with_large_chunk_refs(self):
        chunk = Chunk(
            document_id="doc1",
            chunk_id=0,
            content="Large chunk content",
            large_chunk_reference_ids=[1, 2, 3, 4, 5],
        )

        assert len(chunk.large_chunk_reference_ids) == 5

    def test_chunk_to_vespa_fields(self):
        chunk = Chunk(
            document_id="doc1",
            chunk_id=0,
            content="Test content",
            title="Test Title",
            source_type="confluence",
            boost=0.5,
            kg_entities=["Entity1"],
            metadata_suffix="extra info",
        )

        fields = chunk.to_vespa_fields()

        assert fields["document_id"] == "doc1"
        assert fields["chunk_id"] == 0
        assert fields["content"] == "Test content"
        assert fields["title"] == "Test Title"
        assert fields["source_type"] == "confluence"
        assert fields["boost"] == 0.5
        assert fields["kg_entities"] == ["Entity1"]
        assert fields["metadata_suffix"] == "extra info"

    def test_chunk_to_vespa_fields_with_embeddings(self):
        chunk = Chunk(
            document_id="doc1",
            chunk_id=0,
            content="Test",
            embedding=[0.1, 0.2, 0.3],
            title_embedding=[0.4, 0.5, 0.6],
        )

        fields = chunk.to_vespa_fields()

        # Check tensor format
        assert "embeddings" in fields
        assert "blocks" in fields["embeddings"]
        assert "title_embedding" in fields
        assert "values" in fields["title_embedding"]


class TestKGRelationship:
    """Tests for KGRelationship model."""

    def test_relationship_creation(self):
        rel = KGRelationship(
            source="Person:John",
            rel_type="WORKS_AT",
            target="Company:Acme",
        )

        assert rel.source == "Person:John"
        assert rel.rel_type == "WORKS_AT"
        assert rel.target == "Company:Acme"

    def test_relationship_to_dict(self):
        rel = KGRelationship(
            source="A",
            rel_type="relates_to",
            target="B",
        )

        d = rel.to_dict()
        assert d["source"] == "A"
        assert d["rel_type"] == "relates_to"
        assert d["target"] == "B"


class TestEnhancedSearchFilters:
    """Tests for enhanced SearchFilters."""

    def test_filters_with_tags(self):
        filters = SearchFilters(
            tags=["important", "reviewed"],
            source_types=["confluence"],
        )

        assert len(filters.tags) == 2
        assert "important" in filters.tags

    def test_filters_with_time_cutoff(self):
        from datetime import datetime

        cutoff = datetime(2024, 1, 1)
        filters = SearchFilters(time_cutoff=cutoff)

        assert filters.time_cutoff == cutoff

    def test_filters_to_dict(self):
        from datetime import datetime

        filters = SearchFilters(
            source_types=["web"],
            tags=["tag1"],
            time_cutoff=datetime(2024, 6, 1),
        )

        d = filters.to_dict()
        assert "source_types" in d
        assert "tags" in d
        assert "time_cutoff" in d


class TestChunkCleanup:
    """Tests for chunk cleanup functionality."""

    def test_cleanup_removes_title_prefix(self):
        from agent_rag.document_index.vespa.enhanced_vespa_index import EnhancedVespaIndex

        index = EnhancedVespaIndex()
        chunk = Chunk(
            document_id="doc1",
            chunk_id=0,
            content="My Title\nThis is the actual content.",
            title="My Title",
        )

        cleaned = index.cleanup_chunks([chunk])
        assert len(cleaned) == 1
        assert cleaned[0].content == "This is the actual content."

    def test_cleanup_removes_metadata_suffix(self):
        from agent_rag.document_index.vespa.enhanced_vespa_index import EnhancedVespaIndex

        index = EnhancedVespaIndex()
        chunk = Chunk(
            document_id="doc1",
            chunk_id=0,
            content="Main content\nAuthor: John",
            metadata_suffix="Author: John",
        )

        cleaned = index.cleanup_chunks([chunk])
        assert len(cleaned) == 1
        assert cleaned[0].content == "Main content"

    def test_cleanup_removes_doc_summary(self):
        from agent_rag.document_index.vespa.enhanced_vespa_index import EnhancedVespaIndex

        index = EnhancedVespaIndex()
        chunk = Chunk(
            document_id="doc1",
            chunk_id=0,
            content="This is a document about X. Main content here.",
            doc_summary="This is a document about X. ",
        )

        cleaned = index.cleanup_chunks([chunk])
        assert len(cleaned) == 1
        assert cleaned[0].content == "Main content here."

    def test_cleanup_removes_chunk_context(self):
        from agent_rag.document_index.vespa.enhanced_vespa_index import EnhancedVespaIndex

        index = EnhancedVespaIndex()
        chunk = Chunk(
            document_id="doc1",
            chunk_id=0,
            content="Main content here. Context about surrounding chunks.",
            chunk_context=" Context about surrounding chunks.",
        )

        cleaned = index.cleanup_chunks([chunk])
        assert len(cleaned) == 1
        assert cleaned[0].content == "Main content here."

    def test_cleanup_all_augmentations(self):
        from agent_rag.document_index.vespa.enhanced_vespa_index import EnhancedVespaIndex

        index = EnhancedVespaIndex()
        chunk = Chunk(
            document_id="doc1",
            chunk_id=0,
            content="My Title\nDoc summary. Main content.\nAuthor: John",
            title="My Title",
            doc_summary="Doc summary. ",
            metadata_suffix="\nAuthor: John",
        )

        cleaned = index.cleanup_chunks([chunk])
        assert len(cleaned) == 1
        # Should remove title, then doc_summary, then metadata_suffix
        # Content becomes: "Doc summary. Main content.\nAuthor: John" -> remove title
        # Then check remaining
        assert "My Title" not in cleaned[0].content
        assert "Author: John" not in cleaned[0].content

    def test_cleanup_preserves_other_fields(self):
        from agent_rag.document_index.vespa.enhanced_vespa_index import EnhancedVespaIndex

        index = EnhancedVespaIndex()
        chunk = Chunk(
            document_id="doc1",
            chunk_id=5,
            content="Title\nContent",
            title="Title",
            source_type="confluence",
            boost=1.5,
            kg_entities=["Entity1"],
            score=0.95,
        )

        cleaned = index.cleanup_chunks([chunk])
        assert len(cleaned) == 1
        assert cleaned[0].document_id == "doc1"
        assert cleaned[0].chunk_id == 5
        assert cleaned[0].source_type == "confluence"
        assert cleaned[0].boost == 1.5
        assert cleaned[0].kg_entities == ["Entity1"]
        assert cleaned[0].score == 0.95

    def test_cleanup_handles_empty_content(self):
        from agent_rag.document_index.vespa.enhanced_vespa_index import EnhancedVespaIndex

        index = EnhancedVespaIndex()
        chunk = Chunk(
            document_id="doc1",
            chunk_id=0,
            content="",
            title="Title",
        )

        cleaned = index.cleanup_chunks([chunk])
        assert len(cleaned) == 1
        assert cleaned[0].content == ""

    def test_cleanup_handles_no_augmentations(self):
        from agent_rag.document_index.vespa.enhanced_vespa_index import EnhancedVespaIndex

        index = EnhancedVespaIndex()
        chunk = Chunk(
            document_id="doc1",
            chunk_id=0,
            content="Pure original content without any augmentation.",
        )

        cleaned = index.cleanup_chunks([chunk])
        assert len(cleaned) == 1
        assert cleaned[0].content == "Pure original content without any augmentation."
