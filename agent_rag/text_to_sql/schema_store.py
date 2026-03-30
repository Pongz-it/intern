"""Schema vector store for Text-to-SQL.

Uses the existing embedding infrastructure to store and retrieve
database schema elements by semantic similarity.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

from agent_rag.embedding.interface import Embedder
from agent_rag.text_to_sql.models import (
    DatabaseSchema,
    SchemaEmbedding,
)

logger = logging.getLogger(__name__)


class SchemaVectorStore:
    """Vector store for database schema elements.

    Stores table descriptions, column descriptions, and relationships
    as embeddings for semantic similarity search.
    """

    def __init__(
        self,
        embedder: Embedder,
        persist_path: Optional[str] = None,
    ):
        """Initialize the schema vector store.

        Args:
            embedder: Embedder instance for generating vectors
            persist_path: Optional path to persist the store
        """
        self.embedder = embedder
        self.persist_path = persist_path
        self._table_embeddings: dict[str, SchemaEmbedding] = {}
        self._column_embeddings: dict[str, SchemaEmbedding] = {}
        self._relationship_embeddings: dict[str, SchemaEmbedding] = {}
        self._schema_cache: dict[str, DatabaseSchema] = {}
        self._last_refresh: Optional[datetime] = None

        if persist_path:
            self._load_from_disk()

    def _get_persist_dir(self) -> Path:
        """Get the persistence directory."""
        if self.persist_path:
            path = Path(self.persist_path)
        else:
            path = Path("./data/schema_store")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _encode_text(self, text: str) -> list:
        """Encode a single text using the embedder.
        
        Handles different embedder implementations.
        """
        # Try embed_batch first (single item in list)
        if hasattr(self.embedder, 'embed_batch'):
            result = self.embedder.embed_batch([text])
            if result:
                return result[0]
        
        # Fallback to embed method
        if hasattr(self.embedder, 'embed'):
            return self.embedder.embed(text)
        
        # Fallback to encode method (legacy)
        if hasattr(self.embedder, 'encode'):
            result = self.embedder.encode([text])
            if isinstance(result, list) and len(result) > 0:
                return result[0] if isinstance(result[0], list) else result[0]
            return result
        
        raise ValueError(f"Embedder {type(self.embedder)} has no known encoding method")

    def _load_from_disk(self) -> None:
        """Load embeddings from disk."""
        try:
            persist_dir = self._get_persist_dir()

            tables_file = persist_dir / "tables.pkl"
            if tables_file.exists():
                with open(tables_file, "rb") as f:
                    self._table_embeddings = pickle.load(f)

            columns_file = persist_dir / "columns.pkl"
            if columns_file.exists():
                with open(columns_file, "rb") as f:
                    self._column_embeddings = pickle.load(f)

            relationships_file = persist_dir / "relationships.pkl"
            if relationships_file.exists():
                with open(relationships_file, "rb") as f:
                    self._relationship_embeddings = pickle.load(f)

            schema_file = persist_dir / "schemas.pkl"
            if schema_file.exists():
                with open(schema_file, "rb") as f:
                    self._schema_cache = pickle.load(f)

            logger.info(f"Loaded schema store from {persist_dir}")
        except Exception as e:
            logger.warning(f"Failed to load schema store: {e}")

    def _save_to_disk(self) -> None:
        """Save embeddings to disk."""
        if not self.persist_path:
            return

        try:
            persist_dir = self._get_persist_dir()

            with open(persist_dir / "tables.pkl", "wb") as f:
                pickle.dump(self._table_embeddings, f)

            with open(persist_dir / "columns.pkl", "wb") as f:
                pickle.dump(self._column_embeddings, f)

            with open(persist_dir / "relationships.pkl", "wb") as f:
                pickle.dump(self._relationship_embeddings, f)

            with open(persist_dir / "schemas.pkl", "wb") as f:
                pickle.dump(self._schema_cache, f)

            logger.info(f"Saved schema store to {persist_dir}")
        except Exception as e:
            logger.warning(f"Failed to save schema store: {e}")

    async def index_schema(self, schema: DatabaseSchema) -> None:
        """Index a database schema.

        Generates embeddings for all tables, columns, and relationships.
        """
        logger.info(f"Indexing schema with {len(schema.tables)} tables")

        for table in schema.tables:
            await self._index_table(table)

        for table in schema.tables:
            for rel in table.relationships:
                await self._index_relationship(table.name, rel)

        self._schema_cache[schema.version] = schema
        self._last_refresh = datetime.now()
        self._save_to_disk()

        logger.info(f"Indexed {len(self._table_embeddings)} tables, "
                   f"{len(self._column_embeddings)} columns")

    async def _index_table(self, table) -> None:
        """Index a single table."""
        description = self._build_table_description(table)
        embedding = self._encode_text(description)

        schema_emb = SchemaEmbedding(
            element_type="table",
            element_name=table.name,
            description=description,
            embedding=embedding.tolist() if hasattr(embedding, "tolist") else list(embedding),
            table_name=table.name,
        )
        self._table_embeddings[table.name] = schema_emb

        for column in table.columns:
            await self._index_column(table.name, column)

    async def _index_column(self, table_name: str, column) -> None:
        """Index a single column."""
        description = self._build_column_description(table_name, column)
        embedding = self._encode_text(description)

        key = f"{table_name}.{column.name}"
        schema_emb = SchemaEmbedding(
            element_type="column",
            element_name=column.name,
            description=description,
            embedding=embedding.tolist() if hasattr(embedding, "tolist") else list(embedding),
            table_name=table_name,
            column_name=column.name,
        )
        self._column_embeddings[key] = schema_emb

    async def _index_relationship(self, table_name: str, relationship) -> None:
        """Index a table relationship."""
        description = (
            f"Relationship: {relationship.from_table}.{relationship.from_column} "
            f"references {relationship.to_table}.{relationship.to_column}"
        )
        embedding = self._encode_text(description)

        key = f"{relationship.from_table}.{relationship.from_column}"
        schema_emb = SchemaEmbedding(
            element_type="relationship",
            element_name=key,
            description=description,
            embedding=embedding.tolist() if hasattr(embedding, "tolist") else list(embedding),
            table_name=relationship.from_table,
        )
        self._relationship_embeddings[key] = schema_emb

    def _build_table_description(self, table) -> str:
        """Build a natural language description of a table."""
        parts = [f"Table: {table.name}"]

        if table.description:
            parts.append(table.description)

        cols = []
        for col in table.columns:
            col_desc = col.name
            if col.is_primary_key:
                col_desc += " (primary key)"
            if col.is_foreign_key:
                col_desc += f" -> {col.foreign_table}"
            cols.append(col_desc)

        if cols:
            parts.append(f"Columns: {', '.join(cols)}")

        if table.sample_queries:
            parts.append(f"Example queries: {'; '.join(table.sample_queries)}")

        return ". ".join(parts)

    def _build_column_description(self, table_name: str, column) -> str:
        """Build a natural language description of a column."""
        parts = [f"Column: {table_name}.{column.name}"]

        if column.description:
            parts.append(column.description)

        parts.append(f"Type: {column.column_type.value}")

        if column.is_primary_key:
            parts.append("Primary key")
        if column.is_foreign_key:
            parts.append(f"Foreign key to {column.foreign_table}.{column.foreign_column}")
        if not column.is_nullable:
            parts.append("Required (not null)")
        if column.sample_values:
            sample_str = ", ".join(str(v) for v in column.sample_values[:3])
            parts.append(f"Sample values: {sample_str}")

        if column.synonyms:
            parts.append(f"Also called: {', '.join(column.synonyms)}")

        return ". ".join(parts)

    async def search_tables(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Search for tables matching a query."""
        if not self._table_embeddings:
            return []

        query_emb = self._encode_text(query)

        scores = []
        for key, schema_emb in self._table_embeddings.items():
            score = self._cosine_similarity(
                query_emb,
                schema_emb.embedding,
            )
            scores.append((key, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    async def search_columns(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, str, float]]:
        """Search for columns matching a query."""
        if not self._column_embeddings:
            return []

        query_emb = self._encode_text(query)

        scores = []
        for key, schema_emb in self._column_embeddings.items():
            score = self._cosine_similarity(
                query_emb,
                schema_emb.embedding,
            )
            table_name = schema_emb.table_name
            column_name = schema_emb.column_name
            scores.append((table_name, column_name, score))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]

    async def search_relationships(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[str, str, str, float]]:
        """Search for relationships matching a query."""
        if not self._relationship_embeddings:
            return []

        query_emb = self._encode_text(query)

        scores = []
        for key, schema_emb in self._relationship_embeddings.items():
            score = self._cosine_similarity(
                query_emb,
                schema_emb.embedding,
            )
            parts = key.split(".")
            from_table = parts[0] if len(parts) > 1 else ""
            from_column = parts[1] if len(parts) > 1 else key
            scores.append((from_table, from_column, schema_emb.description, score))

        scores.sort(key=lambda x: x[3], reverse=True)
        return scores[:top_k]

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        if not vec1 or not vec2:
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def get_schema(self, version: str = "1.0") -> Optional[DatabaseSchema]:
        """Get a cached schema by version."""
        return self._schema_cache.get(version)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._table_embeddings.clear()
        self._column_embeddings.clear()
        self._relationship_embeddings.clear()
        self._schema_cache.clear()
        self._last_refresh = None

        if self.persist_path:
            persist_dir = self._get_persist_dir()
            for f in persist_dir.glob("*.pkl"):
                f.unlink()
