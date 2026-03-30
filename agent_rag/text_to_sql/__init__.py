"""Text-to-SQL main module.

Integrates schema extraction, intent analysis, SQL generation,
and database execution into a unified interface.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Any

from agent_rag.embedding.interface import Embedder
from agent_rag.llm.interface import LLM

from agent_rag.text_to_sql.config import TextToSQLConfig
from agent_rag.text_to_sql.external_db_scanner import (
    ExternalDatabaseScanner,
    DBKeywordInjector,
)
from agent_rag.text_to_sql.external_database_adapter import (
    ExternalDatabaseAdapter,
)
from agent_rag.text_to_sql.intent_analyzer import IntentAnalyzer, QueryIntentAnalysis
from agent_rag.text_to_sql.keyword_manager import KeywordManager
from agent_rag.text_to_sql.models import (
    DatabaseSchema,
    KeywordCategory,
    QueryIntent,
    SQLQueryResult,
)
from agent_rag.text_to_sql.schema_store import SchemaVectorStore
from agent_rag.text_to_sql.sql_generator import SQLQueryEngine

logger = logging.getLogger(__name__)


@dataclass
class TextToSQLResult:
    """Result of Text-to-SQL query execution."""

    is_data_query: bool
    confidence: float
    intent: Optional[QueryIntent]
    sql: Optional[str]
    data: list[dict]
    row_count: int
    error_message: Optional[str]
    execution_time_ms: float
    metadata: dict


class TextToSQL:
    """Main Text-to-SQL service.

    Provides a unified interface for:
    - Schema extraction and indexing
    - Intent analysis
    - SQL generation and execution
    """

    def __init__(
        self,
        llm: LLM,
        embedder: Embedder,
        config: Optional[TextToSQLConfig] = None,
        external_connector: Optional[Any] = None,
        enable_db_discovery: bool = True,
    ):
        """Initialize Text-to-SQL service.

        Args:
            llm: LLM instance for query generation
            embedder: Embedder instance for schema indexing
            config: Optional configuration
            external_connector: ExternalDatabaseConnector instance (required)
            enable_db_discovery: Whether to enable external database discovery
        """
        if external_connector is None:
            raise ValueError("external_connector is required")

        self.llm = llm
        self.embedder = embedder
        self.external_connector = external_connector
        self.config = config or TextToSQLConfig.from_env()
        self.enable_db_discovery = enable_db_discovery

        self.adapter = ExternalDatabaseAdapter(connector=external_connector)
        self.schema: Optional[DatabaseSchema] = None
        self.vector_store: Optional[SchemaVectorStore] = None
        self.query_engine: Optional[SQLQueryEngine] = None

        self.keyword_manager = KeywordManager(None, external_connector=external_connector)
        
        self.db_scanner: Optional[ExternalDatabaseScanner] = None
        self.keyword_injector: Optional[DBKeywordInjector] = None
        
        if self.enable_db_discovery and external_connector:
            logger.info("[TextToSQL] Initializing external database scanner...")
            self.db_scanner = ExternalDatabaseScanner(external_connector=external_connector)
        
        if self.db_scanner:
            self.keyword_injector = DBKeywordInjector(self.db_scanner)
            logger.info("[TextToSQL] Database keyword injector initialized")
        
        self.analyzer = IntentAnalyzer(
            llm,
            self.config.intent_confidence_threshold,
            self.keyword_manager,
            db_scanner=self.db_scanner,
        )

    async def initialize(
        self,
        refresh_schema: bool = False,
        schema_version: str = "1.0",
    ) -> None:
        """Initialize the Text-to-SQL service.

        Args:
            refresh_schema: Force refresh schema from database
            schema_version: Version identifier for cached schema
        """
        logger.info("Initializing Text-to-SQL service")

        self.schema = await self.adapter.extract_schema()

        self.vector_store = SchemaVectorStore(
            embedder=self.embedder,
            persist_path="./data/schema_store" if self.config.enable_schema_caching else None,
        )

        if refresh_schema or not self.vector_store.get_schema(schema_version):
            await self.vector_store.index_schema(self.schema)
        else:
            self.schema = self.vector_store.get_schema(schema_version)

        self.query_engine = SQLQueryEngine(
            llm=self.llm,
            schema=self.schema,
            adapter=self.adapter,
            config=self.config,
            external_db_scanner=self.db_scanner,
        )

        await self.keyword_manager.initialize()
        await self.analyzer.load_dynamic_keywords()
        
        if self.db_scanner:
            try:
                logger.info("[TextToSQL] Scanning external database for keyword discovery...")
                discovered_db = await self.db_scanner.scan_database(
                    db_name="external_db",
                    include_sample_data=True,
                    sample_size=3,
                )
                
                all_keywords = self.db_scanner.get_all_keywords()
                logger.info(f"[TextToSQL] Discovered {len(all_keywords)} database keywords")
                
                for keyword in all_keywords:
                    logger.debug(f"  - {keyword}")
                
                if discovered_db and discovered_db.tables:
                    from agent_rag.text_to_sql.models import DatabaseTable

                    logger.info(f"[TextToSQL] Adding {len(discovered_db.tables)} external tables to schema...")

                    for discovered_table in discovered_db.tables:
                        table_exists = any(t.name == discovered_table.name for t in self.schema.tables)
                        if not table_exists:
                            from agent_rag.text_to_sql.models import TableColumn, ColumnType

                            table = DatabaseTable(
                                name=discovered_table.name,
                                description=discovered_table.description or f"External table: {discovered_table.name}",
                            )

                            for col in discovered_table.columns:
                                column_type_map = {
                                    "varchar": ColumnType.VARCHAR,
                                    "text": ColumnType.TEXT,
                                    "char": ColumnType.CHAR,
                                    "integer": ColumnType.INTEGER,
                                    "int": ColumnType.INTEGER,
                                    "int4": ColumnType.INTEGER,
                                    "bigint": ColumnType.BIGINT,
                                    "int8": ColumnType.BIGINT,
                                    "smallint": ColumnType.SMALLINT,
                                    "numeric": ColumnType.NUMERIC,
                                    "decimal": ColumnType.DECIMAL,
                                    "float": ColumnType.FLOAT,
                                    "double precision": ColumnType.DOUBLE,
                                    "boolean": ColumnType.BOOLEAN,
                                    "bool": ColumnType.BOOLEAN,
                                    "date": ColumnType.DATE,
                                    "timestamp": ColumnType.TIMESTAMP,
                                    "timestamptz": ColumnType.TIMESTAMP,
                                    "json": ColumnType.JSON,
                                    "jsonb": ColumnType.JSON,
                                }

                                col_type_lower = (col.column_type or "").lower()
                                column_type = column_type_map.get(col_type_lower, ColumnType.TEXT)

                                column = TableColumn(
                                    name=col.name,
                                    column_type=column_type,
                                    description=col.description or "",
                                    is_primary_key=col.is_primary_key,
                                    is_nullable=col.is_nullable,
                                    default_value=col.default_value,
                                )
                                table.columns.append(column)

                            self.schema.tables.append(table)
                            logger.info(f"[TextToSQL] Added external table: {discovered_table.name} ({len(discovered_table.columns)} columns)")

                    if self.query_engine:
                        self.query_engine.schema = self.schema
                        logger.info(f"[TextToSQL] Updated query engine schema with {len(self.schema.tables)} tables total")

                logger.info("[TextToSQL] External database discovery complete")
            except Exception as e:
                import traceback
                logger.warning(f"[TextToSQL] Failed to scan external database: {e}")
                logger.debug(traceback.format_exc())

    async def execute(
        self,
        query: str,
        auto_index: bool = True,
    ) -> TextToSQLResult:
        """Execute a natural language query.

        Args:
            query: User's natural language query
            auto_index: Auto-index schema if not initialized

        Returns:
            TextToSQLResult with query results
        """
        import time

        start_time = time.time()

        if not self.query_engine:
            if auto_index:
                await self.initialize()
            else:
                return TextToSQLResult(
                    is_data_query=False,
                    confidence=0.0,
                    intent=None,
                    sql=None,
                    data=[],
                    row_count=0,
                    error_message="TextToSQL not initialized",
                    execution_time_ms=0.0,
                    metadata={},
                )

        intent_analysis = await self.analyzer.analyze(query)

        if not intent_analysis.is_data_query:
            return TextToSQLResult(
                is_data_query=False,
                confidence=intent_analysis.confidence,
                intent=None,
                sql=None,
                data=[],
                row_count=0,
                error_message=None,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"reason": "Not a data query"},
            )

        if intent_analysis.confidence < self.config.intent_confidence_threshold:
            return TextToSQLResult(
                is_data_query=True,
                confidence=intent_analysis.confidence,
                intent=intent_analysis.primary_intent,
                sql=None,
                data=[],
                row_count=0,
                error_message=f"Low confidence: {intent_analysis.confidence}",
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"intent_analysis": intent_analysis},
            )

        result = await self.query_engine.execute(
            query,
            intent=intent_analysis.primary_intent,
            suggested_tables=intent_analysis.suggested_tables or None,
            suggested_columns=intent_analysis.suggested_columns or None,
        )

        execution_time = (time.time() - start_time) * 1000

        return TextToSQLResult(
            is_data_query=True,
            confidence=intent_analysis.confidence,
            intent=intent_analysis.primary_intent,
            sql=result.sql,
            data=result.data,
            row_count=result.row_count,
            error_message=result.error_message,
            execution_time_ms=execution_time,
            metadata={
                "intent_analysis": intent_analysis,
                "tables_used": self._extract_tables(result.sql) if result.sql else [],
            },
        )

    async def search_schema(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict:
        """Search schema by semantic similarity.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Dictionary with table and column search results
        """
        if not self.vector_store:
            return {"tables": [], "columns": []}

        table_results = await self.vector_store.search_tables(query, top_k)
        column_results = await self.vector_store.search_columns(query, top_k * 2)

        return {
            "tables": [
                {"name": name, "score": score}
                for name, score in table_results
            ],
            "columns": [
                {"table": table, "column": column, "score": score}
                for table, column, score in column_results
            ],
        }

    async def refresh_schema(self) -> None:
        """Refresh schema from database."""
        await self.initialize(refresh_schema=True)

    def _extract_tables(self, sql: Optional[str]) -> list[str]:
        """Extract table names from SQL."""
        if not sql:
            return []

        import re

        tables = set()
        from_pattern = r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        join_pattern = r"\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)"

        for match in re.finditer(from_pattern, sql, re.IGNORECASE):
            tables.add(match.group(1))

        for match in re.finditer(join_pattern, sql, re.IGNORECASE):
            tables.add(match.group(1))

        return list(tables)

    def get_schema(self) -> Optional[DatabaseSchema]:
        """Get current database schema."""
        return self.schema

    def get_injected_context(self, query: str) -> dict:
        """Get injected context information for a query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with suggested tables, columns, and mappings
        """
        if not self.keyword_injector:
            return {
                "suggested_tables": [],
                "suggested_columns": [],
                "table_column_mapping": {},
                "detected_synonyms": [],
            }
        
        return self.keyword_injector.get_injected_context(query)

    def should_use_hybrid_search(self, query: str) -> bool:
        """Check if hybrid search should be used for the query.
        
        Args:
            query: User query
            
        Returns:
            True if query contains database keywords and hybrid search should be used
        """
        if not self.keyword_injector:
            return False
        
        return self.keyword_injector.should_use_hybrid_search(query)

    def get_discovered_keywords(self) -> list[str]:
        """Get all discovered database keywords.
        
        Returns:
            List of discovered table names, column names, and their synonyms
        """
        if not self.db_scanner:
            return []
        
        return list(self.db_scanner.get_all_keywords())

    async def reload_db_discovery(self) -> None:
        """Reload external database discovery from cache or scan again."""
        if not self.db_scanner:
            return
        
        try:
            cached_db = self.db_scanner.load_discovery("external_db")
            if not cached_db:
                await self.db_scanner.scan_database(
                    db_name="external_db",
                    include_sample_data=True,
                    sample_size=3,
                )
                self.db_scanner.save_discovery("external_db")
            
            all_keywords = self.db_scanner.get_all_keywords()
            logger.info(f"[TextToSQL] Reloaded {len(all_keywords)} database keywords")
        except Exception as e:
            logger.warning(f"[TextToSQL] Failed to reload database discovery: {e}")


async def create_text_to_sql(
    llm: LLM,
    embedder: Embedder,
    config: Optional[TextToSQLConfig] = None,
    external_connector: Optional[Any] = None,
    enable_db_discovery: bool = True,
) -> TextToSQL:
    """Factory function to create and initialize TextToSQL.

    Args:
        llm: LLM instance
        embedder: Embedder instance
        config: Optional configuration
        external_connector: Optional external PostgreSQL connector
        enable_db_discovery: Whether to enable external database discovery

    Returns:
        Initialized TextToSQL instance
    """
    text_to_sql = TextToSQL(
        llm, 
        embedder, 
        config=config, 
        external_connector=external_connector,
        enable_db_discovery=enable_db_discovery,
    )
    await text_to_sql.initialize()
    return text_to_sql


def create_text_to_sql_sync(
    llm: LLM,
    embedder: Embedder,
    config: Optional[TextToSQLConfig] = None,
    external_connector: Optional[Any] = None,
    enable_db_discovery: bool = True,
) -> TextToSQL:
    """Synchronous factory function to create and initialize TextToSQL.

    Args:
        llm: LLM instance
        embedder: Embedder instance
        config: Optional configuration
        external_connector: Optional external PostgreSQL connector
        enable_db_discovery: Whether to enable external database discovery

    Returns:
        Initialized TextToSQL instance
    """
    import asyncio
    import concurrent.futures

    text_to_sql = TextToSQL(
        llm, 
        embedder, 
        config=config, 
        external_connector=external_connector,
        enable_db_discovery=enable_db_discovery,
    )

    def init_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(text_to_sql.initialize())
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(init_in_thread)
        future.result()

    return text_to_sql
