"""Unified database adapter for Text-to-SQL.

Uses ExternalDatabaseConnector for all database operations.
Supports multiple database types: PostgreSQL, MySQL, SQLite, etc.
All methods are async, internally using asyncio.to_thread() for sync operations.
"""

import asyncio
import logging
from typing import Any, Optional

from sqlalchemy import text

from agent_rag.text_to_sql.models import (
    ColumnType,
    DatabaseSchema,
    DatabaseTable,
    SQLQueryResult,
    TableColumn,
    TableIndex,
    TableRelationship,
)

logger = logging.getLogger(__name__)


class ExternalDatabaseAdapter:
    """Adapter for database operations using ExternalDatabaseConnector.

    All methods are async but internally execute sync operations via asyncio.to_thread().
    Supports multiple database types through dynamic query generation.
    """

    TYPE_MAPPING = {
        "integer": ColumnType.INTEGER,
        "int": ColumnType.INTEGER,
        "int4": ColumnType.INTEGER,
        "bigint": ColumnType.BIGINT,
        "int8": ColumnType.BIGINT,
        "smallint": ColumnType.SMALLINT,
        "int2": ColumnType.SMALLINT,
        "decimal": ColumnType.DECIMAL,
        "numeric": ColumnType.NUMERIC,
        "dec": ColumnType.DECIMAL,
        "float": ColumnType.FLOAT,
        "float4": ColumnType.REAL,
        "float8": ColumnType.DOUBLE,
        "double precision": ColumnType.DOUBLE,
        "real": ColumnType.REAL,
        "varchar": ColumnType.VARCHAR,
        "character varying": ColumnType.VARCHAR,
        "char": ColumnType.CHAR,
        "character": ColumnType.CHAR,
        "text": ColumnType.TEXT,
        "longtext": ColumnType.TEXT,
        "mediumtext": ColumnType.TEXT,
        "tinytext": ColumnType.TEXT,
        "string": ColumnType.VARCHAR,
        "boolean": ColumnType.BOOLEAN,
        "bool": ColumnType.BOOLEAN,
        "date": ColumnType.DATE,
        "time": ColumnType.TIME,
        "timestamp": ColumnType.TIMESTAMP,
        "timestamptz": ColumnType.TIMESTAMP,
        "datetime": ColumnType.DATETIME,
        "timestamp without time zone": ColumnType.TIMESTAMP,
        "timestamp with time zone": ColumnType.TIMESTAMP,
        "json": ColumnType.JSON,
        "jsonb": ColumnType.JSONB,
        "array": ColumnType.ARRAY,
        "uuid": ColumnType.UUID,
        "bytea": ColumnType.BYTEA,
        "blob": ColumnType.BYTEA,
        "tinyint": ColumnType.INTEGER,
        "mediumint": ColumnType.INTEGER,
        "year": ColumnType.INTEGER,
    }

    def __init__(self, connector: Any):
        """Initialize the database adapter.

        Args:
            connector: ExternalDatabaseConnector instance (sync session, multi-DB).
        """
        self._connector = connector
        self._db_type = getattr(connector, "db_type", None) or getattr(
            getattr(connector, "config", None), "db_type", "postgresql"
        )

    def _map_column_type(self, db_type: str) -> ColumnType:
        """Map database type string to ColumnType enum."""
        base_type = db_type.lower().split("(")[0].strip()
        return self.TYPE_MAPPING.get(base_type, ColumnType.UNKNOWN)

    def _get_table_list_query(self) -> str:
        """Get query to list tables based on database type."""
        config = getattr(self._connector, "config", None) if self._connector else None
        if config and hasattr(config, "get_table_list_query"):
            return config.get_table_list_query()

        queries = {
            "postgresql": """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """,
            "mysql": """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
                ORDER BY table_name
            """,
            "sqlite": """
                SELECT name AS table_name
                FROM sqlite_master
                WHERE type='table'
                AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """,
        }
        return queries.get(self._db_type, queries["postgresql"])

    def _get_table_schema_query(self) -> str:
        """Get query to get table schema based on database type."""
        config = getattr(self._connector, "config", None) if self._connector else None
        if config and hasattr(config, "get_table_schema_query"):
            return config.get_table_schema_query()

        queries = {
            "postgresql": """
                SELECT
                    c.column_name,
                    c.data_type,
                    c.column_default,
                    c.is_nullable,
                    c.ordinal_position
                FROM information_schema.columns c
                WHERE c.table_schema = 'public' AND c.table_name = :table_name
                ORDER BY c.ordinal_position
            """,
            "mysql": """
                SELECT
                    COLUMN_NAME as column_name,
                    DATA_TYPE as data_type,
                    IS_NULLABLE as is_nullable,
                    COLUMN_DEFAULT as column_default,
                    ORDINAL_POSITION as ordinal_position
                FROM information_schema.columns
                WHERE table_schema = DATABASE() AND table_name = :table_name
                ORDER BY ORDINAL_POSITION
            """,
            "sqlite": """
                PRAGMA table_info(:table_name)
            """,
        }
        return queries.get(self._db_type, queries["postgresql"])

    async def extract_schema(
        self, include_tables: Optional[list[str]] = None
    ) -> DatabaseSchema:
        """Extract database schema information (async wrapper for sync operation)."""
        return await asyncio.to_thread(self._extract_schema_sync, include_tables)

    def _extract_schema_sync(
        self, include_tables: Optional[list[str]] = None
    ) -> DatabaseSchema:
        """Extract schema (sync operation)."""
        schema = DatabaseSchema()
        query = self._get_table_list_query()
        with self._connector.session() as session:
            result = session.execute(text(query))
            tables = result.fetchall()
        for table_name, in tables:
            if include_tables and table_name not in include_tables:
                continue
            table = self._extract_table_schema(table_name, "")
            schema.tables.append(table)
        return schema

    def _extract_table_schema(
        self, table_name: str, description: str
    ) -> DatabaseTable:
        """Extract schema for a single table (sync operation)."""
        table = DatabaseTable(name=table_name, description=description)
        query = self._get_table_schema_query()

        with self._connector.session() as session:
            result = session.execute(text(query), {"table_name": table_name})
            columns = result.fetchall()

        if self._db_type == "sqlite":
            for col in columns:
                col_name = col[1]
                data_type = col[2]
                nullable = not bool(col[3])
                column = TableColumn(
                    name=col_name,
                    column_type=self._map_column_type(data_type),
                    description="",
                    is_nullable=nullable,
                )
                table.columns.append(column)
        elif self._db_type == "postgresql":
            for col in columns:
                col_name, data_type, default, nullable, position = col
                column = TableColumn(
                    name=col_name,
                    column_type=self._map_column_type(data_type),
                    description="",
                    is_nullable=(nullable == "YES"),
                    default_value=default,
                )
                table.columns.append(column)
            self._enrich_postgresql_table(table)
        else:
            for col in columns:
                col_name, data_type, nullable, default, position = col
                column = TableColumn(
                    name=col_name,
                    column_type=self._map_column_type(data_type),
                    description="",
                    is_nullable=(nullable == "YES"),
                    default_value=default,
                )
                table.columns.append(column)

        return table

    def _enrich_postgresql_table(self, table: DatabaseTable) -> None:
        """Enrich table with PostgreSQL-specific: PK, FK, indexes, sample data."""
        table_name = table.name
        try:
            self._extract_postgresql_primary_key(table)
            self._extract_postgresql_foreign_keys(table)
            self._extract_postgresql_indexes(table)
            self._extract_sample_data(table)
        except Exception as e:
            logger.debug(f"PostgreSQL enrichment for {table_name} skipped: {e}")

    def _extract_postgresql_primary_key(self, table: DatabaseTable) -> None:
        """Extract primary key using pg_index (PostgreSQL only)."""
        pk_query = """
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = :table_name::regclass AND i.indisprimary AND a.attnum > 0
        """
        with self._connector.session() as session:
            result = session.execute(text(pk_query), {"table_name": table.name})
            pk_cols = result.fetchall()
            if pk_cols:
                table.primary_key = pk_cols[0][0]
                for col in table.columns:
                    if col.name == table.primary_key:
                        col.is_primary_key = True
                        break

    def _extract_postgresql_foreign_keys(self, table: DatabaseTable) -> None:
        """Extract foreign keys from information_schema (PostgreSQL only)."""
        fk_query = """
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = 'public' AND tc.table_name = :table_name
        """
        with self._connector.session() as session:
            result = session.execute(text(fk_query), {"table_name": table.name})
            for row in result.fetchall():
                col_name, foreign_table, foreign_col = row
                table.relationships.append(TableRelationship(
                    from_table=table.name,
                    from_column=col_name,
                    to_table=foreign_table,
                    to_column=foreign_col,
                    description=f"{table.name}.{col_name} -> {foreign_table}.{foreign_col}",
                ))
                for col in table.columns:
                    if col.name == col_name:
                        col.is_foreign_key = True
                        col.foreign_table = foreign_table
                        col.foreign_column = foreign_col
                        break

    def _extract_postgresql_indexes(self, table: DatabaseTable) -> None:
        """Extract indexes from pg_indexes (PostgreSQL only)."""
        index_query = """
            SELECT indexname, indexdef, (indexdef LIKE '%UNIQUE%') AS is_unique
            FROM pg_indexes
            WHERE schemaname = 'public' AND tablename = :table_name
        """
        with self._connector.session() as session:
            result = session.execute(text(index_query), {"table_name": table.name})
            for idx_name, idx_def, is_unique in result.fetchall():
                col_match = idx_def.find("(")
                if col_match != -1:
                    cols_str = idx_def[col_match + 1 : idx_def.rindex(")")]
                    idx_cols = [c.strip() for c in cols_str.split(",")]
                    table.indexes.append(TableIndex(
                        name=idx_name,
                        columns=idx_cols,
                        is_unique=bool(is_unique),
                        description=idx_def,
                    ))

    def _extract_sample_data(self, table: DatabaseTable) -> None:
        """Extract sample rows (generic SQL, works for any DB)."""
        try:
            with self._connector.session() as session:
                result = session.execute(text(f"SELECT * FROM {table.name} LIMIT 5"))
                rows = result.fetchall()
                keys = list(result.keys()) if result.keys() else []
                if rows and keys:
                    for col in table.columns:
                        if col.name not in keys:
                            continue
                        for row in rows:
                            idx = keys.index(col.name)
                            value = row[idx]
                            if value is not None and len(col.sample_values) < 10:
                                col.sample_values.append(str(value))
        except Exception:
            pass

    async def execute_query(
        self, sql: str, params: Optional[dict[str, Any]] = None
    ) -> SQLQueryResult:
        """Execute a SQL query and return results (async wrapper for sync operation)."""
        return await asyncio.to_thread(self._execute_query_sync, sql, params)

    def _execute_query_sync(
        self, sql: str, params: Optional[dict[str, Any]] = None
    ) -> SQLQueryResult:
        """Execute SQL (sync operation)."""
        import time
        start_time = time.time()
        try:
            with self._connector.session() as session:
                result = session.execute(text(sql), params or {})
                rows = result.fetchall()
                columns = result.keys()
                data = []
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        row_dict[col] = value.isoformat() if hasattr(value, "isoformat") else value
                    data.append(row_dict)
                return SQLQueryResult(
                    sql=sql,
                    success=True,
                    row_count=len(data),
                    data=data,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
        except Exception as e:
            logger.error("SQL execution failed: %s", e)
            return SQLQueryResult(
                sql=sql,
                success=False,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _get_row_count_sync(self, table_name: str) -> Optional[int]:
        """Get row count (sync path)."""
        if self._db_type == "sqlite":
            try:
                safe_name = '"' + table_name.replace('"', '""') + '"'
                with self._connector.session() as session:
                    result = session.execute(text(f"SELECT COUNT(*) FROM {safe_name}"))
                    row = result.fetchone()
                    return row[0] if row else None
            except Exception as e:
                logger.warning("Could not get row count for %s: %s", table_name, e)
                return None
        queries = {
            "postgresql": """
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = :table_name
            """,
            "mysql": """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = :table_name
            """,
        }
        query = queries.get(self._db_type, queries["postgresql"])
        try:
            with self._connector.session() as session:
                result = session.execute(text(query), {"table_name": table_name})
                row = result.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.warning("Could not get row count for %s: %s", table_name, e)
            return None
