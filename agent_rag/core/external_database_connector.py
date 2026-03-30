"""External database connector.

Supports multiple database types: PostgreSQL, MySQL, SQLite, Oracle, SQL Server, Snowflake, etc.
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional
from sqlalchemy import text

from agent_rag.core.external_database_config import ExternalDatabaseConfig

logger = logging.getLogger(__name__)


class ExternalDatabaseConnector:
    """Connector for external database with multi-database support."""
    
    def __init__(self, config: Optional[ExternalDatabaseConfig] = None):
        """Initialize the database connector."""
        self.config = config or ExternalDatabaseConfig.from_env()
        self._engine = None
        self._session_factory = None
        self._db_type = self.config.db_type.lower()

    def _get_engine(self):
        """Get or create database engine."""
        if self._engine is not None:
            return self._engine

        db_type = self._db_type

        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            if db_type in ("postgresql", "mysql", "sqlite", "duckdb", "redshift"):
                self._engine = create_engine(
                    self.config.connection_url,
                    pool_size=self.config.pool_size,
                    max_overflow=self.config.max_overflow,
                    pool_timeout=self.config.connection_timeout,
                    echo=False,
                )

            elif db_type == "oracle":
                try:
                    self._engine = create_engine(
                        self.config.connection_url,
                        pool_size=self.config.pool_size,
                        max_overflow=self.config.max_overflow,
                        echo=False,
                    )
                except ImportError:
                    logger.warning("[ExternalDB] oracle+oracledb not installed, trying cx_Oracle")
                    from sqlalchemy import event
                    self._engine = create_engine(
                        f"oracle://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}",
                        pool_size=self.config.pool_size,
                        max_overflow=self.config.max_overflow,
                        echo=False,
                    )

            elif db_type in ("mssql", "sqlserver"):
                self._engine = create_engine(
                    self.config.connection_url,
                    pool_size=self.config.pool_size,
                    max_overflow=self.config.max_overflow,
                    pool_timeout=self.config.connection_timeout,
                    echo=False,
                )

            elif db_type == "snowflake":
                self._engine = create_engine(
                    self.config.connection_url,
                    pool_size=self.config.pool_size,
                    max_overflow=self.config.max_overflow,
                    echo=False,
                )

            elif db_type == "bigquery":
                from sqlalchemy_bigquery import BigQueryDialect
                self._engine = create_engine(
                    self.config.connection_url,
                    echo=False,
                )

            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            self._session_factory = sessionmaker(bind=self._engine)
            logger.info(f"[ExternalDB] Created {db_type} engine: {self.config.host}:{self.config.port}")
            return self._engine

        except ImportError as e:
            logger.error(f"[ExternalDB] Missing driver for {db_type}: {e}")
            raise ValueError(f"Database driver for {db_type} not installed. Please install the required package.")

    @contextmanager
    def session(self) -> Generator[Any, None, None]:
        """Get database session."""
        engine = self._get_engine()
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.session() as session:
                if self._db_type == "bigquery":
                    result = session.execute(text("SELECT 1"))
                else:
                    result = session.execute(text("SELECT 1"))
                result.fetchone()
                return True
        except Exception as e:
            logger.error(f"[ExternalDB] Connection test failed: {e}")
            return False

    def execute_query(
        self,
        sql: str,
        params: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> "ExternalDatabaseQueryResult":
        """Execute a SQL query and return results."""
        
        start_time = datetime.now()
        
        try:
            with self.session() as session:
                
                if limit:
                    if self._db_type == "bigquery":
                        sql = f"SELECT * FROM ({sql}) LIMIT {limit}"
                    else:
                        sql = f"SELECT * FROM ({sql}) AS subquery LIMIT {limit}"
                
                if params:
                    result = session.execute(text(sql), params)
                else:
                    result = session.execute(text(sql))
                
                rows = result.fetchall()
                
                columns = result.keys()
                data = [
                    dict(zip(columns, row))
                    for row in rows
                ]
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return ExternalDatabaseQueryResult(
                    sql=sql,
                    success=True,
                    data=data,
                    row_count=len(data),
                    execution_time_ms=execution_time,
                )
                
        except Exception as e:
            logger.error(f"[ExternalDB] Query failed: {e}")
            return ExternalDatabaseQueryResult(
                sql=sql,
                success=False,
                error_message=str(e),
                execution_time_ms=0,
            )

    def get_table_list(self) -> list[str]:
        """Get list of tables in the database."""
        try:
            query = self.config.get_table_list_query()
            with self.session() as session:
                result = session.execute(text(query))
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.error(f"[ExternalDB] Failed to get table list: {e}")
            return []

    def get_table_schema(self, table_name: str) -> dict:
        """Get schema information for a table."""
        try:
            query = self.config.get_table_schema_query()
            with self.session() as session:
                result = session.execute(text(query), {"table_name": table_name})
                
                if self._db_type == "sqlite":
                    columns = [
                        {
                            "name": row[1],  # name is at index 1
                            "type": row[2],  # type is at index 2
                            "nullable": not bool(row[3]),  # notnull is at index 3
                            "default": row[4],  # dflt_value is at index 4
                        }
                        for row in result.fetchall()
                    ]
                else:
                    columns = [
                        {
                            "name": row[0],
                            "type": row[1],
                            "nullable": row[2] == "YES",
                            "default": row[3],
                        }
                        for row in result.fetchall()
                    ]
                
                return {
                    "table": table_name,
                    "columns": columns,
                }
        except Exception as e:
            logger.error(f"[ExternalDB] Failed to get schema for {table_name}: {e}")
            return {"table": table_name, "columns": []}

    def get_sample_data(self, table_name: str, limit: int = 5) -> list[dict]:
        """Get sample data from a table."""
        try:
            query = self.config.get_sample_data_query(table_name, limit)
            with self.session() as session:
                result = session.execute(text(query))
                rows = result.fetchall()
                columns = result.keys()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"[ExternalDB] Failed to get sample data for {table_name}: {e}")
            return []

    def close(self):
        """Close the database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("[ExternalDB] Connection closed")

    @property
    def db_type(self) -> str:
        """Get database type."""
        return self._db_type


class ExternalDatabaseQueryResult:
    """Result from external database query."""
    
    sql: str
    success: bool
    data: list[dict]
    row_count: int
    execution_time_ms: float
    error_message: Optional[str] = None

    def __init__(
        self,
        sql: str,
        success: bool,
        data: list[dict] = None,
        row_count: int = 0,
        execution_time_ms: float = 0,
        error_message: str = None,
    ):
        self.sql = sql
        self.success = success
        self.data = data or []
        self.row_count = row_count
        self.execution_time_ms = execution_time_ms
        self.error_message = error_message

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "row_count": self.row_count,
            "execution_time_ms": self.execution_time_ms,
            "data": self.data[:50] if self.data else [],
            "error": self.error_message,
        }


# Backward compatibility
ExternalPostgresConnector = ExternalDatabaseConnector
ExternalPostgresQueryResult = ExternalDatabaseQueryResult
