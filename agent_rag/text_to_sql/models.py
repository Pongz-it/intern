"""Text-to-SQL data models for schema storage."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ColumnType(Enum):
    """SQL column types."""

    INTEGER = "INTEGER"
    BIGINT = "BIGINT"
    SMALLINT = "SMALLINT"
    DECIMAL = "DECIMAL"
    NUMERIC = "NUMERIC"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    REAL = "REAL"
    VARCHAR = "VARCHAR"
    CHAR = "CHAR"
    TEXT = "TEXT"
    BOOLEAN = "BOOLEAN"
    DATE = "DATE"
    TIME = "TIME"
    DATETIME = "DATETIME"
    TIMESTAMP = "TIMESTAMP"
    JSON = "JSON"
    JSONB = "JSONB"
    ARRAY = "ARRAY"
    UUID = "UUID"
    BYTEA = "BYTEA"
    UNKNOWN = "UNKNOWN"


class QueryIntent(Enum):
    """User query intent types for data queries."""

    AGGREGATION = "aggregation"
    FILTER = "filter"
    JOIN = "join"
    SORT = "sort"
    CALCULATION = "calculation"
    COMPARISON = "comparison"
    TIME_SERIES = "time_series"
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    DISTINCT = "distinct"
    GROUP_BY = "group_by"
    HAVING = "having"
    SUBQUERY = "subquery"
    UNION = "union"


class KeywordCategory(Enum):
    """Keyword categories for intent analysis."""

    QUANTITY = "quantity"
    AMOUNT = "amount"
    TIME = "time"
    STATISTICS = "statistics"
    SORTING = "sorting"


KEYWORD_CATEGORY_MAP = {
    "quantity": ["数量词"],
    "amount": ["金额词"],
    "time": ["时间词"],
    "statistics": ["统计词"],
    "sorting": ["排序词"],
}


@dataclass
class TableColumn:
    """Represents a database table column."""

    name: str
    column_type: ColumnType
    description: str = ""
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_table: Optional[str] = None
    foreign_column: Optional[str] = None
    is_nullable: bool = True
    default_value: Optional[str] = None
    sample_values: list[Any] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)


@dataclass
class TableRelationship:
    """Represents a foreign key relationship between tables."""

    from_table: str
    from_column: str
    to_table: str
    to_column: str
    description: str = ""


@dataclass
class TableIndex:
    """Represents a database table index."""

    name: str
    columns: list[str]
    is_unique: bool = False
    index_type: str = "btree"
    description: str = ""


@dataclass
class DatabaseTable:
    """Represents a database table schema."""

    name: str
    description: str = ""
    columns: list[TableColumn] = field(default_factory=list)
    relationships: list[TableRelationship] = field(default_factory=list)
    indexes: list[TableIndex] = field(default_factory=list)
    primary_key: Optional[str] = None
    sample_queries: list[str] = field(default_factory=list)
    row_count_estimate: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def get_column(self, name: str) -> Optional[TableColumn]:
        """Get a column by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def get_foreign_key_columns(self) -> list[TableColumn]:
        """Get all foreign key columns."""
        return [col for col in self.columns if col.is_foreign_key]

    def get_numeric_columns(self) -> list[TableColumn]:
        """Get all numeric columns."""
        numeric_types = {
            ColumnType.INTEGER,
            ColumnType.BIGINT,
            ColumnType.SMALLINT,
            ColumnType.DECIMAL,
            ColumnType.NUMERIC,
            ColumnType.FLOAT,
            ColumnType.DOUBLE,
            ColumnType.REAL,
        }
        return [col for col in self.columns if col.column_type in numeric_types]

    def get_datetime_columns(self) -> list[TableColumn]:
        """Get all datetime columns."""
        datetime_types = {
            ColumnType.DATE,
            ColumnType.TIME,
            ColumnType.DATETIME,
            ColumnType.TIMESTAMP,
        }
        return [col for col in self.columns if col.column_type in datetime_types]


@dataclass
class DatabaseSchema:
    """Represents a complete database schema."""

    tables: list[DatabaseTable] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    last_updated: datetime = field(default_factory=datetime.now)

    def get_table(self, name: str) -> Optional[DatabaseTable]:
        """Get a table by name."""
        for table in self.tables:
            if table.name == name:
                return table
        return None

    def get_related_tables(self, table_name: str) -> list[DatabaseTable]:
        """Get tables related to the given table through foreign keys."""
        related = []
        table = self.get_table(table_name)
        if not table:
            return related

        for col in table.columns:
            if col.is_foreign_key and col.foreign_table:
                related_table = self.get_table(col.foreign_table)
                if related_table and related_table not in related:
                    related.append(related_table)

        for other_table in self.tables:
            if other_table.name == table_name:
                continue
            for col in other_table.columns:
                if col.is_foreign_key and col.foreign_table == table_name:
                    if other_table not in related:
                        related.append(other_table)

        return related


@dataclass
class SchemaEmbedding:
    """Stores embedding vector for schema element."""

    element_type: str
    element_name: str
    description: str
    embedding: list[float]
    table_name: Optional[str] = None
    column_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SQLQuery:
    """Represents a generated SQL query."""

    original_query: str
    sql: str
    tables_used: list[str]
    columns_used: list[str]
    intent: QueryIntent
    confidence: float
    parameters: dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SQLQueryResult:
    """Represents the result of SQL query execution."""

    sql: str
    success: bool
    row_count: int = 0
    data: list[dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    executed_at: datetime = field(default_factory=datetime.now)


@dataclass
class QueryAuditLog:
    """Audit log entry for SQL queries."""

    id: str
    user_query: str
    generated_sql: str
    tables_accessed: list[str]
    intent: QueryIntent
    success: bool
    row_count: int
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
