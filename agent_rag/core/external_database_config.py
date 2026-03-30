"""External database configuration.

Supports multiple database types: PostgreSQL, MySQL, SQLite, Oracle, SQL Server, Snowflake, etc.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExternalDatabaseConfig:
    """Configuration for external database."""
    
    enabled: bool = False
    db_type: str = "postgresql"
    host: str = "localhost"
    port: int = 0  # 0 means use database-specific default port
    username: str = ""
    password: str = ""
    database: str = ""

    # Additional connection options
    ssl_mode: Optional[str] = None
    ssl_cert_path: Optional[str] = None
    odbc_driver: str = "ODBC Driver 17 for SQL Server"  # 可通过环境变量配置
    
    table_prefix: str = ""
    allowed_tables: list[str] = field(default_factory=list)
    blocked_tables: list[str] = field(default_factory=list)
    
    max_query_results: int = 1000
    default_limit: int = 100
    
    connection_timeout: int = 30
    pool_size: int = 5
    max_overflow: int = 10

    # Database type specific configurations
    oracle_service_name: Optional[str] = None
    snowflake_database: Optional[str] = None
    snowflake_warehouse: Optional[str] = None
    bigquery_project: Optional[str] = None

    @property
    def port_with_default(self) -> int:
        """Get port with database-specific default if not set."""
        if self.port > 0:
            return self.port
        defaults = {
            "postgresql": 5432,
            "mysql": 3306,
            "oracle": 1521,
            "mssql": 1433,
            "sqlserver": 1433,
            "snowflake": 443,
            "redshift": 5439,
        }
        return defaults.get(self.db_type.lower(), 5432)

    @property
    def connection_url(self) -> str:
        """Build database connection URL based on db_type."""
        import urllib.parse

        password_encoded = urllib.parse.quote(self.password, safe='')
        host = self.host
        port = self.port_with_default
        database = self.database
        username = self.username
        driver = urllib.parse.quote(self.odbc_driver, safe=' ')

        db_type = self.db_type.lower()

        if db_type == "sqlite":
            return f"sqlite:///{database}"

        elif db_type == "mysql":
            return f"mysql+pymysql://{username}:{password_encoded}@{host}:{port}/{database}"

        elif db_type == "oracle":
            url = f"oracle+oracledb://{username}:{password_encoded}@{host}:{port}"
            if self.oracle_service_name:
                url += f"/?service_name={self.oracle_service_name}"
            return url

        elif db_type == "mssql" or db_type == "sqlserver":
            return f"mssql+pyodbc://{username}:{password_encoded}@{host}:{port}/{database}?driver={driver}"

        elif db_type == "snowflake":
            account = host
            url = f"snowflake://{username}:{password_encoded}@{account}"
            if self.snowflake_database:
                url += f"/{self.snowflake_database}"
            if self.snowflake_warehouse:
                url += f"?warehouse={self.snowflake_warehouse}"
            return url

        elif db_type == "bigquery":
            if self.bigquery_project:
                return f"bigquery://{self.bigquery_project}/{database}"
            return f"bigquery://{database}"

        elif db_type == "redshift":
            return f"redshift+psycopg2://{username}:{password_encoded}@{host}:{port}/{database}"

        elif db_type == "duckdb":
            return f"duckdb:///{database}"

        else:  # postgresql default
            return f"postgresql+psycopg2://{username}:{password_encoded}@{host}:{port}/{database}"

    def get_table_list_query(self) -> str:
        """Get SQL query to list tables based on database type."""
        db_type = self.db_type.lower()
        
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
            "oracle": """
                SELECT table_name 
                FROM user_tables 
                ORDER BY table_name
            """,
            "mssql": """
                SELECT TABLE_NAME 
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA = 'dbo'
                ORDER BY TABLE_NAME
            """,
            "snowflake": f"""
                SELECT TABLE_NAME 
                FROM {self.database}.INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = 'PUBLIC'
                ORDER BY TABLE_NAME
            """,
            "bigquery": f"""
                SELECT table_name 
                FROM `{self.database}.INFORMATION_SCHEMA.TABLES`
            """,
            "duckdb": """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
                ORDER BY table_name
            """,
        }
        
        return queries.get(db_type, queries["postgresql"])

    def get_table_schema_query(self) -> str:
        """Get SQL query to get table schema based on database type."""
        db_type = self.db_type.lower()
        
        queries = {
            "postgresql": """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    ordinal_position
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position
            """,
            "mysql": f"""
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
            "oracle": """
                SELECT 
                    column_name,
                    data_type,
                    nullable,
                    data_default,
                    column_id as ordinal_position
                FROM user_tab_columns
                WHERE table_name = :table_name
                ORDER BY column_id
            """,
            "mssql": """
                SELECT 
                    COLUMN_NAME as column_name,
                    DATA_TYPE as data_type,
                    IS_NULLABLE as is_nullable,
                    COLUMN_DEFAULT as column_default,
                    ORDINAL_POSITION as ordinal_position
                FROM information_schema.COLUMNS
                WHERE TABLE_NAME = :table_name
                ORDER BY ORDINAL_POSITION
            """,
            "snowflake": f"""
                SELECT 
                    COLUMN_NAME as column_name,
                    DATA_TYPE as data_type,
                    IS_NULLABLE as is_nullable,
                    COLUMN_DEFAULT as column_default,
                    ORDINAL_POSITION as ordinal_position
                FROM {self.database}.INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = :table_name
                ORDER BY ORDINAL_POSITION
            """,
            "bigquery": f"""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    NULL as column_default,
                    ordinal_position
                FROM `{self.database}.INFORMATION_SCHEMA.COLUMNS`
                WHERE table_name = :table_name
                ORDER BY ordinal_position
            """,
            "duckdb": """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    ordinal_position
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position
            """,
        }
        
        return queries.get(db_type, queries["postgresql"])

    def get_sample_data_query(self, table_name: str, limit: int = 5) -> str:
        """Get SQL query to fetch sample data from a table."""
        db_type = self.db_type.lower()
        
        if db_type == "bigquery":
            return f"SELECT * FROM `{self.database}.{table_name}` LIMIT {limit}"
        
        return f"SELECT * FROM {table_name} LIMIT {limit}"

    @classmethod
    def from_env(cls, prefixes: list[str] = None) -> "ExternalDatabaseConfig":
        """Create config from environment variables.

        Args:
            prefixes: List of environment variable prefixes to try in order.
                      Default: ["AGENT_RAG_EXTERNAL_DB", "AGENT_RAG_API_EXTERNAL_DB"]
        """
        import os

        if prefixes is None:
            prefixes = ["AGENT_RAG_EXTERNAL_DB", "AGENT_RAG_API_EXTERNAL_DB"]

        def get_env(key: str, default: str = "") -> str:
            for prefix in prefixes:
                value = os.environ.get(f"{prefix}_{key}", None)
                if value is not None and value != "":
                    return value
            return os.environ.get(key, default)

        def get_env_int(key: str, default: int = 0) -> int:
            value = get_env(key, str(default))
            try:
                return int(value)
            except ValueError:
                return default

        def get_env_bool(key: str, default: bool = False) -> bool:
            value = get_env(key, str(default)).lower()
            return value in ("true", "1", "yes")

        def get_env_list(key: str) -> list[str]:
            value = get_env(key, "")
            if not value:
                return []
            return [v.strip() for v in value.split(",") if v.strip()]

        return cls(
            enabled=get_env_bool("ENABLED", False),
            db_type=get_env("TYPE", "postgresql"),
            host=get_env("HOST", "localhost"),
            port=get_env_int("PORT", 0),
            username=get_env("USERNAME", ""),
            password=get_env("PASSWORD", ""),
            database=get_env("DATABASE", ""),
            ssl_mode=get_env("SSL_MODE", None),
            ssl_cert_path=get_env("SSL_CERT_PATH", None),
            odbc_driver=get_env("ODBC_DRIVER", "ODBC Driver 17 for SQL Server"),
            table_prefix=get_env("TABLE_PREFIX", ""),
            allowed_tables=get_env_list("ALLOWED_TABLES"),
            blocked_tables=get_env_list("BLOCKED_TABLES"),
            max_query_results=get_env_int("MAX_RESULTS", 1000),
            default_limit=get_env_int("DEFAULT_LIMIT", 100),
            connection_timeout=get_env_int("TIMEOUT", 30),
            pool_size=get_env_int("POOL_SIZE", 5),
            max_overflow=get_env_int("MAX_OVERFLOW", 10),
            oracle_service_name=get_env("ORACLE_SERVICE_NAME", None),
            snowflake_database=get_env("SNOWFLAKE_DATABASE", None),
            snowflake_warehouse=get_env("SNOWFLAKE_WAREHOUSE", None),
            bigquery_project=get_env("BIGQUERY_PROJECT", None),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "db_type": self.db_type,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "database": self.database,
            "max_query_results": self.max_query_results,
            "default_limit": self.default_limit,
        }


# Backward compatibility
ExternalPostgresConfig = ExternalDatabaseConfig
