"""Database session management with async SQLAlchemy.

Provides:
- Async engine and session factory
- Context managers for session handling
- Dependency injection for FastAPI
- Health check utilities
"""

import contextlib
import logging
from typing import AsyncGenerator, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from agent_rag.core.env_config import database_config

logger = logging.getLogger(__name__)

# SQLAlchemy base for all models
Base = declarative_base()


class DatabaseManager:
    """
    Manages database connections with async SQLAlchemy.

    Features:
    - Connection pooling with configurable settings
    - Async context managers for session handling
    - Health check support
    - Graceful shutdown
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: Optional[int] = None,
        max_overflow: Optional[int] = None,
        pool_timeout: Optional[int] = None,
        pool_recycle: Optional[int] = None,
        echo: Optional[bool] = None,
    ):
        """
        Initialize database manager.

        Args:
            database_url: PostgreSQL async URL. If None, uses env config.
            pool_size: Connection pool size. If None, uses env config.
            max_overflow: Max overflow connections. If None, uses env config.
            pool_timeout: Pool checkout timeout. If None, uses env config.
            pool_recycle: Pool recycle time. If None, uses env config.
            echo: Whether to log SQL statements. If None, uses env config.
        """
        self._database_url = database_url or database_config.async_database_url
        self._pool_size = pool_size or database_config.pool_size
        self._max_overflow = max_overflow or database_config.max_overflow
        self._pool_timeout = pool_timeout or database_config.pool_timeout
        self._pool_recycle = pool_recycle or database_config.pool_recycle
        self._echo = echo if echo is not None else database_config.echo

        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    @property
    def engine(self) -> AsyncEngine:
        """Get or create the async engine."""
        if self._engine is None:
            self._engine = create_async_engine(
                self._database_url,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=self._pool_timeout,
                pool_recycle=self._pool_recycle,
                echo=self._echo,
                pool_pre_ping=True,  # Verify connections before use
            )
            logger.info(
                f"Created async engine (pool_size={self._pool_size}, "
                f"max_overflow={self._max_overflow})"
            )
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
        return self._session_factory

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Async context manager for database sessions.

        Usage:
            async with db_manager.session() as session:
                result = await session.execute(query)

        Yields:
            AsyncSession with automatic commit/rollback handling
        """
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    @contextlib.asynccontextmanager
    async def readonly_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Async context manager for read-only database sessions.

        No commit is performed; use for queries only.

        Yields:
            AsyncSession for read-only operations
        """
        session = self.session_factory()
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Dependency injection for FastAPI.

        Usage:
            @router.get("/items")
            async def get_items(session: AsyncSession = Depends(db_manager.get_session)):
                ...

        Yields:
            AsyncSession for request handling
        """
        async with self.session() as session:
            yield session

    async def health_check(self) -> bool:
        """
        Check database connectivity.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close the engine and all connections."""
        if self._engine is not None:
            await self._engine.dispose()
            logger.info("Closed database engine")
            self._engine = None
            self._session_factory = None

    async def create_tables(self) -> None:
        """
        Create all tables defined in Base.metadata.

        Note: In production, prefer using Alembic migrations.
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Created all database tables")

    async def drop_tables(self) -> None:
        """
        Drop all tables defined in Base.metadata.

        Warning: This is destructive and should only be used in tests.
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Dropped all database tables")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get the global database manager instance.

    Creates one if it doesn't exist.

    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Convenience function for FastAPI dependency injection.

    Usage:
        @router.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            ...
    """
    async with get_db_manager().session() as session:
        yield session


async def init_database() -> None:
    """
    Initialize the database connection.

    Call this at application startup.
    """
    db = get_db_manager()
    if await db.health_check():
        logger.info("Database connection established")
    else:
        raise RuntimeError("Failed to connect to database")


async def close_database() -> None:
    """
    Close the database connection.

    Call this at application shutdown.
    """
    global _db_manager
    if _db_manager is not None:
        await _db_manager.close()
        _db_manager = None


# Compatibility alias for existing code that expects AsyncSessionLocal
class AsyncSessionLocal:
    """
    Compatibility wrapper for existing code that uses AsyncSessionLocal as a context manager.

    Usage:
        async with AsyncSessionLocal() as session:
            result = await session.execute(query)
    """

    def __init__(self):
        self._db_manager = get_db_manager()

    async def __aenter__(self) -> AsyncSession:
        self._session = self._db_manager.session_factory()
        return self._session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self._session.rollback()
        else:
            await self._session.commit()
        await self._session.close()
