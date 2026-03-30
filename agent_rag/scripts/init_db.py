"""Database initialization script.

Creates database tables using Alembic migrations.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def check_database_connection() -> bool:
    """Check if database is accessible.

    Returns:
        True if connection successful
    """
    try:
        from agent_rag.core.database import get_db_manager

        db = get_db_manager()
        healthy = await db.health_check()
        if healthy:
            logger.info("Database connection successful")
        else:
            logger.error("Database health check failed")
        return healthy
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


async def create_tables_direct() -> bool:
    """Create tables directly without Alembic (for quick setup).

    Returns:
        True if successful
    """
    try:
        from agent_rag.core.database import get_db_manager
        from agent_rag.ingestion.models import Base

        db = get_db_manager()

        logger.info("Creating tables directly from SQLAlchemy models...")
        async with db.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False


def run_migrations(revision: str = "head") -> bool:
    """Run Alembic migrations.

    Args:
        revision: Target revision (default: head)

    Returns:
        True if successful
    """
    try:
        from alembic import command
        from alembic.config import Config

        # Find alembic.ini
        alembic_ini = Path(__file__).parent.parent.parent.parent / "alembic.ini"
        if not alembic_ini.exists():
            logger.error(f"alembic.ini not found at {alembic_ini}")
            return False

        alembic_cfg = Config(str(alembic_ini))

        logger.info(f"Running migrations to revision: {revision}")
        command.upgrade(alembic_cfg, revision)

        logger.info("Migrations completed successfully")
        return True
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def check_current_revision() -> str:
    """Get current database revision.

    Returns:
        Current revision or 'none'
    """
    try:
        from alembic import command
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        from alembic.runtime.environment import EnvironmentContext

        alembic_ini = Path(__file__).parent.parent.parent.parent / "alembic.ini"
        if not alembic_ini.exists():
            return "alembic.ini not found"

        alembic_cfg = Config(str(alembic_ini))

        # Import to trigger env.py
        import agent_rag.core.env_config

        # Get current revision
        from sqlalchemy import create_engine, text
        from agent_rag.core.env_config import database_config

        # Convert async URL to sync for this check
        sync_url = database_config.async_database_url.replace("+asyncpg", "")
        engine = create_engine(sync_url)

        with engine.connect() as conn:
            result = conn.execute(text("SELECT version_num FROM alembic_version"))
            row = result.fetchone()
            if row:
                return row[0]
            return "none"
    except Exception as e:
        return f"error: {e}"


def main():
    """CLI entry point for database initialization."""
    parser = argparse.ArgumentParser(
        description="Initialize Agent RAG database"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check database connection",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current migration status",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Create tables directly without Alembic (for quick dev setup)",
    )
    parser.add_argument(
        "--revision",
        default="head",
        help="Target migration revision (default: head)",
    )

    args = parser.parse_args()

    # Check connection
    if args.check:
        connected = asyncio.run(check_database_connection())
        sys.exit(0 if connected else 1)

    # Show status
    if args.status:
        revision = check_current_revision()
        print(f"Current revision: {revision}")
        sys.exit(0)

    # Check database connection first
    if not asyncio.run(check_database_connection()):
        logger.error("Cannot connect to database. Please check your configuration.")
        sys.exit(1)

    # Create tables
    if args.direct:
        success = asyncio.run(create_tables_direct())
    else:
        success = run_migrations(args.revision)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
