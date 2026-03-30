"""Unified initialization script for Agent RAG.

Initializes all required infrastructure:
- PostgreSQL: Database tables via Alembic migrations
- Vespa: Schema deployment via application package
- MinIO: Bucket creation (optional)
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AgentRAGInitializer:
    """Unified initializer for Agent RAG infrastructure."""

    def __init__(self, skip_vespa: bool = False, skip_db: bool = False, skip_minio: bool = True):
        """Initialize the initializer.

        Args:
            skip_vespa: Skip Vespa initialization
            skip_db: Skip database initialization
            skip_minio: Skip MinIO initialization (default True as optional)
        """
        self.skip_vespa = skip_vespa
        self.skip_db = skip_db
        self.skip_minio = skip_minio
        self.results = {}

    async def init_database(self, direct: bool = False) -> bool:
        """Initialize PostgreSQL database.

        Args:
            direct: Use direct table creation instead of migrations

        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("Initializing PostgreSQL Database")
        logger.info("=" * 60)

        try:
            from agent_rag.core.database import get_db_manager

            db = get_db_manager()

            # Check connection
            logger.info("Checking database connection...")
            if not await db.health_check():
                logger.error("Database connection failed")
                return False
            logger.info("Database connection successful")

            if direct:
                # Direct table creation
                from agent_rag.ingestion.models import Base

                logger.info("Creating tables directly from SQLAlchemy models...")
                async with db.engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                logger.info("Tables created successfully")
            else:
                # Use Alembic migrations
                from agent_rag.scripts.init_db import run_migrations

                logger.info("Running Alembic migrations...")
                if not run_migrations("head"):
                    logger.error("Migrations failed")
                    return False
                logger.info("Migrations completed")

            return True

        except ImportError as e:
            logger.error(f"Missing dependencies. Install with: pip install agent-rag[database]")
            logger.error(f"Import error: {e}")
            return False
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False

    def init_vespa(self, render_only: bool = False) -> bool:
        """Initialize Vespa schema.

        Args:
            render_only: Only render templates, don't deploy

        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("Initializing Vespa Schema")
        logger.info("=" * 60)

        try:
            from agent_rag.scripts.deploy_vespa import VespaDeployer
            import tempfile

            deployer = VespaDeployer()

            # Check health first
            if not render_only:
                logger.info("Checking Vespa health...")
                if not deployer.check_health():
                    logger.warning("Vespa is not accessible. Please start Vespa first.")
                    logger.warning("You can render templates with --render-vespa-only and deploy later.")
                    return False
                logger.info("Vespa is healthy")

            # Render templates
            from pathlib import Path
            import tempfile

            with tempfile.TemporaryDirectory(prefix="vespa_app_") as tmpdir:
                output_dir = Path(tmpdir)
                logger.info("Rendering Vespa schema templates...")
                app_dir = deployer.render_templates(output_dir)
                logger.info(f"Templates rendered to {app_dir}")

                if render_only:
                    # Copy to permanent location
                    dest = Path(__file__).parent.parent / "document_index" / "vespa" / "rendered_app"
                    import shutil
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(app_dir, dest)
                    logger.info(f"Rendered application saved to {dest}")
                    return True

                # Deploy
                logger.info("Deploying Vespa application...")
                if not deployer.deploy(app_dir, wait_for_convergence=True):
                    logger.error("Vespa deployment failed")
                    return False

                # Validate
                logger.info("Validating deployment...")
                if not deployer.validate_deployment():
                    logger.warning("Deployment validation failed, but may still work")

            logger.info("Vespa initialization completed")
            return True

        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"Vespa initialization failed: {e}")
            return False

    async def init_minio(self) -> bool:
        """Initialize MinIO buckets.

        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("Initializing MinIO Buckets")
        logger.info("=" * 60)

        try:
            from minio import Minio
            from agent_rag.core.env_config import minio_config

            client = Minio(
                f"{minio_config.host}:{minio_config.port}",
                access_key=minio_config.access_key,
                secret_key=minio_config.secret_key,
                secure=minio_config.secure,
            )

            # Create buckets
            buckets = [
                minio_config.raw_bucket,
                minio_config.parsed_bucket,
                minio_config.images_bucket,
            ]

            for bucket in buckets:
                if not client.bucket_exists(bucket):
                    client.make_bucket(bucket)
                    logger.info(f"Created bucket: {bucket}")
                else:
                    logger.info(f"Bucket exists: {bucket}")

            logger.info("MinIO initialization completed")
            return True

        except ImportError:
            logger.warning("MinIO client not installed. Install with: pip install minio")
            return False
        except Exception as e:
            logger.error(f"MinIO initialization failed: {e}")
            return False

    async def run(
        self,
        direct_db: bool = False,
        render_vespa_only: bool = False,
    ) -> bool:
        """Run all initializations.

        Args:
            direct_db: Use direct table creation instead of migrations
            render_vespa_only: Only render Vespa templates

        Returns:
            True if all succeeded
        """
        logger.info("=" * 60)
        logger.info("Agent RAG Infrastructure Initialization")
        logger.info("=" * 60)

        all_success = True

        # Initialize PostgreSQL
        if not self.skip_db:
            success = await self.init_database(direct=direct_db)
            self.results["postgresql"] = success
            all_success = all_success and success
        else:
            logger.info("Skipping PostgreSQL initialization")

        # Initialize Vespa
        if not self.skip_vespa:
            success = self.init_vespa(render_only=render_vespa_only)
            self.results["vespa"] = success
            all_success = all_success and success
        else:
            logger.info("Skipping Vespa initialization")

        # Initialize MinIO
        if not self.skip_minio:
            success = await self.init_minio()
            self.results["minio"] = success
            all_success = all_success and success
        else:
            logger.info("Skipping MinIO initialization")

        # Summary
        logger.info("=" * 60)
        logger.info("Initialization Summary")
        logger.info("=" * 60)
        for component, success in self.results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"  {component}: {status}")

        return all_success


def main():
    """CLI entry point for unified initialization."""
    parser = argparse.ArgumentParser(
        description="Initialize Agent RAG infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize everything (PostgreSQL + Vespa)
  python -m agent_rag.scripts.init_all

  # Initialize only PostgreSQL with direct table creation
  python -m agent_rag.scripts.init_all --skip-vespa --direct

  # Initialize only Vespa
  python -m agent_rag.scripts.init_all --skip-db

  # Render Vespa templates without deploying
  python -m agent_rag.scripts.init_all --skip-db --render-vespa-only

  # Initialize with MinIO
  python -m agent_rag.scripts.init_all --with-minio
        """,
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip PostgreSQL database initialization",
    )
    parser.add_argument(
        "--skip-vespa",
        action="store_true",
        help="Skip Vespa schema initialization",
    )
    parser.add_argument(
        "--with-minio",
        action="store_true",
        help="Include MinIO bucket initialization",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use direct table creation instead of Alembic migrations",
    )
    parser.add_argument(
        "--render-vespa-only",
        action="store_true",
        help="Only render Vespa templates, don't deploy",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check infrastructure status",
    )

    args = parser.parse_args()

    if args.check:
        # Status check mode
        print("\n=== Agent RAG Infrastructure Status ===\n")

        # Check PostgreSQL
        print("PostgreSQL:")
        try:
            from agent_rag.core.database import get_db_manager

            async def check_db():
                db = get_db_manager()
                return await db.health_check()

            if asyncio.run(check_db()):
                print("  ✓ Connected")
                from agent_rag.scripts.init_db import check_current_revision
                rev = check_current_revision()
                print(f"  Migration revision: {rev}")
            else:
                print("  ✗ Connection failed")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        # Check Vespa
        print("\nVespa:")
        try:
            from agent_rag.scripts.deploy_vespa import VespaDeployer
            deployer = VespaDeployer()
            if deployer.check_health():
                print("  ✓ Healthy")
            else:
                print("  ✗ Not accessible")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        # Check MinIO
        print("\nMinIO:")
        try:
            from minio import Minio
            from agent_rag.core.env_config import minio_config
            client = Minio(
                f"{minio_config.host}:{minio_config.port}",
                access_key=minio_config.access_key,
                secret_key=minio_config.secret_key,
                secure=minio_config.secure,
            )
            buckets = list(client.list_buckets())
            print(f"  ✓ Connected ({len(buckets)} buckets)")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()
        sys.exit(0)

    # Run initialization
    initializer = AgentRAGInitializer(
        skip_vespa=args.skip_vespa,
        skip_db=args.skip_db,
        skip_minio=not args.with_minio,
    )

    success = asyncio.run(initializer.run(
        direct_db=args.direct,
        render_vespa_only=args.render_vespa_only,
    ))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
