"""Database migration management script.

Wraps Alembic commands for convenient migration management.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_alembic_config():
    """Get Alembic configuration.

    Returns:
        Alembic Config object
    """
    from alembic.config import Config

    alembic_ini = Path(__file__).parent.parent.parent.parent / "alembic.ini"
    if not alembic_ini.exists():
        logger.error(f"alembic.ini not found at {alembic_ini}")
        sys.exit(1)

    return Config(str(alembic_ini))


def cmd_upgrade(revision: str = "head"):
    """Upgrade database to revision.

    Args:
        revision: Target revision
    """
    from alembic import command

    config = get_alembic_config()
    logger.info(f"Upgrading to revision: {revision}")
    command.upgrade(config, revision)
    logger.info("Upgrade completed")


def cmd_downgrade(revision: str):
    """Downgrade database to revision.

    Args:
        revision: Target revision (use -1 for previous)
    """
    from alembic import command

    config = get_alembic_config()
    logger.info(f"Downgrading to revision: {revision}")
    command.downgrade(config, revision)
    logger.info("Downgrade completed")


def cmd_revision(message: str, autogenerate: bool = False):
    """Create new migration revision.

    Args:
        message: Revision message
        autogenerate: Auto-detect changes from models
    """
    from alembic import command

    config = get_alembic_config()
    logger.info(f"Creating new revision: {message}")
    command.revision(config, message=message, autogenerate=autogenerate)
    logger.info("Revision created")


def cmd_current():
    """Show current revision."""
    from alembic import command

    config = get_alembic_config()
    command.current(config)


def cmd_history():
    """Show revision history."""
    from alembic import command

    config = get_alembic_config()
    command.history(config)


def cmd_heads():
    """Show available heads."""
    from alembic import command

    config = get_alembic_config()
    command.heads(config)


def cmd_check():
    """Check if there are pending migrations."""
    from alembic import command
    from alembic.script import ScriptDirectory

    config = get_alembic_config()
    script = ScriptDirectory.from_config(config)

    # Get current head
    head = script.get_current_head()

    # Get database current revision
    from agent_rag.scripts.init_db import check_current_revision
    current = check_current_revision()

    if current == head:
        print("Database is up to date")
        return True
    else:
        print(f"Database at revision {current}, head is {head}")
        print("Run 'agent-rag-migrate upgrade head' to apply pending migrations")
        return False


def main():
    """CLI entry point for migration management."""
    parser = argparse.ArgumentParser(
        description="Manage Agent RAG database migrations"
    )
    subparsers = parser.add_subparsers(dest="command", help="Migration commands")

    # Upgrade command
    upgrade_parser = subparsers.add_parser("upgrade", help="Upgrade to revision")
    upgrade_parser.add_argument(
        "revision",
        nargs="?",
        default="head",
        help="Target revision (default: head)",
    )

    # Downgrade command
    downgrade_parser = subparsers.add_parser("downgrade", help="Downgrade to revision")
    downgrade_parser.add_argument(
        "revision",
        help="Target revision (use -1 for previous)",
    )

    # Revision command
    revision_parser = subparsers.add_parser("revision", help="Create new revision")
    revision_parser.add_argument(
        "-m", "--message",
        required=True,
        help="Revision message",
    )
    revision_parser.add_argument(
        "--autogenerate",
        action="store_true",
        help="Auto-detect changes from models",
    )

    # Info commands
    subparsers.add_parser("current", help="Show current revision")
    subparsers.add_parser("history", help="Show revision history")
    subparsers.add_parser("heads", help="Show available heads")
    subparsers.add_parser("check", help="Check for pending migrations")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "upgrade":
            cmd_upgrade(args.revision)
        elif args.command == "downgrade":
            cmd_downgrade(args.revision)
        elif args.command == "revision":
            cmd_revision(args.message, args.autogenerate)
        elif args.command == "current":
            cmd_current()
        elif args.command == "history":
            cmd_history()
        elif args.command == "heads":
            cmd_heads()
        elif args.command == "check":
            success = cmd_check()
            sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
