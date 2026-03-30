"""Vespa application configuration templates."""

from pathlib import Path

# Template directory path
TEMPLATES_DIR = Path(__file__).parent
SCHEMAS_DIR = TEMPLATES_DIR / "schemas"

__all__ = ["TEMPLATES_DIR", "SCHEMAS_DIR"]
