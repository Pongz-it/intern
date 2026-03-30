"""OCR module exports and provider registration."""

from agent_rag.ingestion.ocr.registry import get_ocr_provider, list_ocr_providers
from agent_rag.ingestion.ocr.providers import register_default_ocr_providers

register_default_ocr_providers()

__all__ = ["get_ocr_provider", "list_ocr_providers"]
