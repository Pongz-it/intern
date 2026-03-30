"""OCR provider registrations."""

from agent_rag.ingestion.ocr.registry import register_provider
from agent_rag.ingestion.ocr.providers.aws_textract_ocr import AWSTextractOCRProvider
from agent_rag.ingestion.ocr.providers.google_vision_ocr import GoogleVisionOCRProvider
from agent_rag.ingestion.ocr.providers.llm_ocr import LLMOCRProvider
from agent_rag.ingestion.ocr.providers.tesseract_ocr import TesseractOCRProvider


def register_default_ocr_providers() -> None:
    """Register built-in OCR providers."""
    register_provider(TesseractOCRProvider())
    register_provider(GoogleVisionOCRProvider())
    register_provider(AWSTextractOCRProvider())
    register_provider(LLMOCRProvider())
