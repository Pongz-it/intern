"""OCR provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class OCRResult:
    """Structured OCR result."""
    text: str
    confidence: float | None
    provider: str
    metadata: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "provider": self.provider,
            "metadata": self.metadata,
            "error": self.error,
        }


class OCRProvider(ABC):
    """Base OCR provider interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name used for selection."""
        raise NotImplementedError

    @abstractmethod
    async def extract_text(self, image_content: bytes) -> OCRResult:
        """Extract text from image bytes."""
        raise NotImplementedError
