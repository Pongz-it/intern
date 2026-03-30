"""Google Vision OCR provider."""

from agent_rag.ingestion.ocr.base import OCRProvider, OCRResult


class GoogleVisionOCRProvider(OCRProvider):
    """OCR provider using Google Vision API."""

    @property
    def name(self) -> str:
        return "google_vision"

    async def extract_text(self, image_content: bytes) -> OCRResult:
        try:
            from google.cloud import vision

            client = vision.ImageAnnotatorClient()
            image = vision.Image(content=image_content)

            response = client.text_detection(image=image)
            texts = response.text_annotations

            if not texts:
                return OCRResult(
                    text="",
                    confidence=0.0,
                    provider=self.name,
                    metadata={},
                )

            full_text = texts[0].description
            confidences = [
                text.confidence for text in texts[1:] if hasattr(text, "confidence")
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=full_text.strip(),
                confidence=avg_confidence,
                provider=self.name,
                metadata={
                    "word_count": len(full_text.split()),
                    "char_count": len(full_text),
                    "element_count": len(texts),
                },
            )
        except Exception as e:
            return OCRResult(
                text="",
                confidence=0.0,
                provider=self.name,
                metadata={},
                error=str(e),
            )
