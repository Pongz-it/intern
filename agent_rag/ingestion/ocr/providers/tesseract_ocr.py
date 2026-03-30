"""Tesseract OCR provider."""

from agent_rag.ingestion.ocr.base import OCRProvider, OCRResult


class TesseractOCRProvider(OCRProvider):
    """OCR provider using Tesseract."""

    @property
    def name(self) -> str:
        return "tesseract"

    async def extract_text(self, image_content: bytes) -> OCRResult:
        try:
            import pytesseract
            from PIL import Image
            from io import BytesIO

            image = Image.open(BytesIO(image_content))
            text = pytesseract.image_to_string(image)

            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence / 100.0,
                provider=self.name,
                metadata={
                    "word_count": len(text.split()),
                    "char_count": len(text),
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
