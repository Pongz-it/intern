"""AWS Textract OCR provider."""

from agent_rag.ingestion.ocr.base import OCRProvider, OCRResult


class AWSTextractOCRProvider(OCRProvider):
    """OCR provider using AWS Textract."""

    @property
    def name(self) -> str:
        return "aws_textract"

    async def extract_text(self, image_content: bytes) -> OCRResult:
        try:
            import boto3

            client = boto3.client("textract")
            response = client.detect_document_text(Document={"Bytes": image_content})

            text_blocks = []
            confidences = []

            for block in response["Blocks"]:
                if block["BlockType"] == "LINE":
                    text_blocks.append(block["Text"])
                    confidences.append(block["Confidence"])

            full_text = "\n".join(text_blocks)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=full_text.strip(),
                confidence=avg_confidence / 100.0,
                provider=self.name,
                metadata={
                    "word_count": len(full_text.split()),
                    "char_count": len(full_text),
                    "block_count": len(text_blocks),
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
