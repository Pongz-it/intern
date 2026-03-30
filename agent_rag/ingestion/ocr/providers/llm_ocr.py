"""LLM-based OCR provider using LiteLLM."""

import base64

from agent_rag.core.env_config import IngestionEnvConfig
from agent_rag.ingestion.ocr.base import OCRProvider, OCRResult


class LLMOCRProvider(OCRProvider):
    """OCR provider that uses an LLM with image input support."""

    @property
    def name(self) -> str:
        return "llm"

    async def extract_text(self, image_content: bytes) -> OCRResult:
        try:
            import litellm
        except ImportError as e:
            raise RuntimeError(
                "litellm not installed. Run: pip install litellm"
            ) from e

        config = IngestionEnvConfig()
        model = config.ocr_llm_model
        api_key = config.ocr_llm_api_key
        api_base = config.ocr_llm_api_base

        image_b64 = base64.b64encode(image_content).decode("utf-8")
        data_url = f"data:image/png;base64,{image_b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract all text from the image. Return plain text only.",
                    },
                    {"type": "input_image", "image_url": {"url": data_url}},
                ],
            }
        ]

        params = {"model": model, "messages": messages}
        if api_key:
            params["api_key"] = api_key
        if api_base:
            params["api_base"] = api_base

        response = litellm.completion(**params)
        content = response.choices[0].message.content or ""

        return OCRResult(
            text=content,
            confidence=None,
            provider=self.name,
            metadata={"model": model},
        )
