"""OCR provider registry."""

from typing import Dict

from agent_rag.ingestion.ocr.base import OCRProvider


class OCRProviderRegistry:
    """Registry for OCR providers."""

    def __init__(self) -> None:
        self._providers: Dict[str, OCRProvider] = {}

    def register(self, provider: OCRProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> OCRProvider | None:
        return self._providers.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._providers.keys())


_registry = OCRProviderRegistry()


def register_provider(provider: OCRProvider) -> None:
    _registry.register(provider)


def get_ocr_provider(name: str) -> OCRProvider | None:
    return _registry.get(name)


def list_ocr_providers() -> list[str]:
    return _registry.list_names()
