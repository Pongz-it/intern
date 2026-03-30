"""LiteLLM-based embedding provider."""

from typing import Any, Optional

from agent_rag.core.config import EmbeddingConfig
from agent_rag.core.exceptions import EmbeddingError
from agent_rag.embedding.interface import Embedder
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_litellm_embedding_model_name(
    model: str,
    provider: Optional[str] = None,
    api_base: Optional[str] = None,
) -> str:
    """Normalize embedding model naming for LiteLLM calls.

    This project commonly uses OpenAI-compatible gateways exposed as ``/v1``.
    LiteLLM expects those embedding models to be prefixed with ``openai/``
    even when the upstream model itself is hosted by another provider.
    """
    normalized_model = (model or "").strip()
    normalized_provider = (provider or "").strip().lower()
    normalized_api_base = (api_base or "").strip().lower().rstrip("/")

    if not normalized_model:
        return normalized_model

    model_suffix = normalized_model.split("/", 1)[-1]

    if normalized_api_base.endswith("/v1"):
        return f"openai/{model_suffix}"

    if "/" in normalized_model:
        return normalized_model

    if normalized_provider and normalized_provider not in {
        "litellm",
        "openai",
        "openai_compatible",
        "custom",
    }:
        return f"{normalized_provider}/{normalized_model}"

    return normalized_model


class LiteLLMEmbedder(Embedder):
    """Embedding provider using LiteLLM."""

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._initialized = False
        self._actual_dimension: Optional[int] = None
        self._litellm_model = normalize_litellm_embedding_model_name(
            model=self.config.model,
            provider=self.config.provider,
            api_base=self.config.api_base,
        )

    def _ensure_initialized(self) -> None:
        """Ensure LiteLLM is initialized."""
        if self._initialized:
            return

        try:
            import litellm
            self._initialized = True
        except ImportError:
            raise EmbeddingError(
                "LiteLLM is not installed. Install with: pip install litellm",
                model=self.config.model,
            )

    def _build_params(self, input_texts: list[str]) -> dict[str, Any]:
        """Build parameters for LiteLLM embedding call."""
        params: dict[str, Any] = {
            "model": self._litellm_model,
            "input": input_texts,
        }

        if self.config.api_key:
            params["api_key"] = self.config.api_key

        if self.config.api_base:
            params["api_base"] = self.config.api_base

        params.update(self.config.extra_options)

        return params

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        self._ensure_initialized()

        import litellm

        params = self._build_params([text])

        try:
            response = litellm.embedding(**params)
            embedding = response.data[0]["embedding"]
            if self._actual_dimension is None:
                self._actual_dimension = len(embedding)
            return embedding
        except Exception as e:
            raise EmbeddingError(
                f"Failed to embed text with model '{self._litellm_model}': {e}",
                model=self._litellm_model,
            )

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        if not texts:
            return []

        self._ensure_initialized()

        import litellm

        # Process in batches
        all_embeddings: list[list[float]] = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            params = self._build_params(batch)

            try:
                response = litellm.embedding(**params)
                # Sort by index to ensure correct order
                sorted_data = sorted(response.data, key=lambda x: x["index"])
                batch_embeddings = [item["embedding"] for item in sorted_data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to embed batch with model '{self._litellm_model}': {e}",
                    model=self._litellm_model,
                )

        return all_embeddings
