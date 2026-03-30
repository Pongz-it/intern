"""Unstructured API fallback for document parsing."""

from typing import Any

from agent_rag.core.env_config import IngestionEnvConfig


def get_unstructured_api_key() -> str | None:
    """Return Unstructured API key if configured."""
    return IngestionEnvConfig().unstructured_api_key


def unstructured_to_text(content: bytes, filename: str) -> str:
    """
    Parse content via Unstructured API.

    Raises:
        RuntimeError if dependencies are missing or API key is not set.
    """
    api_key = get_unstructured_api_key()
    if not api_key:
        raise RuntimeError("Unstructured API key not configured.")

    try:
        from unstructured_client import UnstructuredClient  # type: ignore
        from unstructured_client.models import operations, shared  # type: ignore
        from unstructured.staging.base import dict_to_elements  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "unstructured-client not installed. Run: pip install unstructured-client unstructured"
        ) from e

    client = UnstructuredClient(api_key_auth=api_key)

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(content=content, file_name=filename),
            strategy="fast",
        )
    )

    response = client.general.partition(req)
    if response.status_code != 200:
        raise RuntimeError(
            f"Unstructured API error: status={response.status_code}"
        )

    elements = dict_to_elements(response.elements)
    return "\n\n".join(str(el) for el in elements)
