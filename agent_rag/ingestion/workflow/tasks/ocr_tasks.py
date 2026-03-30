"""Hatchet tasks for OCR phase (image extraction and OCR processing)."""

import logging
from typing import Optional

from pydantic import BaseModel

from agent_rag.ingestion.ocr import get_ocr_provider
from agent_rag.ingestion.storage import get_minio_adapter

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Schemas for Task Inputs/Outputs
# ============================================================================


class ExtractImagesInput(BaseModel):
    """Input for extract_images_task."""

    item_id: str
    tenant_id: str
    parsed_ref: str  # MinIO path to parsed text
    image_count: int


class ExtractImagesOutput(BaseModel):
    """Output from extract_images_task."""

    item_id: str
    image_ids: list[str]
    image_count: int


class OCRImagesInput(BaseModel):
    """Input for ocr_images_task."""

    item_id: str
    tenant_id: str
    image_ids: list[str]
    ocr_provider: str = "tesseract"  # tesseract, google_vision, aws_textract, llm


class OCRImagesOutput(BaseModel):
    """Output from ocr_images_task."""

    item_id: str
    ocr_results: dict[str, dict]  # image_id -> {text, confidence, metadata}
    ocr_count: int


# ============================================================================
# Hatchet Task: extract_images_task
# ============================================================================


async def extract_images_task(input: ExtractImagesInput) -> ExtractImagesOutput:
    """
    Task 5: Extract images from parsed document.

    Images are already stored in MinIO by parse_document_task.
    This task verifies image storage and prepares for OCR.

    Returns:
        ExtractImagesOutput with image IDs
    """
    logger.info(
        f"Extracting images: item_id={input.item_id}, "
        f"expected_count={input.image_count}"
    )

    storage = get_minio_adapter()

    # List images for this item
    image_prefix = f"images/{input.tenant_id}/{input.item_id}/"

    try:
        image_objects = await storage.list_objects(prefix=image_prefix)

        # Extract image IDs from object names
        # Format: images/{tenant_id}/{item_id}/{image_id}.{ext}
        image_ids = []
        for obj in image_objects:
            # Extract image_id from object name
            parts = obj.object_name.split("/")
            if len(parts) == 4:
                filename = parts[3]  # {image_id}.{ext}
                image_id = filename.rsplit(".", 1)[0]
                image_ids.append(image_id)

        logger.info(
            f"Images extracted: {len(image_ids)} images found "
            f"(expected {input.image_count})"
        )

        return ExtractImagesOutput(
            item_id=input.item_id,
            image_ids=image_ids,
            image_count=len(image_ids),
        )

    except Exception as e:
        logger.error(f"Failed to extract images for {input.item_id}: {e}")
        # Return empty result on failure
        return ExtractImagesOutput(
            item_id=input.item_id,
            image_ids=[],
            image_count=0,
        )


# ============================================================================
# Hatchet Task: ocr_images_task
# ============================================================================


async def ocr_images_task(input: OCRImagesInput) -> OCRImagesOutput:
    """
    Task 6: Perform OCR on extracted images.

    Supports multiple OCR providers:
    - tesseract: Local OCR (free, offline)
    - google_vision: Google Cloud Vision API
    - aws_textract: AWS Textract

    Stores OCR results to MinIO.

    Returns:
        OCRImagesOutput with OCR text and metadata
    """
    logger.info(
        f"Performing OCR: item_id={input.item_id}, "
        f"image_count={len(input.image_ids)}, provider={input.ocr_provider}"
    )

    storage = get_minio_adapter()
    ocr_results = {}

    for image_id in input.image_ids:
        try:
            # Retrieve image from MinIO
            image_content = await storage.retrieve_image(
                tenant_id=input.tenant_id,
                item_id=input.item_id,
                image_id=image_id,
            )

            # Perform OCR based on provider
            provider = get_ocr_provider(input.ocr_provider)
            if provider is None:
                logger.warning(
                    f"Unknown OCR provider: {input.ocr_provider}, using tesseract"
                )
                provider = get_ocr_provider("tesseract")
            if provider is None:
                raise RuntimeError("No OCR providers registered")

            ocr_result = (await provider.extract_text(image_content)).to_dict()

            # Store OCR result to MinIO
            await storage.store_ocr_result(
                tenant_id=input.tenant_id,
                item_id=input.item_id,
                image_id=image_id,
                ocr_data=ocr_result,
            )

            ocr_results[image_id] = ocr_result

            logger.debug(
                f"OCR completed for image {image_id}: "
                f"{len(ocr_result.get('text', ''))} chars, "
                f"confidence={ocr_result.get('confidence', 0):.2f}"
            )

        except Exception as e:
            logger.error(f"OCR failed for image {image_id}: {e}")
            # Store error result
            ocr_results[image_id] = {
                "text": "",
                "confidence": 0.0,
                "error": str(e),
            }

    logger.info(
        f"OCR batch complete: {len(ocr_results)}/{len(input.image_ids)} images processed"
    )

    return OCRImagesOutput(
        item_id=input.item_id,
        ocr_results=ocr_results,
        ocr_count=len(ocr_results),
    )

