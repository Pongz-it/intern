"""MinIO storage adapter for ingestion content and artifacts."""

import io
import json
import logging
from pathlib import Path
from typing import Any, BinaryIO, Optional
from uuid import UUID

from minio import Minio
from minio.error import S3Error

from agent_rag.core.env_config import ingestion_config

logger = logging.getLogger(__name__)


class MinIOAdapter:
    """
    MinIO storage adapter for ingestion content.

    Handles storage of:
    - Raw content files
    - Parsed text (markdown)
    - Extracted images
    - OCR results
    - Derived artifacts (doc summaries, chunk contexts)
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        secure: Optional[bool] = None,
    ):
        """
        Initialize MinIO client.

        Args:
            endpoint: MinIO server endpoint (host:port)
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket_name: Bucket name for ingestion storage
            secure: Use HTTPS (True) or HTTP (False)
        """
        self.endpoint = endpoint or ingestion_config.minio_endpoint
        self.access_key = access_key or ingestion_config.minio_access_key
        self.secret_key = secret_key or ingestion_config.minio_secret_key
        self.bucket_name = bucket_name or ingestion_config.minio_ingestion_bucket
        self.secure = secure if secure is not None else ingestion_config.minio_secure

        logger.info(
            f"[MinIO INIT] Initializing MinIOAdapter: endpoint={self.endpoint}, "
            f"bucket={self.bucket_name}, secure={self.secure}"
        )

        # Initialize MinIO client
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

        # Ensure bucket exists
        self._ensure_bucket()
        logger.info(f"[MinIO INIT] Bucket '{self.bucket_name}' ready")

    def _ensure_bucket(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as e:
            raise RuntimeError(f"Failed to create MinIO bucket: {e}")

    def _build_key(
        self,
        category: str,
        tenant_id: str,
        item_id: UUID,
        filename: str,
    ) -> str:
        """
        Build MinIO object key.

        Format: {category}/{tenant_id}/{item_id}/{filename}

        Args:
            category: Storage category (raw, parsed, images, ocr, derived)
            tenant_id: Tenant ID
            item_id: Ingestion item UUID
            filename: File name

        Returns:
            MinIO object key
        """
        return f"{category}/{tenant_id}/{item_id}/{filename}"

    # ========================================================================
    # Raw Content Storage
    # ========================================================================

    def store_raw_content(
        self,
        tenant_id: str,
        item_id: UUID,
        filename: str,
        content: bytes | BinaryIO,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Store raw content file.

        Args:
            tenant_id: Tenant ID
            item_id: Ingestion item UUID
            filename: Original filename
            content: File content (bytes or file-like object)
            content_type: MIME type

        Returns:
            MinIO key to stored content
        """
        key = self._build_key("raw", tenant_id, item_id, filename)

        logger.info(
            f"[MinIO STORE] endpoint={self.endpoint}, bucket={self.bucket_name}, "
            f"key={key}, tenant={tenant_id}, item_id={item_id}"
        )

        # Convert bytes to BytesIO if needed
        if isinstance(content, bytes):
            data = io.BytesIO(content)
            length = len(content)
        else:
            # For file-like objects, get size
            data = content
            data.seek(0, 2)  # Seek to end
            length = data.tell()
            data.seek(0)  # Reset to start

        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=key,
                data=data,
                length=length,
                content_type=content_type or "application/octet-stream",
            )
            # Verify the object was stored
            exists = self.object_exists(key)
            logger.info(f"[MinIO STORE] Completed: key={key}, size={length}, verified_exists={exists}")
            return key
        except S3Error as e:
            logger.error(f"[MinIO STORE] Failed: key={key}, error={e}")
            raise RuntimeError(f"Failed to store raw content: {e}")

    def get_raw_content(self, key: str) -> bytes:
        """
        Retrieve raw content.

        Args:
            key: MinIO key to content

        Returns:
            File content as bytes
        """
        response = None
        try:
            response = self.client.get_object(self.bucket_name, key)
            return response.read()
        except S3Error as e:
            raise RuntimeError(f"Failed to retrieve raw content: {e}")
        finally:
            if response:
                response.close()
                response.release_conn()

    async def retrieve_raw_content(
        self,
        tenant_id: str,
        item_id: str,
        filename: str,
    ) -> bytes:
        """
        Retrieve raw content by tenant/item/filename.

        Convenience async wrapper that builds the key and retrieves content.

        Args:
            tenant_id: Tenant ID
            item_id: Ingestion item ID (string or UUID)
            filename: Original filename

        Returns:
            Raw content bytes
        """
        key = self._build_key("raw", tenant_id, item_id, filename)
        logger.info(
            f"[MinIO RETRIEVE] endpoint={self.endpoint}, bucket={self.bucket_name}, "
            f"key={key}, tenant={tenant_id}, item_id={item_id}"
        )
        # Check if object exists before retrieving
        exists = self.object_exists(key)
        logger.info(f"[MinIO RETRIEVE] Object exists check: {exists}")
        if not exists:
            # List objects with similar prefix to debug
            prefix = f"raw/{tenant_id}/"
            objects = list(self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True))
            logger.warning(
                f"[MinIO RETRIEVE] Object not found! Available objects with prefix '{prefix}': "
                f"{[obj.object_name for obj in objects[:10]]}"
            )
        return self.get_raw_content(key)

    # ========================================================================
    # Parsed Text Storage
    # ========================================================================

    def store_parsed_text(
        self,
        tenant_id: str,
        item_id: UUID,
        text: str,
    ) -> str:
        """
        Store parsed text as markdown.

        Args:
            tenant_id: Tenant ID
            item_id: Ingestion item UUID
            text: Parsed text content

        Returns:
            MinIO key to parsed text
        """
        key = self._build_key("parsed", tenant_id, item_id, "text.md")
        content = text.encode("utf-8")

        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=key,
                data=io.BytesIO(content),
                length=len(content),
                content_type="text/markdown; charset=utf-8",
            )
            return key
        except S3Error as e:
            raise RuntimeError(f"Failed to store parsed text: {e}")

    def get_parsed_text(self, key: str) -> str:
        """
        Retrieve parsed text.

        Args:
            key: MinIO key to parsed text

        Returns:
            Parsed text content
        """
        try:
            response = self.client.get_object(self.bucket_name, key)
            return response.read().decode("utf-8")
        except S3Error as e:
            raise RuntimeError(f"Failed to retrieve parsed text: {e}")
        finally:
            response.close()
            response.release_conn()

    async def retrieve_parsed_text(
        self,
        tenant_id: str,
        item_id: str,
    ) -> str:
        """
        Retrieve parsed text by tenant/item.

        Convenience async wrapper that builds the key and retrieves parsed text.

        Args:
            tenant_id: Tenant ID
            item_id: Ingestion item ID (string or UUID)

        Returns:
            Parsed text content
        """
        key = self._build_key("parsed", tenant_id, item_id, "text.md")
        return self.get_parsed_text(key)

    # ========================================================================
    # Image Storage
    # ========================================================================

    def store_image(
        self,
        tenant_id: str,
        item_id: UUID,
        image_id: str,
        image_content: bytes,
        extension: str,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Store extracted image.

        Args:
            tenant_id: Tenant ID
            item_id: Ingestion item UUID
            image_id: Unique image identifier
            image_content: Image bytes
            extension: File extension (png, jpg, etc.)
            content_type: MIME type

        Returns:
            MinIO key to stored image
        """
        filename = f"{image_id}.{extension}"
        key = self._build_key("images", tenant_id, item_id, filename)

        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=key,
                data=io.BytesIO(image_content),
                length=len(image_content),
                content_type=content_type or f"image/{extension}",
            )
            return key
        except S3Error as e:
            raise RuntimeError(f"Failed to store image: {e}")

    def get_image(self, key: str) -> bytes:
        """
        Retrieve image content.

        Args:
            key: MinIO key to image

        Returns:
            Image bytes
        """
        try:
            response = self.client.get_object(self.bucket_name, key)
            return response.read()
        except S3Error as e:
            raise RuntimeError(f"Failed to retrieve image: {e}")
        finally:
            response.close()
            response.release_conn()

    async def retrieve_image(
        self,
        tenant_id: str,
        item_id: str,
        image_id: str,
    ) -> bytes:
        """
        Retrieve image by tenant/item/image_id.

        Finds the image file by listing objects with the image_id prefix.

        Args:
            tenant_id: Tenant ID
            item_id: Ingestion item ID
            image_id: Image identifier

        Returns:
            Image bytes
        """
        # List objects to find the image (extension may vary)
        prefix = f"images/{tenant_id}/{item_id}/{image_id}."
        objects = list(self.client.list_objects(self.bucket_name, prefix=prefix))

        if not objects:
            raise RuntimeError(f"Image not found: {prefix}*")

        # Get the first matching object
        key = objects[0].object_name
        return self.get_image(key)

    async def list_objects(self, prefix: str) -> list:
        """
        List objects with given prefix.

        Args:
            prefix: Object key prefix

        Returns:
            List of object info
        """
        try:
            return list(self.client.list_objects(self.bucket_name, prefix=prefix))
        except S3Error as e:
            raise RuntimeError(f"Failed to list objects: {e}")

    # ========================================================================
    # OCR Results Storage
    # ========================================================================

    def store_ocr_result(
        self,
        tenant_id: str,
        item_id: UUID,
        image_id: str,
        ocr_data: dict[str, Any],
    ) -> str:
        """
        Store OCR result as JSON.

        Args:
            tenant_id: Tenant ID
            item_id: Ingestion item UUID
            image_id: Image identifier
            ocr_data: OCR result dictionary

        Returns:
            MinIO key to OCR result
        """
        filename = f"{image_id}.json"
        key = self._build_key("ocr", tenant_id, item_id, filename)
        content = json.dumps(ocr_data, ensure_ascii=False, indent=2).encode("utf-8")

        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=key,
                data=io.BytesIO(content),
                length=len(content),
                content_type="application/json; charset=utf-8",
            )
            return key
        except S3Error as e:
            raise RuntimeError(f"Failed to store OCR result: {e}")

    def get_ocr_result(self, key: str) -> dict[str, Any]:
        """
        Retrieve OCR result.

        Args:
            key: MinIO key to OCR result

        Returns:
            OCR result dictionary
        """
        try:
            response = self.client.get_object(self.bucket_name, key)
            data = response.read().decode("utf-8")
            return json.loads(data)
        except S3Error as e:
            raise RuntimeError(f"Failed to retrieve OCR result: {e}")
        finally:
            response.close()
            response.release_conn()

    # ========================================================================
    # Derived Artifacts Storage
    # ========================================================================

    def store_doc_summary(
        self,
        tenant_id: str,
        item_id: UUID,
        summary: str,
    ) -> str:
        """
        Store document summary.

        Args:
            tenant_id: Tenant ID
            item_id: Ingestion item UUID
            summary: Document summary text

        Returns:
            MinIO key to doc summary
        """
        key = self._build_key("derived", tenant_id, item_id, "doc_summary.txt")
        content = summary.encode("utf-8")

        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=key,
                data=io.BytesIO(content),
                length=len(content),
                content_type="text/plain; charset=utf-8",
            )
            return key
        except S3Error as e:
            raise RuntimeError(f"Failed to store doc summary: {e}")

    def get_doc_summary(self, key: str) -> str:
        """
        Retrieve document summary.

        Args:
            key: MinIO key to doc summary

        Returns:
            Document summary text
        """
        try:
            response = self.client.get_object(self.bucket_name, key)
            return response.read().decode("utf-8")
        except S3Error as e:
            raise RuntimeError(f"Failed to retrieve doc summary: {e}")
        finally:
            response.close()
            response.release_conn()

    def store_chunk_contexts(
        self,
        tenant_id: str,
        item_id: UUID,
        contexts: dict[int, str],
    ) -> str:
        """
        Store chunk contexts as JSON.

        Args:
            tenant_id: Tenant ID
            item_id: Ingestion item UUID
            contexts: Mapping of chunk_id to context text

        Returns:
            MinIO key to chunk contexts
        """
        key = self._build_key("derived", tenant_id, item_id, "chunk_context.json")
        content = json.dumps(contexts, ensure_ascii=False, indent=2).encode("utf-8")

        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=key,
                data=io.BytesIO(content),
                length=len(content),
                content_type="application/json; charset=utf-8",
            )
            return key
        except S3Error as e:
            raise RuntimeError(f"Failed to store chunk contexts: {e}")

    def get_chunk_contexts(self, key: str) -> dict[int, str]:
        """
        Retrieve chunk contexts.

        Args:
            key: MinIO key to chunk contexts

        Returns:
            Mapping of chunk_id to context text
        """
        try:
            response = self.client.get_object(self.bucket_name, key)
            data = response.read().decode("utf-8")
            # Convert string keys back to int
            raw_dict = json.loads(data)
            return {int(k): v for k, v in raw_dict.items()}
        except S3Error as e:
            raise RuntimeError(f"Failed to retrieve chunk contexts: {e}")
        finally:
            response.close()
            response.release_conn()

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def delete_object(self, key: str) -> None:
        """
        Delete object from MinIO.

        Args:
            key: MinIO key to delete
        """
        try:
            self.client.remove_object(self.bucket_name, key)
        except S3Error as e:
            raise RuntimeError(f"Failed to delete object: {e}")

    def delete_item_artifacts(self, tenant_id: str, item_id: UUID) -> None:
        """
        Delete all artifacts for an ingestion item.

        Args:
            tenant_id: Tenant ID
            item_id: Ingestion item UUID
        """
        # List all objects under this item's prefix
        prefixes = [
            f"raw/{tenant_id}/{item_id}/",
            f"parsed/{tenant_id}/{item_id}/",
            f"images/{tenant_id}/{item_id}/",
            f"ocr/{tenant_id}/{item_id}/",
            f"derived/{tenant_id}/{item_id}/",
        ]

        for prefix in prefixes:
            try:
                objects = self.client.list_objects(
                    self.bucket_name,
                    prefix=prefix,
                    recursive=True,
                )
                for obj in objects:
                    self.client.remove_object(self.bucket_name, obj.object_name)
            except S3Error as e:
                # Log error but continue cleanup
                print(f"Warning: Failed to delete objects with prefix {prefix}: {e}")

    def object_exists(self, key: str) -> bool:
        """
        Check if object exists in MinIO.

        Args:
            key: MinIO key to check

        Returns:
            True if object exists, False otherwise
        """
        try:
            self.client.stat_object(self.bucket_name, key)
            return True
        except S3Error:
            return False

    def get_object_info(self, key: str) -> dict[str, Any]:
        """
        Get object metadata.

        Args:
            key: MinIO key

        Returns:
            Object metadata dict
        """
        try:
            stat = self.client.stat_object(self.bucket_name, key)
            return {
                "key": key,
                "size": stat.size,
                "etag": stat.etag,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified,
                "metadata": stat.metadata,
            }
        except S3Error as e:
            raise RuntimeError(f"Failed to get object info: {e}")


# Singleton instance
_minio_adapter: Optional[MinIOAdapter] = None


def get_minio_adapter() -> MinIOAdapter:
    """
    Get singleton MinIO adapter instance.

    Returns:
        MinIO adapter instance
    """
    global _minio_adapter
    if _minio_adapter is None:
        _minio_adapter = MinIOAdapter()
    return _minio_adapter
