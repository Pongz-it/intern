"""Memory vector storage using Qdrant."""

import json
import uuid
import http.client
from datetime import datetime
from typing import Any, Optional

from agent_rag.core.session_memory_models import MemoryType, UserMemory
from agent_rag.embedding.interface import Embedder
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)

MEMORY_COLLECTION = "user_memories"


class QdrantRESTClient:
    """Simple REST client for Qdrant.

    We intentionally use one short-lived HTTP connection per request here.
    The previous pooled ``HTTPConnection`` implementation could surface
    ``CannotSendRequest: Request-sent`` when the same client instance was hit
    by overlapping requests during session + memory loading.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 6333, max_pool_size: int = 10):
        self.host = host
        self.port = port
        self.max_pool_size = max_pool_size

    def _request(self, method: str, path: str, body: Optional[Any] = None) -> dict:
        """Make a REST request to Qdrant."""
        conn = http.client.HTTPConnection(self.host, self.port, timeout=30)
        headers = {"Content-Type": "application/json"}
        try:
            if body is not None:
                body = json.dumps(body)
            conn.request(method, path, body, headers)
            response = conn.getresponse()
            if response.status >= 400:
                error_msg = response.read().decode()
                raise Exception(f"Qdrant error {response.status}: {error_msg}")
            data = response.read().decode()
            if data:
                return json.loads(data)
            return {}
        finally:
            conn.close()

    def get_collection(self, collection_name: str) -> dict:
        """Get collection info."""
        return self._request("GET", f"/collections/{collection_name}")

    def create_collection(self, collection_name: str, vectors_config: dict) -> dict:
        """Create collection."""
        payload = {"vectors": vectors_config}
        return self._request("PUT", f"/collections/{collection_name}", payload)

    def delete_collection(self, collection_name: str) -> dict:
        """Delete collection."""
        return self._request("DELETE", f"/collections/{collection_name}")

    def upsert(self, collection_name: str, points: list) -> dict:
        """Upsert points."""
        payload = {"points": points}
        return self._request("PUT", f"/collections/{collection_name}/points", payload)

    def query_points(
        self,
        collection_name: str,
        query: list,
        query_filter: Optional[dict] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> dict:
        """Query points."""
        payload = {
            "vector": query,
            "limit": limit,
            "with_payload": True,
        }
        if query_filter:
            payload["filter"] = query_filter
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        return self._request("POST", f"/collections/{collection_name}/points/search", payload)


class MemoryStore:
    """Memory storage using Qdrant vector database."""

    def __init__(
        self,
        embedder: Embedder,
        qdrant_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        collection_name: str = MEMORY_COLLECTION,
        recreate_collection: bool = False,
    ):
        """Initialize memory store.

        Args:
            embedder: Embedder for generating vectors
            qdrant_path: Optional path to local Qdrant storage
            qdrant_url: Optional Qdrant server URL
            collection_name: Collection name
            recreate_collection: Whether to recreate collection on init
        """
        self.embedder = embedder
        self.collection_name = collection_name

        if qdrant_url:
            host = qdrant_url.replace("http://", "").replace("https://", "").split(":")[0]
            port = int(qdrant_url.replace("http://", "").replace("https://", "").split(":")[1]) if ":" in qdrant_url.replace("http://", "").replace("https://", "") else 6333
            self._client = QdrantRESTClient(host=host, port=port)
        elif qdrant_path:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(path=qdrant_path, check_compatibility=False)
        else:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(":memory:", check_compatibility=False)

        if recreate_collection:
            try:
                if hasattr(self._client, 'delete_collection'):
                    self._client.delete_collection(collection_name)
            except Exception:
                pass

        self._discover_and_ensure_collection()
        logger.info(f"[MemoryStore] Initialized with Qdrant at {qdrant_path or qdrant_url or 'memory'}")

    def _discover_and_ensure_collection(self) -> None:
        """Discover actual dimension and ensure collection exists."""
        try:
            dummy_embedding = self.embedder.embed("dimension check")
            self._dimension = len(dummy_embedding)
            logger.info(f"[MemoryStore] Discovered actual embedding dimension: {self._dimension}")
        except Exception as e:
            logger.warning(f"[MemoryStore] Failed to discover dimension, using configured value: {e}")
            self._dimension = self.embedder.dimension

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure collection exists with retry logic."""
        import time

        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                self._client.get_collection(self.collection_name)
                logger.info(f"[MemoryStore] Collection {self.collection_name} exists")
                return
            except Exception as e:
                error_str = str(e)
                is_502 = "502" in error_str or "Bad Gateway" in error_str
                is_not_found = "NotFound" in error_str or "doesn't exist" in error_str or "Not found" in error_str

                if is_502 and attempt < max_retries - 1:
                    logger.warning(f"[MemoryStore] get_collection failed (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue

                if not is_502 and not is_not_found:
                    raise

                try:
                    self._client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config={
                            "size": self._dimension,
                            "distance": "Cosine",
                        },
                    )
                    logger.info(f"[MemoryStore] Created collection: {self.collection_name}")
                    return
                except Exception as create_error:
                    if "AlreadyExists" in str(create_error):
                        logger.info(f"[MemoryStore] Collection {self.collection_name} already exists")
                        return

                    if attempt < max_retries - 1:
                        logger.warning(f"[MemoryStore] create_collection failed (attempt {attempt + 1}/{max_retries}): {create_error}, retrying...")
                        time.sleep(retry_delay * (attempt + 1))
                        continue

                    raise create_error

        raise Exception(f"[MemoryStore] Failed to ensure collection {self.collection_name} after {max_retries} retries")

    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        session_id: str,
        user_id: Optional[str] = None,
        importance: float = 0.5,
        source_message_id: Optional[str] = None,
        source_conversation: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> UserMemory:
        """Add a memory.

        Args:
            content: Memory content
            memory_type: Type of memory
            session_id: Session ID
            user_id: Optional user ID
            importance: Importance score (0-1)
            source_message_id: Source message ID
            source_conversation: Source conversation context
            metadata: Optional metadata

        Returns:
            Created memory
        """
        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty")

        content = content.strip()

        try:
            embedding = self.embedder.embed(content)
        except Exception as embed_err:
            logger.error(f"[MemoryStore] Embedding failed for content: '{content[:50]}...': {embed_err}")
            raise ValueError(f"Embedding generation failed: {embed_err}")

        if embedding is None or len(embedding) == 0:
            logger.error(f"[MemoryStore] Empty embedding returned for content: '{content[:50]}...'")
            raise ValueError(f"Generated embedding is empty for content: {content[:50]}...")

        if len(embedding) != self._dimension:
            logger.warning(f"[MemoryStore] Embedding dimension mismatch: got {len(embedding)}, expected {self._dimension}")

        memory_id = str(uuid.uuid4())
        now = datetime.utcnow()

        point = {
            "id": memory_id,
            "vector": embedding,
            "payload": {
                "id": memory_id,
                "content": content,
                "memory_type": memory_type.value,
                "session_id": session_id,
                "user_id": user_id,
                "importance": importance,
                "source_message_id": source_message_id,
                "source_conversation": source_conversation,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "metadata": metadata or {},
            },
        }

        self._client.upsert(collection_name=self.collection_name, points=[point])
        logger.info(f"[MemoryStore] Added memory: {memory_id}")

        return UserMemory(
            id=memory_id,
            session_id=session_id,
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            importance=importance,
            source_message_id=source_message_id,
            source_conversation=source_conversation,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

    async def search_memories(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_types: Optional[list[MemoryType]] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> list[UserMemory]:
        """Search memories by query.

        Args:
            query: Search query
            session_id: Optional session ID filter (for session-specific memories)
            user_id: Optional user ID filter
            memory_types: Optional memory type filters
            limit: Maximum results
            score_threshold: Minimum score threshold

        Returns:
            List of matching memories
        """
        query_embedding = self.embedder.embed(query)

        search_filter = None
        conditions = []

        if session_id:
            conditions.append({"key": "session_id", "match": {"value": session_id}})
        elif user_id:
            conditions.append({"key": "user_id", "match": {"value": user_id}})
        if memory_types:
            type_values = [mt.value for mt in memory_types]
            if len(type_values) == 1:
                conditions.append({"key": "memory_type", "match": {"value": type_values[0]}})
            else:
                conditions.append({"key": "memory_type", "match": {"any": type_values}})

        if conditions:
            search_filter = {"must": conditions}

        results = self._client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=search_filter,
            limit=limit,
            score_threshold=score_threshold,
        )

        memories = []
        points = results if isinstance(results, list) else results.get("result", results.get("points", []))
        for r in points:
            payload = r.get("payload", {})
            memories.append(
                UserMemory(
                    id=payload.get("id"),
                    session_id=payload.get("session_id", ""),
                    user_id=payload.get("user_id"),
                    memory_type=MemoryType(payload.get("memory_type", "fact")),
                    content=payload.get("content", ""),
                    embedding=query_embedding,
                    importance=payload.get("importance", 0.5),
                    source_message_id=payload.get("source_message_id"),
                    source_conversation=payload.get("source_conversation"),
                    created_at=datetime.fromisoformat(payload.get("created_at", datetime.utcnow().isoformat())),
                    updated_at=datetime.fromisoformat(payload.get("updated_at", datetime.utcnow().isoformat())),
                    metadata=payload.get("metadata", {}),
                )
            )

        return memories

    async def get_memories_by_session(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[UserMemory]:
        """Get memories by session ID.

        Args:
            session_id: Session ID to filter by
            limit: Maximum results

        Returns:
            List of memories for the session
        """
        return await self.get_all_memories(session_id=session_id, limit=limit)

    async def get_all_memories(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[UserMemory]:
        """Get all memories.

        Args:
            session_id: Optional session ID filter
            user_id: Optional user ID filter
            limit: Maximum results

        Returns:
            List of memories
        """
        search_filter = None
        conditions = []

        if session_id:
            conditions.append({"key": "session_id", "match": {"value": session_id}})
        elif user_id:
            conditions.append({"key": "user_id", "match": {"value": user_id}})

        if conditions:
            search_filter = {"must": conditions}

        results = self._client.query_points(
            collection_name=self.collection_name,
            query=[0.0] * self._dimension,
            query_filter=search_filter,
            limit=limit,
        )

        memories = []
        points = results if isinstance(results, list) else results.get("result", results.get("points", []))
        for r in points:
            payload = r.get("payload", {})
            memories.append(
                UserMemory(
                    id=payload.get("id"),
                    session_id=payload.get("session_id", ""),
                    user_id=payload.get("user_id"),
                    memory_type=MemoryType(payload.get("memory_type", "fact")),
                    content=payload.get("content", ""),
                    embedding=[0.0] * self._dimension,
                    importance=payload.get("importance", 0.5),
                    source_message_id=payload.get("source_message_id"),
                    source_conversation=payload.get("source_conversation"),
                    created_at=datetime.fromisoformat(payload.get("created_at", datetime.utcnow().isoformat())),
                    updated_at=datetime.fromisoformat(payload.get("updated_at", datetime.utcnow().isoformat())),
                    metadata=payload.get("metadata", {}),
                )
            )

        return memories
