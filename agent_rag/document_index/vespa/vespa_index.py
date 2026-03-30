"""Vespa-based document index implementation."""

import re
from typing import Any, Optional

import httpx

from agent_rag.core.config import DocumentIndexConfig
from agent_rag.core.exceptions import DocumentIndexError
from agent_rag.core.models import Chunk, SearchFilters
from agent_rag.document_index.interface import ChunkRequest, DocumentIndex
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class _VespaBM25Stats:
    """Placeholder BM25 stats for Vespa (Vespa handles BM25 internally)."""
    def __init__(self) -> None:
        self.doc_freqs = {}


class VespaIndex(DocumentIndex):
    """Vespa-based document index for production use."""

    def __init__(
        self,
        config: Optional[DocumentIndexConfig] = None,
        host: str = "localhost",
        port: int = 8080,
        app_name: str = "agent_rag",
        timeout: int = 30,
        schema_name: str = "agent_rag_chunk",
        title_content_ratio: float = 0.2,
        decay_factor: float = 0.5,
    ) -> None:
        if config:
            self.host = config.vespa_host
            self.port = config.vespa_port
            self.app_name = config.vespa_app_name
            self.timeout = config.vespa_timeout
            self.schema_name = config.vespa_schema_name
            self.title_content_ratio = config.vespa_title_content_ratio
            self.decay_factor = config.vespa_decay_factor
        else:
            self.host = host
            self.port = port
            self.app_name = app_name
            self.timeout = timeout
            self.schema_name = schema_name
            self.title_content_ratio = title_content_ratio
            self.decay_factor = decay_factor

        self.base_url = f"http://{self.host}:{self.port}"
        self._client: Optional[httpx.Client] = None

        self.chunks: dict[str, Chunk] = {}
        self.documents: dict[str, list[str]] = {}
        self.bm25 = _VespaBM25Stats()

    def load_existing(self) -> int:
        """Load existing documents from Vespa into memory index.

        This method queries Vespa for all existing documents and populates
        the local chunks and documents dictionaries. This is needed because
        VespaIndex maintains an in-memory cache of document metadata for
        the 'knowledge base is empty' check.

        Returns:
            Number of documents loaded.
        """
        logger.info(f"[VespaIndex] Loading existing documents from Vespa at {self.base_url}")

        try:
            import httpx
            load_client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=httpx.HTTPTransport(local_address="0.0.0.0"),
                follow_redirects=True,
            )

            params = {
                "yql": "select * from agent_rag_chunk where true",
                "hits": 400,
            }

            response = load_client.get("/search/", params=params)
            
            if not response.is_success:
                logger.error(f"[VespaIndex] Failed to load documents: HTTP {response.status_code}")
                load_client.close()
                return 0
            
            data = response.json()

            load_client.close()

            root = data.get("root", {})
            children = root.get("children", [])
            total_count = root.get("fields", {}).get("totalCount", 0)

            loaded_count = 0
            for child in children:
                fields = child.get("fields", {})

                doc_id = fields.get("document_id")
                chunk_id = fields.get("chunk_id", 0)

                if not doc_id:
                    continue

                unique_id = f"{doc_id}_{chunk_id}"

                try:
                    chunk = Chunk(
                        document_id=doc_id,
                        chunk_id=chunk_id,
                        content=fields.get("content", ""),
                        title=fields.get("title"),
                        source_type=fields.get("source_type"),
                        link=fields.get("source_links"),
                        metadata={},
                    )

                    self.chunks[unique_id] = chunk

                    if doc_id not in self.documents:
                        self.documents[doc_id] = []
                    if unique_id not in self.documents[doc_id]:
                        self.documents[doc_id].append(unique_id)

                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"[VespaIndex] Failed to parse chunk {unique_id}: {e}")
                    continue

            logger.info(f"[VespaIndex] Loaded {loaded_count} chunks from {total_count} documents in Vespa")
            return loaded_count

        except httpx.HTTPError as e:
            logger.error(f"[VespaIndex] Failed to load existing documents: {e}")
            return 0

    @property
    def client(self) -> httpx.Client:
        """Get HTTP client with trust_env=False to avoid proxy issues."""
        if self._client is None:
            import httpx
            transport = httpx.HTTPTransport(local_address="0.0.0.0")
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=transport,
                follow_redirects=False,
            )
        return self._client

    def _is_chinese_query(self, query: str) -> bool:
        """Check if query contains significant Chinese characters."""
        if not query:
            return False
        chinese_chars = sum(1 for char in query if (
            0x4E00 <= ord(char) <= 0x9FFF or
            0x3400 <= ord(char) <= 0x4DBF or
            0x3000 <= ord(char) <= 0x303F
        ))
        return chinese_chars / len(query) > 0.3

    def _build_yql_query(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[list[float]] = None,
        filters: Optional[SearchFilters] = None,
        hybrid_alpha: float = 0.5,
        num_results: int = 10,
        ranking_profile: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build Vespa YQL query."""
        conditions = []

        if filters:
            custom_filters = filters.custom_filters or {}
            document_sets = (
                filters.document_sets
                if filters.document_sets is not None
                else custom_filters.get("document_sets")
            )
            user_folder = (
                filters.user_folder
                if filters.user_folder is not None
                else custom_filters.get("user_folder")
            )
            user_project = (
                filters.user_project
                if filters.user_project is not None
                else custom_filters.get("user_project")
            )

            if filters.source_types:
                source_filter = " OR ".join(
                    f'source_type contains "{st}"' for st in filters.source_types
                )
                conditions.append(f"({source_filter})")

            if filters.document_ids:
                doc_filter = " OR ".join(
                    f'document_id contains "{did}"' for did in filters.document_ids
                )
                conditions.append(f"({doc_filter})")

            if isinstance(document_sets, list) and document_sets:
                doc_set_filter = " OR ".join(
                    f'document_sets contains "{doc_set}"' for doc_set in document_sets
                )
                conditions.append(f"({doc_set_filter})")

            if user_folder is not None:
                try:
                    folder_id = int(user_folder)
                    conditions.append(f"(user_folder = {folder_id})")
                except (TypeError, ValueError):
                    pass

            if isinstance(user_project, list) and user_project:
                project_filter = " OR ".join(
                    f'user_project contains "{project_id}"' for project_id in user_project
                )
                conditions.append(f"({project_filter})")

        where_clause = " AND ".join(conditions) if conditions else "true"

        target_hits = max(10 * num_results, 1000)

        is_chinese = self._is_chinese_query(query) if query else False

        if query_embedding:
            if is_chinese:
                yql = (
                    f"select * from {self.schema_name} where {where_clause} and "
                    f"(({{targetHits: {target_hits}}}nearestNeighbor(embeddings, query_embedding)) "
                    f'or ({{defaultIndex: "content"}}userInput(@query)) '
                    f'or ({{defaultIndex: "title"}}userInput(@query)))'
                )
            else:
                yql = (
                    f"select * from {self.schema_name} where {where_clause} and "
                    f"(({{targetHits: {target_hits}}}nearestNeighbor(embeddings, query_embedding)) "
                    f"or ({{targetHits: {target_hits}}}nearestNeighbor(title_embedding, query_embedding)) "
                    f'or ({{grammar: "weakAnd"}}userInput(@query)) '
                    f'or ({{defaultIndex: \"content_summary\"}}userInput(@query)))'
                )
        else:
            yql = f"select * from {self.schema_name} where {where_clause}"
            if query:
                if is_chinese:
                    yql += ' and ({defaultIndex: "content"}userInput(@query))'
                else:
                    yql += ' and ({grammar: "weakAnd"}userInput(@query))'

        if not ranking_profile:
            if query_embedding:
                dim = len(query_embedding)
                ranking_profile = f"hybrid_search_semantic_base_{dim}"
            else:
                ranking_profile = "bm25_only"

        body: dict[str, Any] = {
            "yql": yql,
            "hits": num_results,
            "ranking.profile": ranking_profile,
            "timeout": self.timeout,
        }

        if query_embedding:
            body["input.query(query_embedding)"] = query_embedding
            body["input.query(alpha)"] = hybrid_alpha
            body["input.query(title_content_ratio)"] = self.title_content_ratio
            body["input.query(decay_factor)"] = self.decay_factor

        if query:
            body["query"] = query

        return body

    def _parse_vespa_hit(self, hit: dict[str, Any]) -> Chunk:
        """Parse Vespa hit to Chunk."""
        fields = hit.get("fields", {})

        return Chunk(
            document_id=fields.get("document_id", ""),
            chunk_id=int(fields.get("chunk_id", 0)),
            content=fields.get("content", ""),
            embedding=fields.get("embedding"),
            title=fields.get("title"),
            semantic_identifier=fields.get("semantic_identifier"),
            source_type=fields.get("source_type"),
            link=fields.get("link"),
            metadata=fields.get("metadata", {}),
            score=hit.get("relevance", 0.0),
        )

    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        filters: Optional[SearchFilters] = None,
        hybrid_alpha: float = 0.5,
        num_results: int = 10,
    ) -> list[Chunk]:
        """Perform hybrid search."""
        body = self._build_yql_query(
            query=query,
            query_embedding=query_embedding,
            filters=filters,
            hybrid_alpha=hybrid_alpha,
            num_results=num_results,
        )

        try:
            response = self.client.post("/search/", json=body)
            response.raise_for_status()
            data = response.json()

            hits = data.get("root", {}).get("children", [])
            return [self._parse_vespa_hit(hit) for hit in hits]
        except httpx.HTTPError as e:
            raise DocumentIndexError(
                f"Vespa search failed: {e}",
                index_type="vespa",
            )

    def semantic_search(
        self,
        query_embedding: list[float],
        filters: Optional[SearchFilters] = None,
        num_results: int = 10,
    ) -> list[Chunk]:
        """Perform semantic search."""
        body = self._build_yql_query(
            query_embedding=query_embedding,
            filters=filters,
            hybrid_alpha=1.0,  # Pure semantic
            num_results=num_results,
            ranking_profile=f"hybrid_search_semantic_base_{len(query_embedding)}",
        )

        try:
            response = self.client.post("/search/", json=body)
            response.raise_for_status()
            data = response.json()

            hits = data.get("root", {}).get("children", [])
            return [self._parse_vespa_hit(hit) for hit in hits]
        except httpx.HTTPError as e:
            raise DocumentIndexError(
                f"Vespa semantic search failed: {e}",
                index_type="vespa",
            )

    def keyword_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        num_results: int = 10,
    ) -> list[Chunk]:
        """Perform keyword search."""
        body = self._build_yql_query(
            query=query,
            filters=filters,
            hybrid_alpha=0.0,  # Pure keyword
            num_results=num_results,
            ranking_profile="bm25_only",
        )

        try:
            response = self.client.post("/search/", json=body)
            response.raise_for_status()
            data = response.json()

            hits = data.get("root", {}).get("children", [])
            return [self._parse_vespa_hit(hit) for hit in hits]
        except httpx.HTTPError as e:
            raise DocumentIndexError(
                f"Vespa keyword search failed: {e}",
                index_type="vespa",
            )

    def get_chunks_by_document(
        self,
        document_id: str,
        chunk_range: Optional[tuple[int, int]] = None,
    ) -> list[Chunk]:
        """Get chunks for a document."""
        yql = f"select * from {self.schema_name} where document_id contains '{document_id}'"

        if chunk_range:
            start, end = chunk_range
            yql += f" and chunk_id >= {start} and chunk_id < {end}"

        yql += " order by chunk_id asc"

        body = {"yql": yql, "hits": 400}

        logger.debug(f"[VespaIndex] get_chunks_by_document: document_id={document_id}, yql={yql}")

        try:
            response = self.client.post("/search/", json=body)
            response.raise_for_status()
            data = response.json()

            hits = data.get("root", {}).get("children", [])
            return [self._parse_vespa_hit(hit) for hit in hits]
        except httpx.HTTPError as e:
            logger.error(f"[VespaIndex] get_chunks_by_document failed: yql={yql}, error={e}")
            raise DocumentIndexError(
                f"Vespa get chunks failed: {e}",
                index_type="vespa",
            )

    def id_based_retrieval(
        self,
        chunk_requests: list[ChunkRequest],
        batch_retrieval: bool = True,
    ) -> list[Chunk]:
        """Retrieve chunks by explicit id ranges using cleaned doc IDs."""
        results: list[Chunk] = []
        for req in chunk_requests:
            doc_id = replace_invalid_doc_id_characters(req.document_id)
            start = max(0, req.min_chunk_id)
            end = req.max_chunk_id + 1
            results.extend(self.get_chunks_by_document(doc_id, (start, end)))
        return results


    def get_chunk(
        self,
        document_id: str,
        chunk_id: int,
    ) -> Optional[Chunk]:
        """Get a specific chunk."""
        doc_id = f"{document_id}_{chunk_id}"

        try:
            response = self.client.get(f"/document/v1/chunk/chunk/docid/{doc_id}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()

            fields = data.get("fields", {})
            return Chunk(
                document_id=fields.get("document_id", ""),
                chunk_id=int(fields.get("chunk_id", 0)),
                content=fields.get("content", ""),
                embedding=fields.get("embedding"),
                title=fields.get("title"),
                source_type=fields.get("source_type"),
                link=fields.get("link"),
                metadata=fields.get("metadata", {}),
            )
        except httpx.HTTPError as e:
            raise DocumentIndexError(
                f"Vespa get chunk failed: {e}",
                index_type="vespa",
            )

    def index_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[str]:
        """Index chunks."""
        indexed_ids: list[str] = []

        logger.info(f"[DEBUG] VespaIndex.index_chunks: received {len(chunks)} chunks to index")

        for i, chunk in enumerate(chunks):
            doc_id = make_short_doc_id(chunk.document_id, chunk.chunk_id)

            fields = {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "title": chunk.title,
                "source_type": chunk.source_type,
                "source_links": chunk.link,
                "metadata": str(chunk.metadata) if chunk.metadata else "",
            }

            if chunk.embedding:
                emb_len = len(chunk.embedding) if isinstance(chunk.embedding, list) else 0
                fields["embeddings"] = {"values": chunk.embedding}
                logger.debug(f"[DEBUG] Chunk {i+1}/{len(chunks)} {doc_id}: embedding dimension={emb_len}")

                logger.debug(f"[DEBUG] Request URL: /document/v1/{self.schema_name}/{self.schema_name}/docid/{doc_id}")

            try:
                url = f"/document/v1/{self.schema_name}/{self.schema_name}/docid/{doc_id}"
                json_data = {"fields": fields}

                logger.debug(f"[DEBUG] Sending request to {url}")
                response = self.client.post(url, json=json_data, timeout=30.0)

                logger.debug(f"[DEBUG] Response status: {response.status_code}")
                response.raise_for_status()
                indexed_ids.append(doc_id)

                unique_id = f"{chunk.document_id}_{chunk.chunk_id}"
                self.chunks[unique_id] = chunk
                if chunk.document_id not in self.documents:
                    self.documents[chunk.document_id] = []
                if unique_id not in self.documents[chunk.document_id]:
                    self.documents[chunk.document_id].append(unique_id)

                logger.debug(f"[DEBUG] Successfully indexed chunk {doc_id}")
            except httpx.HTTPError as e:
                logger.error(f"[DEBUG] HTTP error indexing chunk {doc_id}: {e}")
                logger.error(f"[DEBUG] Error type: {type(e).__name__}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"[DEBUG] Vespa response status: {e.response.status_code}")
                    logger.error(f"[DEBUG] Vespa response text: {e.response.text[:500]}")
            except httpx.ConnectError as e:
                logger.error(f"[DEBUG] Connection error indexing chunk {doc_id}: {e}")
            except Exception as e:
                logger.error(f"[DEBUG] Unexpected error indexing chunk {doc_id}: {type(e).__name__}: {e}")

        logger.info(f"[DEBUG] VespaIndex.index_chunks: returning {len(indexed_ids)} indexed_ids")
        return indexed_ids

    def delete_document(
        self,
        document_id: str,
    ) -> bool:
        """Delete a document."""
        # First get all chunks for this document
        chunks = self.get_chunks_by_document(document_id)

        for chunk in chunks:
            self.delete_chunk(document_id, chunk.chunk_id)

        return True

    def delete_chunk(
        self,
        document_id: str,
        chunk_id: int,
    ) -> bool:
        """Delete a specific chunk."""
        doc_id = f"{document_id}_{chunk_id}"

        try:
            response = self.client.delete(f"/document/v1/chunk/chunk/docid/{doc_id}")
            return response.status_code == 200
        except httpx.HTTPError as e:
            logger.error(f"Failed to delete chunk {doc_id}: {e}")
            return False

    def close(self) -> None:
        """Close the client."""
        if self._client:
            self._client.close()
            self._client = None


def replace_invalid_doc_id_characters(doc_id: str) -> str:
    """Replace characters that can cause issues in Vespa docid."""
    return re.sub(r"[^a-zA-Z0-9_\\-\\.]", "_", doc_id)


def make_short_doc_id(document_id: str, chunk_id: int, max_length: int = 100) -> str:
    """Create a short, URL-safe document ID for Vespa.

    Uses MD5 hash of the full document_id to ensure URL safety and consistent length.
    """
    import hashlib

    # Create a short hash from the full document_id
    short_hash = hashlib.md5(document_id.encode('utf-8')).hexdigest()[:16]
    return f"doc_{short_hash}_{chunk_id}"
