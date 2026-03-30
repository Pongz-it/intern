"""Enhanced Vespa-based document index with full feature support."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator, Iterator, Optional

import httpx

from agent_rag.core.config import DocumentIndexConfig
from agent_rag.core.exceptions import DocumentIndexError
from agent_rag.core.models import Chunk, KGRelationship, SearchFilters
from agent_rag.document_index.interface import DocumentIndex
from agent_rag.document_index.vespa.schema_config import VespaSchemaConfig
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
NUM_THREADS = 32
# Separator used in indexed content (title/content separation)
RETURN_SEPARATOR = "\n"
# Blurb size for partial title matching
BLURB_SIZE = 128
BATCH_SIZE = 128
MAX_RETRY_ATTEMPTS = 5
INITIAL_RETRY_DELAY = 1.0
MAX_OR_CONDITIONS = 10


@dataclass
class IndexingResult:
    """Result of an indexing operation."""
    indexed_ids: list[str]
    failed_ids: list[str]
    error_messages: dict[str, str]


@dataclass
class VisitResult:
    """Result of a visit operation."""
    chunks: list[Chunk]
    continuation_token: Optional[str]
    total_count: int


class EnhancedVespaIndex(DocumentIndex):
    """
    Enhanced Vespa-based document index with production features.

    Features:
    - Parallel indexing with ThreadPool
    - Visit API for large-scale traversal
    - Dynamic highlighting
    - Multi-embedding support
    - Knowledge graph fields
    - Boost and recency ranking
    - Rate limit handling with exponential backoff
    """

    def __init__(
        self,
        config: Optional[DocumentIndexConfig] = None,
        schema_config: Optional[VespaSchemaConfig] = None,
        host: str = "localhost",
        port: int = 8080,
        app_name: str = "agent_rag",
        timeout: int = 30,
        num_threads: int = NUM_THREADS,
    ) -> None:
        # Vespa connection config
        if config:
            self.host = config.vespa_host
            self.port = config.vespa_port
            self.app_name = config.vespa_app_name
            self.timeout = config.vespa_timeout
        else:
            self.host = host
            self.port = port
            self.app_name = app_name
            self.timeout = timeout

        self.base_url = f"http://{self.host}:{self.port}"
        self._client: Optional[httpx.Client] = None

        # Schema configuration
        self.schema_config = schema_config or VespaSchemaConfig()
        self.schema_name = self.schema_config.schema_name

        # Parallel processing
        self.num_threads = num_threads
        self._executor: Optional[ThreadPoolExecutor] = None

    @property
    def client(self) -> httpx.Client:
        """Get HTTP client with connection pooling and proxy fix."""
        if self._client is None:
            transport = httpx.HTTPTransport(local_address="0.0.0.0")
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                http2=True,
                transport=transport,
                follow_redirects=False,
            )
        return self._client

    @property
    def executor(self) -> ThreadPoolExecutor:
        """Get thread pool executor for parallel operations."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.num_threads)
        return self._executor

    # ========== Query Building ==========

    def _build_yql_query(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[list[float]] = None,
        title_embedding: Optional[list[float]] = None,
        filters: Optional[SearchFilters] = None,
        hybrid_alpha: float = 0.5,
        title_content_ratio: float = 0.2,
        decay_factor: float = 0.5,
        num_results: int = 10,
        include_hidden: bool = False,
        tenant_id: Optional[str] = None,
        ranking_profile: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build Vespa YQL query with all features."""
        conditions = []

        # Hidden filter
        if not include_hidden:
            conditions.append("hidden = false")

        # Tenant filter
        if tenant_id:
            conditions.append(f'tenant_id contains "{tenant_id}"')

        # Custom filters
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
                source_filters = [f'source_type contains "{st}"' for st in filters.source_types[:MAX_OR_CONDITIONS]]
                conditions.append(f"({' OR '.join(source_filters)})")

            if filters.document_ids:
                doc_filters = [f'document_id contains "{did}"' for did in filters.document_ids[:MAX_OR_CONDITIONS]]
                conditions.append(f"({' OR '.join(doc_filters)})")

            if filters.tags:
                tag_filters = [f'metadata_list contains "{tag}"' for tag in filters.tags[:MAX_OR_CONDITIONS]]
                conditions.append(f"({' OR '.join(tag_filters)})")

            if isinstance(document_sets, list) and document_sets:
                doc_set_filters = [
                    f'document_sets contains "{doc_set}"'
                    for doc_set in document_sets[:MAX_OR_CONDITIONS]
                ]
                conditions.append(f"({' OR '.join(doc_set_filters)})")

            if user_folder is not None:
                try:
                    folder_id = int(user_folder)
                    conditions.append(f"(user_folder = {folder_id})")
                except (TypeError, ValueError):
                    pass

            if isinstance(user_project, list) and user_project:
                project_filters = [
                    f'user_project contains "{project_id}"'
                    for project_id in user_project[:MAX_OR_CONDITIONS]
                ]
                conditions.append(f"({' OR '.join(project_filters)})")

            if filters.time_cutoff:
                cutoff_ts = int(filters.time_cutoff.timestamp())
                conditions.append(f"doc_updated_at >= {cutoff_ts}")

        where_clause = " AND ".join(conditions) if conditions else "true"

        # Build YQL
        yql_parts = [f"select * from {self.schema_name} where {where_clause}"]

        if query:
            yql_parts.append(f'and ({{grammar: "weakAnd"}}userInput(@query))')

        yql = " ".join(yql_parts)

        # Determine ranking profile
        if not ranking_profile:
            dim = self.schema_config.dim
            if query_embedding:
                ranking_profile = f"hybrid_search_semantic_base_{dim}"
            elif query:
                ranking_profile = f"hybrid_search_keyword_base_{dim}"
            else:
                ranking_profile = "bm25_only"

        # Build request body
        body: dict[str, Any] = {
            "yql": yql,
            "hits": num_results,
            "ranking": ranking_profile,
            "ranking.properties.decay_factor": decay_factor,
        }

        # Query embedding
        if query_embedding:
            body["ranking.features.query(query_embedding)"] = query_embedding
            body["ranking.properties.alpha"] = hybrid_alpha
            body["ranking.properties.title_content_ratio"] = title_content_ratio

        # Keyword query
        if query:
            body["query"] = query

        return body

    # ========== Search Methods ==========

    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        filters: Optional[SearchFilters] = None,
        hybrid_alpha: float = 0.5,
        num_results: int = 10,
        include_highlights: bool = True,
        tenant_id: Optional[str] = None,
    ) -> list[Chunk]:
        """
        Perform hybrid search combining semantic and keyword matching.

        Args:
            query: Search query text
            query_embedding: Query embedding vector
            filters: Optional search filters
            hybrid_alpha: Balance between semantic (1.0) and keyword (0.0)
            num_results: Number of results to return
            include_highlights: Whether to include match highlights
            tenant_id: Optional tenant ID for multi-tenant filtering

        Returns:
            List of matching chunks with scores
        """
        body = self._build_yql_query(
            query=query,
            query_embedding=query_embedding,
            filters=filters,
            hybrid_alpha=hybrid_alpha,
            num_results=num_results,
            tenant_id=tenant_id,
        )

        # Enable dynamic summary for highlights
        if include_highlights:
            body["presentation.summary"] = "default"

        try:
            response = self.client.post("/search/", json=body)
            response.raise_for_status()
            data = response.json()

            hits = data.get("root", {}).get("children", [])
            return [self._parse_vespa_hit(hit, include_highlights) for hit in hits]
        except httpx.HTTPError as e:
            raise DocumentIndexError(
                f"Vespa hybrid search failed: {e}",
                index_type="vespa",
            )

    def semantic_search(
        self,
        query_embedding: list[float],
        filters: Optional[SearchFilters] = None,
        num_results: int = 10,
        tenant_id: Optional[str] = None,
    ) -> list[Chunk]:
        """Perform pure semantic search."""
        body = self._build_yql_query(
            query_embedding=query_embedding,
            filters=filters,
            hybrid_alpha=1.0,
            num_results=num_results,
            tenant_id=tenant_id,
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
        tenant_id: Optional[str] = None,
    ) -> list[Chunk]:
        """Perform pure keyword (BM25) search."""
        body = self._build_yql_query(
            query=query,
            filters=filters,
            hybrid_alpha=0.0,
            num_results=num_results,
            tenant_id=tenant_id,
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

    def admin_search(
        self,
        query: str,
        num_results: int = 10,
        include_hidden: bool = True,
        tenant_id: Optional[str] = None,
    ) -> list[Chunk]:
        """Admin search with title-heavy ranking and hidden document access."""
        body = self._build_yql_query(
            query=query,
            num_results=num_results,
            include_hidden=include_hidden,
            tenant_id=tenant_id,
            ranking_profile="admin_search",
        )

        try:
            response = self.client.post("/search/", json=body)
            response.raise_for_status()
            data = response.json()

            hits = data.get("root", {}).get("children", [])
            return [self._parse_vespa_hit(hit, include_highlights=True) for hit in hits]
        except httpx.HTTPError as e:
            raise DocumentIndexError(
                f"Vespa admin search failed: {e}",
                index_type="vespa",
            )

    def random_search(
        self,
        filters: Optional[SearchFilters] = None,
        num_results: int = 10,
        tenant_id: Optional[str] = None,
    ) -> list[Chunk]:
        """Get random chunks for sampling purposes."""
        body = self._build_yql_query(
            filters=filters,
            num_results=num_results,
            tenant_id=tenant_id,
            ranking_profile="random_rank",
        )

        try:
            response = self.client.post("/search/", json=body)
            response.raise_for_status()
            data = response.json()

            hits = data.get("root", {}).get("children", [])
            return [self._parse_vespa_hit(hit) for hit in hits]
        except httpx.HTTPError as e:
            raise DocumentIndexError(
                f"Vespa random search failed: {e}",
                index_type="vespa",
            )

    # ========== Document Retrieval ==========

    def get_chunks_by_document(
        self,
        document_id: str,
        chunk_range: Optional[tuple[int, int]] = None,
        exclude_large_chunks: bool = False,
    ) -> list[Chunk]:
        """Get chunks for a document."""
        conditions = [f'document_id contains "{document_id}"']

        if chunk_range:
            start, end = chunk_range
            conditions.append(f"chunk_id >= {start}")
            conditions.append(f"chunk_id < {end}")

        if exclude_large_chunks:
            conditions.append("large_chunk_reference_ids = null")

        yql = f"select * from {self.schema_name} where {' AND '.join(conditions)} order by chunk_id asc"
        body = {"yql": yql, "hits": 10000}

        try:
            response = self.client.post("/search/", json=body)
            response.raise_for_status()
            data = response.json()

            hits = data.get("root", {}).get("children", [])
            return [self._parse_vespa_hit(hit) for hit in hits]
        except httpx.HTTPError as e:
            raise DocumentIndexError(
                f"Vespa get chunks failed: {e}",
                index_type="vespa",
            )

    def get_chunk(
        self,
        document_id: str,
        chunk_id: int,
    ) -> Optional[Chunk]:
        """Get a specific chunk."""
        doc_id = f"{document_id}_{chunk_id}"

        try:
            response = self.client.get(
                f"/document/v1/{self.schema_name}/{self.schema_name}/docid/{doc_id}"
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()

            return self._parse_document_response(data)
        except httpx.HTTPError as e:
            raise DocumentIndexError(
                f"Vespa get chunk failed: {e}",
                index_type="vespa",
            )

    # ========== Visit API for Large-Scale Traversal ==========

    def visit_documents(
        self,
        selection: Optional[str] = None,
        fields_to_include: Optional[list[str]] = None,
        continuation_token: Optional[str] = None,
        wanted_document_count: int = 1000,
        timeout_ms: int = 60000,
    ) -> VisitResult:
        """
        Visit documents using Vespa Visit API for large-scale traversal.

        Args:
            selection: Document selection expression
            fields_to_include: Fields to return (None for all)
            continuation_token: Token for pagination
            wanted_document_count: Number of documents per page
            timeout_ms: Timeout in milliseconds

        Returns:
            VisitResult with chunks and continuation token
        """
        params: dict[str, Any] = {
            "wantedDocumentCount": wanted_document_count,
            "timeout": f"{timeout_ms}ms",
        }

        if selection:
            params["selection"] = selection
        if fields_to_include:
            params["fieldSet"] = f"{self.schema_name}:[{','.join(fields_to_include)}]"
        if continuation_token:
            params["continuation"] = continuation_token

        try:
            response = self.client.get(
                f"/document/v1/{self.schema_name}/{self.schema_name}/docid/",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            chunks = []
            for doc in data.get("documents", []):
                chunk = self._parse_document_response(doc)
                if chunk:
                    chunks.append(chunk)

            return VisitResult(
                chunks=chunks,
                continuation_token=data.get("continuation"),
                total_count=len(chunks),
            )
        except httpx.HTTPError as e:
            raise DocumentIndexError(
                f"Vespa visit failed: {e}",
                index_type="vespa",
            )

    def visit_all_documents(
        self,
        selection: Optional[str] = None,
        fields_to_include: Optional[list[str]] = None,
        batch_size: int = 1000,
    ) -> Generator[Chunk, None, None]:
        """
        Generator to iterate over all documents.

        Args:
            selection: Document selection expression
            fields_to_include: Fields to return
            batch_size: Number of documents per batch

        Yields:
            Chunk objects
        """
        continuation_token = None

        while True:
            result = self.visit_documents(
                selection=selection,
                fields_to_include=fields_to_include,
                continuation_token=continuation_token,
                wanted_document_count=batch_size,
            )

            for chunk in result.chunks:
                yield chunk

            if not result.continuation_token:
                break

            continuation_token = result.continuation_token

    # ========== Parallel Indexing ==========

    def index_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = BATCH_SIZE,
    ) -> list[str]:
        """Index chunks (sequential fallback)."""
        result = self.index_chunks_parallel(chunks, batch_size)
        return result.indexed_ids

    def index_chunks_parallel(
        self,
        chunks: list[Chunk],
        batch_size: int = BATCH_SIZE,
    ) -> IndexingResult:
        """
        Index chunks in parallel using ThreadPool.

        Args:
            chunks: Chunks to index
            batch_size: Batch size for grouping

        Returns:
            IndexingResult with success/failure counts
        """
        indexed_ids: list[str] = []
        failed_ids: list[str] = []
        error_messages: dict[str, str] = {}

        def index_single_chunk(chunk: Chunk) -> tuple[str, bool, Optional[str]]:
            """Index a single chunk with retry logic."""
            doc_id = chunk.unique_id
            fields = chunk.to_vespa_fields()

            for attempt in range(MAX_RETRY_ATTEMPTS):
                try:
                    response = self.client.post(
                        f"/document/v1/{self.schema_name}/{self.schema_name}/docid/{doc_id}",
                        json={"fields": fields},
                    )

                    if response.status_code == 429:
                        # Rate limited - exponential backoff
                        delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                        time.sleep(delay)
                        continue

                    response.raise_for_status()
                    return (doc_id, True, None)

                except httpx.HTTPError as e:
                    if attempt == MAX_RETRY_ATTEMPTS - 1:
                        return (doc_id, False, str(e))
                    delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                    time.sleep(delay)

            return (doc_id, False, "Max retries exceeded")

        # Submit all indexing tasks
        futures = {
            self.executor.submit(index_single_chunk, chunk): chunk
            for chunk in chunks
        }

        # Collect results
        for future in as_completed(futures):
            doc_id, success, error = future.result()
            if success:
                indexed_ids.append(doc_id)
            else:
                failed_ids.append(doc_id)
                if error:
                    error_messages[doc_id] = error

        logger.info(
            f"Indexed {len(indexed_ids)} chunks, {len(failed_ids)} failed"
        )

        return IndexingResult(
            indexed_ids=indexed_ids,
            failed_ids=failed_ids,
            error_messages=error_messages,
        )

    # ========== Update Operations ==========

    def update_chunk_boost(
        self,
        document_id: str,
        chunk_id: int,
        boost: float,
    ) -> bool:
        """Update boost value for a chunk."""
        doc_id = f"{document_id}_{chunk_id}"

        try:
            response = self.client.put(
                f"/document/v1/{self.schema_name}/{self.schema_name}/docid/{doc_id}",
                json={"fields": {"boost": {"assign": boost}}},
            )
            return response.status_code == 200
        except httpx.HTTPError as e:
            logger.error(f"Failed to update boost for {doc_id}: {e}")
            return False

    def update_chunk_hidden(
        self,
        document_id: str,
        chunk_id: int,
        hidden: bool,
    ) -> bool:
        """Update hidden status for a chunk."""
        doc_id = f"{document_id}_{chunk_id}"

        try:
            response = self.client.put(
                f"/document/v1/{self.schema_name}/{self.schema_name}/docid/{doc_id}",
                json={"fields": {"hidden": {"assign": hidden}}},
            )
            return response.status_code == 200
        except httpx.HTTPError as e:
            logger.error(f"Failed to update hidden for {doc_id}: {e}")
            return False

    def update_knowledge_graph_fields(
        self,
        document_id: str,
        chunk_id: int,
        kg_entities: Optional[list[str]] = None,
        kg_relationships: Optional[list[KGRelationship]] = None,
        kg_terms: Optional[list[str]] = None,
    ) -> bool:
        """Update knowledge graph fields for a chunk."""
        doc_id = f"{document_id}_{chunk_id}"
        updates: dict[str, Any] = {}

        if kg_entities is not None:
            updates["kg_entities"] = {"assign": kg_entities}
        if kg_relationships is not None:
            updates["kg_relationships"] = {"assign": [r.to_dict() for r in kg_relationships]}
        if kg_terms is not None:
            updates["kg_terms"] = {"assign": kg_terms}

        if not updates:
            return True

        try:
            response = self.client.put(
                f"/document/v1/{self.schema_name}/{self.schema_name}/docid/{doc_id}",
                json={"fields": updates},
            )
            return response.status_code == 200
        except httpx.HTTPError as e:
            logger.error(f"Failed to update KG fields for {doc_id}: {e}")
            return False

    # ========== Delete Operations ==========

    def delete_document(
        self,
        document_id: str,
    ) -> bool:
        """Delete a document and all its chunks."""
        chunks = self.get_chunks_by_document(document_id)

        def delete_chunk_task(chunk: Chunk) -> bool:
            return self.delete_chunk(document_id, chunk.chunk_id)

        # Delete in parallel
        futures = [
            self.executor.submit(delete_chunk_task, chunk)
            for chunk in chunks
        ]

        results = [f.result() for f in as_completed(futures)]
        return all(results)

    def delete_chunk(
        self,
        document_id: str,
        chunk_id: int,
    ) -> bool:
        """Delete a specific chunk."""
        doc_id = f"{document_id}_{chunk_id}"

        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                response = self.client.delete(
                    f"/document/v1/{self.schema_name}/{self.schema_name}/docid/{doc_id}"
                )
                return response.status_code == 200
            except httpx.HTTPError as e:
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    logger.error(f"Failed to delete chunk {doc_id}: {e}")
                    return False
                time.sleep(INITIAL_RETRY_DELAY * (2 ** attempt))

        return False

    # ========== Parsing Helpers ==========

    def _parse_vespa_hit(
        self,
        hit: dict[str, Any],
        include_highlights: bool = False,
    ) -> Chunk:
        """Parse Vespa search hit to Chunk."""
        fields = hit.get("fields", {})

        # Extract highlights
        match_highlights = []
        content_summary = None
        if include_highlights:
            summary_features = hit.get("summaryfeatures", {})
            if "content_summary" in summary_features:
                content_summary = summary_features["content_summary"]
                # Extract highlighted segments
                if "<sep />" in content_summary:
                    match_highlights = [
                        s.strip() for s in content_summary.split("<sep />")
                        if s.strip()
                    ]

        # Parse KG relationships
        kg_relationships = []
        for rel in fields.get("kg_relationships", []):
            if isinstance(rel, dict):
                kg_relationships.append(KGRelationship(
                    source=rel.get("source", ""),
                    rel_type=rel.get("rel_type", ""),
                    target=rel.get("target", ""),
                ))

        return Chunk(
            document_id=fields.get("document_id", ""),
            chunk_id=int(fields.get("chunk_id", 0)),
            content=fields.get("content", ""),
            title=fields.get("title"),
            source_type=fields.get("source_type"),
            link=fields.get("source_links"),
            semantic_identifier=fields.get("semantic_identifier"),
            blurb=fields.get("blurb"),
            metadata_suffix=fields.get("metadata_suffix"),
            metadata_list=fields.get("metadata_list", []),
            doc_summary=fields.get("doc_summary"),
            chunk_context=fields.get("chunk_context"),
            large_chunk_reference_ids=fields.get("large_chunk_reference_ids", []),
            kg_entities=fields.get("kg_entities", []),
            kg_relationships=kg_relationships,
            kg_terms=fields.get("kg_terms", []),
            boost=fields.get("boost", 0.0),
            aggregated_chunk_boost_factor=fields.get("aggregated_chunk_boost_factor", 1.0),
            hidden=fields.get("hidden", False),
            primary_owners=fields.get("primary_owners", []),
            secondary_owners=fields.get("secondary_owners", []),
            doc_updated_at=fields.get("doc_updated_at"),
            score=hit.get("relevance", 0.0),
            match_highlights=match_highlights,
            content_summary=content_summary,
            tenant_id=fields.get("tenant_id"),
            section_continuation=fields.get("section_continuation", False),
            image_file_name=fields.get("image_file_name"),
        )

    def _parse_document_response(
        self,
        data: dict[str, Any],
    ) -> Optional[Chunk]:
        """Parse Vespa document API response to Chunk."""
        fields = data.get("fields", {})
        if not fields:
            return None

        # Parse KG relationships
        kg_relationships = []
        for rel in fields.get("kg_relationships", []):
            if isinstance(rel, dict):
                kg_relationships.append(KGRelationship(
                    source=rel.get("source", ""),
                    rel_type=rel.get("rel_type", ""),
                    target=rel.get("target", ""),
                ))

        return Chunk(
            document_id=fields.get("document_id", ""),
            chunk_id=int(fields.get("chunk_id", 0)),
            content=fields.get("content", ""),
            title=fields.get("title"),
            source_type=fields.get("source_type"),
            link=fields.get("source_links"),
            semantic_identifier=fields.get("semantic_identifier"),
            blurb=fields.get("blurb"),
            metadata_suffix=fields.get("metadata_suffix"),
            metadata_list=fields.get("metadata_list", []),
            doc_summary=fields.get("doc_summary"),
            chunk_context=fields.get("chunk_context"),
            large_chunk_reference_ids=fields.get("large_chunk_reference_ids", []),
            kg_entities=fields.get("kg_entities", []),
            kg_relationships=kg_relationships,
            kg_terms=fields.get("kg_terms", []),
            boost=fields.get("boost", 0.0),
            aggregated_chunk_boost_factor=fields.get("aggregated_chunk_boost_factor", 1.0),
            hidden=fields.get("hidden", False),
            primary_owners=fields.get("primary_owners", []),
            secondary_owners=fields.get("secondary_owners", []),
            doc_updated_at=fields.get("doc_updated_at"),
            tenant_id=fields.get("tenant_id"),
            section_continuation=fields.get("section_continuation", False),
            image_file_name=fields.get("image_file_name"),
        )

    # ========== Chunk Cleanup ==========

    def cleanup_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Removes indexing-time content additions from chunks retrieved from Vespa.

        During indexing, chunks are augmented with additional text to improve search
        quality:
        - Title prepended to content (for better keyword/semantic matching)
        - Metadata suffix appended to content
        - Contextual RAG: doc_summary (beginning) and chunk_context (end)

        This function strips these additions before returning chunks to users,
        restoring the original document content.

        Args:
            chunks: List of chunks with potentially augmented content

        Returns:
            Clean Chunk objects with augmentations removed, containing only
            the original document content that should be shown to users.
        """
        cleaned_chunks = []

        for chunk in chunks:
            # Create a copy to avoid modifying original
            content = chunk.content

            # 1. Remove title prefix
            content = self._remove_title_from_content(content, chunk.title)

            # 2. Remove metadata suffix
            content = self._remove_metadata_suffix_from_content(content, chunk.metadata_suffix)

            # 3. Remove contextual RAG additions
            content = self._remove_contextual_rag_from_content(
                content, chunk.doc_summary, chunk.chunk_context
            )

            # Create cleaned chunk
            cleaned_chunk = Chunk(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                content=content,
                embedding=chunk.embedding,
                title=chunk.title,
                source_type=chunk.source_type,
                link=chunk.link,
                metadata=chunk.metadata,
                semantic_identifier=chunk.semantic_identifier,
                metadata_suffix=chunk.metadata_suffix,
                metadata_list=chunk.metadata_list,
                blurb=chunk.blurb,
                section_continuation=chunk.section_continuation,
                title_embedding=chunk.title_embedding,
                embeddings=chunk.embeddings,
                skip_title_embedding=chunk.skip_title_embedding,
                large_chunk_reference_ids=chunk.large_chunk_reference_ids,
                kg_entities=chunk.kg_entities,
                kg_relationships=chunk.kg_relationships,
                kg_terms=chunk.kg_terms,
                doc_summary=chunk.doc_summary,
                chunk_context=chunk.chunk_context,
                boost=chunk.boost,
                aggregated_chunk_boost_factor=chunk.aggregated_chunk_boost_factor,
                hidden=chunk.hidden,
                primary_owners=chunk.primary_owners,
                secondary_owners=chunk.secondary_owners,
                score=chunk.score,
                match_highlights=chunk.match_highlights,
                content_summary=chunk.content_summary,
                created_at=chunk.created_at,
                updated_at=chunk.updated_at,
                doc_updated_at=chunk.doc_updated_at,
                tenant_id=chunk.tenant_id,
                image_file_name=chunk.image_file_name,
            )
            cleaned_chunks.append(cleaned_chunk)

        return cleaned_chunks

    def _remove_title_from_content(
        self,
        content: str,
        title: Optional[str],
    ) -> str:
        """Remove title prefix from content."""
        if not title or not content:
            return content

        # Full match: exact title at the beginning
        if content.startswith(title):
            return content[len(title):].lstrip()

        # Partial match: content starts with truncated title
        # If title was truncated to BLURB_SIZE during indexing
        if len(title) > BLURB_SIZE and content.startswith(title[:BLURB_SIZE]):
            # Split on separator to remove title section
            if RETURN_SEPARATOR in content:
                return content.split(RETURN_SEPARATOR, 1)[1].lstrip()

        return content

    def _remove_metadata_suffix_from_content(
        self,
        content: str,
        metadata_suffix: Optional[str],
    ) -> str:
        """Remove metadata suffix from content."""
        if not metadata_suffix or not content:
            return content

        if content.endswith(metadata_suffix):
            return content.removesuffix(metadata_suffix).rstrip(RETURN_SEPARATOR)

        return content

    def _remove_contextual_rag_from_content(
        self,
        content: str,
        doc_summary: Optional[str],
        chunk_context: Optional[str],
    ) -> str:
        """Remove contextual RAG additions (doc_summary and chunk_context)."""
        if not content:
            return content

        # Remove document summary from beginning
        if doc_summary and content.startswith(doc_summary):
            content = content[len(doc_summary):].lstrip()

        # Remove chunk context from end
        if chunk_context and content.endswith(chunk_context):
            content = content[:-len(chunk_context)].rstrip()

        return content

    # ========== Cleanup ==========

    def close(self) -> None:
        """Close client and executor."""
        if self._client:
            self._client.close()
            self._client = None

        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
