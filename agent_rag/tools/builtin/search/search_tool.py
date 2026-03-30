"""Search tool implementing the 5-step search process."""

from dataclasses import dataclass
import json
from typing import Any, Optional

from agent_rag.core.config import SearchConfig
from agent_rag.core.models import Chunk, SearchFilters, Section
from agent_rag.document_index.interface import DocumentIndex
from agent_rag.embedding.interface import Embedder
from agent_rag.llm.interface import LLM
from agent_rag.retrieval.pipeline import RetrievalPipeline, QuerySpec
from agent_rag.tools.interface import Tool, ToolResponse
from agent_rag.tools.builtin.search.document_filter import (
    clean_up_source,
    select_sections_for_expansion,
)
from agent_rag.tools.builtin.search.query_expansion import (
    keyword_query_expansion,
    semantic_query_rephrase,
)
from agent_rag.tools.builtin.search.search_utils import (
    expand_section_with_context,
    merge_overlapping_sections,
    trim_sections_by_tokens,
)
from agent_rag.utils.concurrency import run_in_parallel
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchToolConfig:
    """Configuration for search tool."""
    document_index: DocumentIndex
    embedder: Embedder
    llm: Optional[LLM] = None  # For query expansion and selection
    search_config: Optional[SearchConfig] = None
    # Skip query expansion for repeat search calls in the same turn
    # This optimization saves ~3-4 LLM calls when search is called multiple times
    skip_query_expansion: bool = False


class SearchTool(Tool[SearchToolConfig]):
    """
    Internal document search tool implementing the 5-step search process:

    1. Query Generation - Generate multiple query variants
    2. Recombination - Merge results using weighted RRF
    3. Selection - LLM selects most relevant documents
    4. Expansion - Expand chunks to include surrounding context
    5. Prompt Building - Build response for LLM
    """

    NAME = "internal_search"
    DESCRIPTION = """Search the internal knowledge base for relevant documents.
Use this tool to find information from indexed documents like company wikis,
documentation, Confluence pages, Google Docs, etc."""

    def __init__(
        self,
        document_index: DocumentIndex,
        embedder: Embedder,
        llm: Optional[LLM] = None,
        search_config: Optional[SearchConfig] = None,
        id: Optional[int] = None,
    ) -> None:
        super().__init__(id)
        self.document_index = document_index
        self.embedder = embedder
        self.llm = llm
        self.search_config = search_config or SearchConfig()
        self.pipeline = RetrievalPipeline(
            document_index=document_index,
            embedder=embedder,
            config=self.search_config,
        )

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def description(self) -> str:
        return self.DESCRIPTION

    def tool_definition(self) -> dict[str, Any]:
        """Get tool definition."""
        return self.build_tool_definition(
            parameters={
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents",
                },
                "source_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of source types to filter (e.g., ['confluence', 'google_drive'])",
                },
            },
            required=["query"],
        )

    def run(
        self,
        override_kwargs: Optional[SearchToolConfig] = None,
        **llm_kwargs: Any,
    ) -> ToolResponse:
        """Execute the 5-step search process."""
        query = llm_kwargs.get("query", "")
        source_types = llm_kwargs.get("source_types")
        context = llm_kwargs.get("context") or {}
        history = llm_kwargs.get("history") or context.get("history")
        user_info = context.get("user_info")
        memories = context.get("memories")

        if not query:
            return ToolResponse(llm_response="Error: No query provided")

        # Use override config if provided
        config = override_kwargs
        document_index = config.document_index if config else self.document_index
        embedder = config.embedder if config else self.embedder
        llm = config.llm if config else self.llm
        search_config = config.search_config if config else self.search_config

        # Check skip_query_expansion from both config and context
        # Context flag takes precedence (allows ChatAgent to control per-call behavior)
        skip_query_expansion = context.get("skip_query_expansion", False)
        if config and config.skip_query_expansion:
            skip_query_expansion = True

        # Build filters
        filters = None
        if source_types:
            filters = SearchFilters(source_types=source_types)

        # Step 1: Query Generation (multi-query)
        logger.info(f"[SearchTool] Step 1: Query Generation - query='{query[:50]}...'")
        query_specs = self._build_query_specs(
            query=query,
            llm=llm,
            config=search_config,
            history=history,
            user_info=user_info,
            memories=memories,
            skip_query_expansion=skip_query_expansion,
        )
        logger.info(f"[SearchTool] Step 1 complete: generated {len(query_specs)} query specs")

        # Step 2: Recombination - handled by pipeline with weighted RRF
        pipeline = RetrievalPipeline(
            document_index=document_index,
            embedder=embedder,
            config=search_config,
        )

        result = pipeline.retrieve_multi(
            query_specs=query_specs,
            filters=filters,
        )

        if not result.chunks:
            return ToolResponse(
                llm_response="No relevant documents found for your query.",
                rich_response={"chunks": [], "sections": []},
            )

        # Step 3: Selection - LLM selects most relevant (if enabled)
        selected_chunks = result.chunks
        if llm and search_config.enable_document_selection:
            selected_chunks = self._select_documents(
                query=query,
                chunks=result.chunks,
                llm=llm,
                max_docs=search_config.max_documents_to_select,
            )
        if search_config.max_chunks_per_response and len(selected_chunks) > search_config.max_chunks_per_response:
            selected_chunks = selected_chunks[:search_config.max_chunks_per_response]

        # Step 4: Expansion - expand to sections
        sections = result.sections
        if not sections and search_config.enable_context_expansion:
            sections = pipeline.merge_adjacent_chunks(
                selected_chunks,
                max_gap=1,
            )
        if llm and sections and search_config.enable_context_expansion:
            selection = select_sections_for_expansion(
                sections=sections,
                query=query,
                llm=llm,
                max_sections=search_config.max_documents_to_select,
                max_chunks_per_section=search_config.max_chunks_for_relevance,
            )
            section_index_map = {id(section): idx for idx, section in enumerate(sections)}

            # Prepare function calls for parallel execution
            expansion_tasks: list[tuple[Any, tuple[Any, ...]]] = []
            full_doc_chunks = (
                search_config.max_full_document_chunks
                if search_config.max_full_document_chunks is not None
                else 5
            )
            for section in selection.sections:
                section_idx = section_index_map.get(id(section), -1)
                force_full = section_idx in selection.full_document_section_ids
                expansion_tasks.append((
                    expand_section_with_context,
                    (section, query, llm, document_index, full_doc_chunks, force_full),
                ))

            # Execute section expansion in parallel for better performance
            expanded_results = run_in_parallel(
                expansion_tasks,
                max_workers=min(len(expansion_tasks), search_config.section_expansion_workers),
                allow_failures=True,
            )
            expanded_sections = [r for r in expanded_results if r is not None]

            sections = merge_overlapping_sections(expanded_sections)
            sections = trim_sections_by_tokens(
                sections=sections,
                token_counter=llm.count_tokens,
                max_tokens=search_config.max_context_tokens,
                max_chunks_per_section=search_config.context_expansion_chunks + 1,
            )

        # Step 5: Prompt Building - build response
        response_text, citation_mapping = self._build_response(
            query=query,
            chunks=selected_chunks,
            sections=sections,
            max_tokens=search_config.max_context_tokens,
            max_content_chars=search_config.max_content_chars_per_chunk,
        )

        return ToolResponse(
            llm_response=response_text,
            rich_response={
                "chunks": selected_chunks,
                "sections": sections,
                "queries": result.queries_used,
            },
            citation_mapping=citation_mapping,
        )

    def _build_query_specs(
        self,
        query: str,
        llm: Optional[LLM],
        config: SearchConfig,
        history: Optional[Any] = None,
        user_info: Optional[str] = None,
        memories: Optional[list[str]] = None,
        skip_query_expansion: bool = False,
    ) -> list[QuerySpec]:
        """Build weighted query specs for retrieval.

        Args:
            query: Original search query
            llm: LLM for query expansion
            config: Search configuration
            history: Chat history for context
            user_info: User information
            memories: User memories
            skip_query_expansion: If True, skip LLM-based query expansion (saves ~3-4 LLM calls)
        """
        specs: list[QuerySpec] = [
            QuerySpec(
                query=query,
                weight=config.original_query_weight,
                hybrid_alpha=config.default_hybrid_alpha,
            )
        ]

        # Skip query expansion if:
        # 1. No LLM available
        # 2. Query expansion disabled in config
        # 3. skip_query_expansion flag is True (repeat search call optimization)
        if not llm or not config.enable_query_expansion or skip_query_expansion:
            if skip_query_expansion:
                logger.debug("[SearchTool] Skipping query expansion (repeat search call)")
            return specs

        # Run query expansion functions in PARALLEL for better performance
        # Follows Onyx pattern: semantic_query_rephrase + keyword_query_expansion
        # This reduces latency from sequential LLM calls to parallel execution
        logger.debug("[SearchTool] Running query expansion in parallel (2 LLM calls)")

        expansion_tasks: list[tuple[Any, tuple[Any, ...]]] = [
            # Task 1: Semantic query rephrase - reformulate into standalone query
            (
                semantic_query_rephrase,
                (query, llm, history, user_info, memories),
            ),
            # Task 2: Keyword query expansion - generate keyword-only queries
            (
                keyword_query_expansion,
                (query, llm, config.max_expanded_queries, history, user_info, memories),
            ),
        ]

        # Execute both expansion tasks in parallel
        expansion_results = run_in_parallel(
            expansion_tasks,
            max_workers=config.query_expansion_workers,
            allow_failures=True,  # Continue even if some expansions fail
        )

        # Process results: [semantic_query, keyword_queries]
        semantic_query = expansion_results[0]  # str | None
        keyword_queries = expansion_results[1] or []  # list[str]

        # Add semantic query if available and different from original
        if semantic_query and semantic_query.strip() != query.strip():
            specs.append(QuerySpec(
                query=semantic_query,
                weight=config.llm_semantic_query_weight,
                hybrid_alpha=config.default_hybrid_alpha,
            ))

        # Add keyword queries
        for kw in keyword_queries:
            # Skip if same as original query
            if kw.strip() != query.strip():
                specs.append(QuerySpec(
                    query=kw,
                    weight=config.llm_keyword_query_weight,
                    hybrid_alpha=config.keyword_query_hybrid_alpha,
                ))

        return specs

    def _select_documents(
        self,
        query: str,
        chunks: list[Chunk],
        llm: LLM,
        max_docs: int = 10,
    ) -> list[Chunk]:
        """Step 3: LLM-based document selection.

        Uses the DOCUMENT_SELECTION_PROMPT from prompts.py for consistency with Onyx.
        """
        if len(chunks) <= max_docs:
            return chunks

        from agent_rag.llm.interface import LLMMessage
        from agent_rag.tools.builtin.search.prompts import DOCUMENT_SELECTION_PROMPT

        # Build document sections for LLM
        # Format: [section_id] Title: ... | Content preview...
        doc_sections = []
        for i, chunk in enumerate(chunks[:20]):  # Limit to top 20 for LLM
            preview = chunk.content[:400] + "..." if len(chunk.content) > 400 else chunk.content
            title = chunk.semantic_identifier or chunk.title or "Untitled"
            source = clean_up_source(chunk.source_type)
            doc_sections.append(f"[{i}] {title} ({source}): {preview}")

        formatted_doc_sections = "\n\n".join(doc_sections)

        prompt = DOCUMENT_SELECTION_PROMPT.format(
            max_sections=max_docs,
            formatted_doc_sections=formatted_doc_sections,
            user_query=query,
        )

        messages = [
            LLMMessage(role="user", content=prompt),
        ]

        try:
            response = llm.chat(messages, max_tokens=100)
            # Parse response
            selected_indices = []
            for part in response.content.replace(",", " ").split():
                try:
                    idx = int(part.strip())
                    if 0 <= idx < len(chunks):
                        selected_indices.append(idx)
                except ValueError:
                    continue

            if selected_indices:
                return [chunks[i] for i in selected_indices[:max_docs]]
        except Exception as e:
            logger.warning(f"Document selection failed: {e}")

        return chunks[:max_docs]


    def _build_response(
        self,
        query: str,
        chunks: list[Chunk],
        sections: list[Section],
        max_tokens: int = 8000,
        max_content_chars: int = 800,
    ) -> tuple[str, dict[int, str]]:
        """Step 5: Build response text with citations.

        Args:
            query: The search query
            chunks: List of chunks to include
            sections: List of sections (for expanded content)
            max_tokens: Maximum total tokens for response (default 8000)
            max_content_chars: Maximum characters per document content (default 800)
        """
        citation_mapping: dict[int, str] = {}
        documents = []

        # Estimate tokens: ~4 chars per token for English, ~2 for Chinese
        estimated_tokens = 0
        overhead_per_doc = 150  # Metadata overhead per document

        for i, chunk in enumerate(chunks):
            citation_num = i + 1
            citation_mapping[citation_num] = chunk.unique_id

            section_content = None
            for section in sections:
                if section.center_chunk.unique_id == chunk.unique_id:
                    section_content = section.combined_content
                    break

            content = section_content or chunk.content
            # Truncate content to max_content_chars
            if len(content) > max_content_chars:
                content = content[:max_content_chars] + "..."

            # Estimate tokens for this document
            doc_tokens = len(content) // 2 + overhead_per_doc  # Conservative estimate

            # Check if adding this document would exceed limit
            if estimated_tokens + doc_tokens > max_tokens and documents:
                logger.info(f"Stopping at {len(documents)} docs due to token limit ({estimated_tokens} tokens)")
                break

            estimated_tokens += doc_tokens

            # Filter metadata to only include essential fields (avoid token explosion)
            filtered_metadata = {}
            if chunk.metadata:
                # Only keep small, essential metadata fields
                safe_keys = {"file_type", "page", "section", "language", "created_at", "modified_at"}
                for key in safe_keys:
                    if key in chunk.metadata:
                        value = chunk.metadata[key]
                        # Only include if value is small (string < 100 chars or number)
                        if isinstance(value, (int, float, bool)) or (isinstance(value, str) and len(value) < 100):
                            filtered_metadata[key] = value

            documents.append({
                "citation_id": citation_num,
                "title": chunk.semantic_identifier or chunk.title or "Untitled",
                "source_type": clean_up_source(chunk.source_type),
                "link": chunk.link,
                "updated_at": chunk.updated_at.strftime("%B %d, %Y %H:%M") if chunk.updated_at else None,
                "authors": (chunk.primary_owners + chunk.secondary_owners) or None,
                "metadata": filtered_metadata if filtered_metadata else None,
                "content": content,
            })

        response_payload = {
            "query": query,
            "total_documents": len(documents),
            "documents": documents,
        }

        return json.dumps(response_payload, ensure_ascii=False, indent=2), citation_mapping
