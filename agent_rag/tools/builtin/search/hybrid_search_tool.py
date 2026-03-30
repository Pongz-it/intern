"""Hybrid search tool with Text-to-SQL support.

Extends the SearchTool to support:
- Intent analysis for data queries
- SQL generation and execution for structured data
- Hybrid retrieval combining document and database results
"""

from dataclasses import dataclass
from typing import Any, Optional

from agent_rag.core.config import SearchConfig
from agent_rag.core.models import Chunk, SearchFilters, Section
from agent_rag.document_index.interface import DocumentIndex
from agent_rag.embedding.interface import Embedder
from agent_rag.llm.interface import LLM
from agent_rag.retrieval.pipeline import RetrievalPipeline, QuerySpec
from agent_rag.text_to_sql import TextToSQL, TextToSQLResult
from agent_rag.tools.interface import Tool, ToolResponse
from agent_rag.tools.builtin.search.document_filter import (
    clean_up_source,
    select_sections_for_expansion,
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
class HybridSearchConfig:
    """Configuration for hybrid search with Text-to-SQL."""

    search_config: SearchConfig
    enable_text_to_sql: bool = True
    text_to_sql_threshold: float = 0.7
    sql_result_weight: float = 0.6
    text_result_weight: float = 0.4
    max_sql_results: int = 10


class HybridSearchTool(Tool[HybridSearchConfig]):
    """
    Hybrid search tool combining document search and Text-to-SQL.

    Process:
    1. Intent Analysis - Detect if query is asking for data
    2. Query Generation - Generate query variants
    3. Hybrid Retrieval - Search documents + execute SQL
    4. Result Fusion - Merge results using weighted fusion
    5. Selection/Expansion/Prompt Building
    """

    NAME = "hybrid_search"
    DESCRIPTION = """Search the knowledge base for relevant information.
Supports both document search and data queries from the database.
Use this for finding information from indexed documents or querying structured data."""

    def __init__(
        self,
        document_index: DocumentIndex,
        embedder: Embedder,
        text_to_sql: Optional[TextToSQL] = None,
        llm: Optional[LLM] = None,
        search_config: Optional[SearchConfig] = None,
        hybrid_config: Optional[HybridSearchConfig] = None,
        id: Optional[int] = None,
    ) -> None:
        super().__init__(id)
        self.document_index = document_index
        self.embedder = embedder
        self.text_to_sql = text_to_sql
        self.llm = llm
        self.search_config = search_config or SearchConfig()
        self.hybrid_config = hybrid_config or HybridSearchConfig(
            search_config=self.search_config
        )

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
                    "description": "The search query or data question",
                },
                "source_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of source types to filter",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["auto", "documents", "data"],
                    "description": "Search type: auto-detect, documents only, or data only",
                    "default": "auto",
                },
            },
            required=["query"],
        )

    def run(
        self,
        override_kwargs: Optional[HybridSearchConfig] = None,
        **llm_kwargs: Any,
    ) -> ToolResponse:
        """Execute hybrid search process."""
        import asyncio

        return asyncio.run(self._run_async(override_kwargs, **llm_kwargs))

    async def _run_async(
        self,
        override_kwargs: Optional[HybridSearchConfig] = None,
        **llm_kwargs: Any,
    ) -> ToolResponse:
        """Execute hybrid search process (async version)."""
        query = llm_kwargs.get("query", "")
        source_types = llm_kwargs.get("source_types")
        search_type = llm_kwargs.get("search_type", "auto")
        context = llm_kwargs.get("context") or {}
        history = llm_kwargs.get("history") or context.get("history")
        user_info = context.get("user_info")
        memories = context.get("memories")

        if not query:
            return ToolResponse(llm_response="Error: No query provided")

        config = override_kwargs
        document_index = config.document_index if config else self.document_index
        embedder = config.embedder if config else self.embedder
        llm = config.llm if config else self.llm
        search_config = config.search_config if config else self.search_config
        hybrid_config = config if config else self.hybrid_config

        filters = None
        if source_types:
            filters = SearchFilters(source_types=source_types)

        document_results = []
        sql_result = None
        is_data_query = False
        injected_context = {}

        if hybrid_config.enable_text_to_sql and self.text_to_sql:
            logger.info(f"[HybridSearch] Step 0: Intent Analysis - query='{query[:50]}...'")
            
            if hasattr(self.text_to_sql, 'should_use_hybrid_search') and self.text_to_sql.should_use_hybrid_search(query):
                logger.info(f"[HybridSearch] Database keywords detected in query, forcing hybrid search")
                search_type = "auto"
                injected_context = self.text_to_sql.get_injected_context(query)
                logger.info(f"[HybridSearch] Injected context: tables={injected_context.get('suggested_tables', [])}, "
                           f"columns={len(injected_context.get('suggested_columns', []))}")
            
            text_sql_result = await self.text_to_sql.execute(query)
            is_data_query = text_sql_result.is_data_query
            sql_result = text_sql_result

            if search_type == "data" and is_data_query:
                return self._build_sql_response(text_sql_result)

            if search_type == "auto" and is_data_query and text_sql_result.confidence >= hybrid_config.text_to_sql_threshold:
                logger.info(f"[HybridSearch] Data query detected (confidence: {text_sql_result.confidence:.2f})")

        if search_type in ("auto", "documents"):
            logger.info(f"[HybridSearch] Step 1: Query Generation - query='{query[:50]}...'")

            skip_query_expansion = context.get("skip_query_expansion", False)
            query_specs = self._build_query_specs(
                query=query,
                llm=llm,
                config=search_config,
                history=history,
                user_info=user_info,
                memories=memories,
                skip_query_expansion=skip_query_expansion,
            )

            logger.info(f"[HybridSearch] Step 2: Document Retrieval")

            result = self.pipeline.retrieve_multi(
                query_specs=query_specs,
                filters=filters,
            )
            document_results = result.chunks

        return self._build_hybrid_response(
            query=query,
            document_results=document_results,
            sql_result=sql_result,
            is_data_query=is_data_query,
            document_index=document_index,
            embedder=embedder,
            llm=llm,
            search_config=search_config,
            hybrid_config=hybrid_config,
            context=context,
            history=history,
            user_info=user_info,
            memories=memories,
        )

    def _build_sql_response(self, result: TextToSQLResult) -> ToolResponse:
        """Build response for SQL-only query."""
        if result.error_message:
            return ToolResponse(
                llm_response=f"Could not answer data query: {result.error_message}",
                rich_response={
                    "sql": result.sql,
                    "error": result.error_message,
                    "is_data_query": True,
                },
            )

        if not result.data:
            return ToolResponse(
                llm_response="No data found for your query.",
                rich_response={
                    "sql": result.sql,
                    "data": [],
                    "row_count": 0,
                    "is_data_query": True,
                },
            )

        data_text = self._format_sql_results(result.data)

        return ToolResponse(
            llm_response=data_text,
            rich_response={
                "sql": result.sql,
                "data": result.data,
                "row_count": result.row_count,
                "intent": result.intent.value if result.intent else None,
                "execution_time_ms": result.execution_time_ms,
                "is_data_query": True,
            },
        )

    def _build_hybrid_response(
        self,
        query: str,
        document_results: list[Chunk],
        sql_result: Optional[TextToSQLResult],
        is_data_query: bool,
        document_index: DocumentIndex,
        embedder: Embedder,
        llm: Optional[LLM],
        search_config: SearchConfig,
        hybrid_config: HybridSearchConfig,
        context: dict,
        history: Any,
        user_info: Optional[str],
        memories: Optional[list[str]],
    ) -> ToolResponse:
        """Build response combining document and SQL results."""
        logger.info(f"[HybridSearch] Building response - docs: {len(document_results)}, SQL: {sql_result and sql_result.data is not None}")

        # 检查 SQL 是否执行失败
        sql_error = sql_result.error_message if sql_result else None

        sql_chunks = []
        if sql_result and sql_result.data:
            sql_chunks = self._sql_results_to_chunks(sql_result)
            logger.info(f"[HybridSearch] Created {len(sql_chunks)} SQL chunks")

        all_chunks = list(document_results)
        all_chunks.extend(sql_chunks)

        if is_data_query and sql_error:
            error_text = f"⚠️ 数据库查询失败: {sql_error}"
            if document_results:
                response_text = f"{error_text}\n\n以下是与您查询相关的文档信息："
            else:
                return ToolResponse(
                    llm_response=error_text,
                    rich_response={
                        "sql": sql_result.sql if sql_result else None,
                        "error": sql_error,
                        "is_data_query": True,
                    },
                )

            rich_response = {
                "sql": sql_result.sql if sql_result else None,
                "error": sql_error,
                "is_data_query": True,
                "chunks": [],
                "sections": [],
            }
            # 继续处理文档结果
            selected_chunks = document_results if document_results else []

            if search_config.max_chunks_per_response and len(selected_chunks) > search_config.max_chunks_per_response:
                selected_chunks = selected_chunks[:search_config.max_chunks_per_response]

            rich_response["chunks"] = selected_chunks

            if selected_chunks:
                sections = []
                if search_config.enable_context_expansion:
                    pipeline = RetrievalPipeline(
                        document_index=document_index,
                        embedder=embedder,
                        config=search_config,
                    )
                    sections = pipeline.merge_adjacent_chunks(selected_chunks, max_gap=1)

                if sections:
                    if llm and search_config.enable_context_expansion:
                        selection = select_sections_for_expansion(
                            sections=sections,
                            query=query,
                            llm=llm,
                            max_sections=search_config.max_documents_to_select,
                            max_chunks_per_section=search_config.max_chunks_for_relevance,
                        )
                        section_index_map = {id(section): idx for idx, section in enumerate(sections)}
                        expanded_sections = expand_section_with_context(
                            sections=[sections[section_index_map[s]] for s in selection],
                            config=search_config,
                        )
                        sections = merge_overlapping_sections(expanded_sections)
                        sections = trim_sections_by_tokens(
                            sections=sections,
                            token_counter=llm.count_tokens if llm else None,
                            max_tokens=search_config.max_context_tokens,
                            max_chunks_per_section=search_config.context_expansion_chunks + 1,
                        )

                rich_response["sections"] = sections
                doc_sections = [s for s in sections if s.center_chunk.source_type != "sql_database"]
                if doc_sections:
                    doc_response, _ = self._build_response(
                        query=query,
                        chunks=[],
                        sections=doc_sections,
                        max_tokens=min(search_config.max_context_tokens, 2000),
                        max_content_chars=search_config.max_content_chars_per_chunk,
                    )
                    response_text += f"\n\n📄 相关文档:\n{doc_response}"

            return ToolResponse(
                llm_response=response_text,
                rich_response=rich_response,
            )

        if not all_chunks:
            return ToolResponse(
                llm_response="No relevant information found.",
                rich_response={"chunks": [], "is_data_query": is_data_query},
            )

        if is_data_query and sql_result and sql_result.data:
            data_text = self._format_sql_results(sql_result.data)
            intent_str = f"({sql_result.intent.value})" if sql_result.intent else ""
            
            if document_results:
                response_text = f"📊 数据库查询结果 {intent_str}:\n\n{data_text}\n\n相关文档已附加在下方。"
            else:
                return ToolResponse(
                    llm_response=data_text,
                    rich_response={
                        "sql": sql_result.sql,
                        "data": sql_result.data,
                        "row_count": sql_result.row_count,
                        "intent": sql_result.intent.value if sql_result.intent else None,
                        "execution_time_ms": sql_result.execution_time_ms,
                        "is_data_query": True,
                    },
                )

            rich_response = {
                "sql": sql_result.sql,
                "data": sql_result.data,
                "row_count": sql_result.row_count,
                "intent": sql_result.intent.value if sql_result.intent else None,
                "execution_time_ms": sql_result.execution_time_ms,
                "is_data_query": True,
                "chunks": [],
                "sections": [],
            }

            selected_chunks = []
            if llm and search_config.enable_document_selection:
                selected_chunks = self._select_documents(
                    query=query,
                    chunks=all_chunks,
                    llm=llm,
                    max_docs=search_config.max_documents_to_select,
                )
            else:
                selected_chunks = all_chunks

            if search_config.max_chunks_per_response and len(selected_chunks) > search_config.max_chunks_per_response:
                selected_chunks = selected_chunks[:search_config.max_chunks_per_response]

            rich_response["chunks"] = selected_chunks

            if selected_chunks:
                sections = []
                if search_config.enable_context_expansion:
                    pipeline = RetrievalPipeline(
                        document_index=document_index,
                        embedder=embedder,
                        config=search_config,
                    )
                    sections = pipeline.merge_adjacent_chunks(selected_chunks, max_gap=1)

                if sections:
                    if llm and search_config.enable_context_expansion:
                        selection = select_sections_for_expansion(
                            sections=sections,
                            query=query,
                            llm=llm,
                            max_sections=search_config.max_documents_to_select,
                            max_chunks_per_section=search_config.max_chunks_for_relevance,
                        )
                        section_index_map = {id(section): idx for idx, section in enumerate(sections)}

                        expansion_tasks = []
                        full_doc_chunks = search_config.max_full_document_chunks or 5
                        for section in selection.sections:
                            section_idx = section_index_map.get(id(section), -1)
                            force_full = section_idx in selection.full_document_section_ids
                            expansion_tasks.append((
                                expand_section_with_context,
                                (section, query, llm, document_index, full_doc_chunks, force_full),
                            ))

                        expanded_results = run_in_parallel(
                            expansion_tasks,
                            max_workers=min(len(expansion_tasks), search_config.section_expansion_workers),
                            allow_failures=True,
                        )
                        expanded_sections = [r for r in expanded_results if r is not None]
                        sections = merge_overlapping_sections(expanded_sections)
                        sections = trim_sections_by_tokens(
                            sections=sections,
                            token_counter=llm.count_tokens if llm else None,
                            max_tokens=search_config.max_context_tokens,
                            max_chunks_per_section=search_config.context_expansion_chunks + 1,
                        )

                    rich_response["sections"] = sections
                    doc_sections = [s for s in sections if s.center_chunk.source_type != "sql_database"]
                    if doc_sections:
                        doc_response, _ = self._build_response(
                            query=query,
                            chunks=[],
                            sections=doc_sections,
                            max_tokens=min(search_config.max_context_tokens, 2000),
                            max_content_chars=search_config.max_content_chars_per_chunk,
                        )
                        response_text += f"\n\n📄 相关文档:\n{doc_response}"

            return ToolResponse(
                llm_response=response_text,
                rich_response=rich_response,
            )

        document_results = document_results or []
        if not document_results:
            return ToolResponse(
                llm_response="No relevant information found.",
                rich_response={"chunks": [], "is_data_query": is_data_query},
            )

        selected_chunks = all_chunks
        if llm and search_config.enable_document_selection:
            selected_chunks = self._select_documents(
                query=query,
                chunks=all_chunks,
                llm=llm,
                max_docs=search_config.max_documents_to_select,
            )

        if search_config.max_chunks_per_response and len(selected_chunks) > search_config.max_chunks_per_response:
            selected_chunks = selected_chunks[:search_config.max_chunks_per_response]

        sections = []
        if search_config.enable_context_expansion:
            pipeline = RetrievalPipeline(
                document_index=document_index,
                embedder=embedder,
                config=search_config,
            )
            sections = pipeline.merge_adjacent_chunks(selected_chunks, max_gap=1)

        if llm and sections and search_config.enable_context_expansion:
            selection = select_sections_for_expansion(
                sections=sections,
                query=query,
                llm=llm,
                max_sections=search_config.max_documents_to_select,
                max_chunks_per_section=search_config.max_chunks_for_relevance,
            )
            section_index_map = {id(section): idx for idx, section in enumerate(sections)}

            expansion_tasks = []
            full_doc_chunks = search_config.max_full_document_chunks or 5
            for section in selection.sections:
                section_idx = section_index_map.get(id(section), -1)
                force_full = section_idx in selection.full_document_section_ids
                expansion_tasks.append((
                    expand_section_with_context,
                    (section, query, llm, document_index, full_doc_chunks, force_full),
                ))

            expanded_results = run_in_parallel(
                expansion_tasks,
                max_workers=min(len(expansion_tasks), search_config.section_expansion_workers),
                allow_failures=True,
            )
            expanded_sections = [r for r in expanded_results if r is not None]

            sections = merge_overlapping_sections(expanded_sections)
            sections = trim_sections_by_tokens(
                sections=sections,
                token_counter=llm.count_tokens if llm else None,
                max_tokens=search_config.max_context_tokens,
                max_chunks_per_section=search_config.context_expansion_chunks + 1,
            )

        response_text, citation_mapping = self._build_response(
            query=query,
            chunks=selected_chunks,
            sections=sections,
            max_tokens=search_config.max_context_tokens,
            max_content_chars=search_config.max_content_chars_per_chunk,
        )

        rich_response = {
            "chunks": selected_chunks,
            "sections": sections,
            "is_data_query": is_data_query,
        }

        if sql_result:
            rich_response["sql_result"] = {
                "sql": sql_result.sql,
                "data": sql_result.data,
                "row_count": sql_result.row_count,
                "intent": sql_result.intent.value if sql_result.intent else None,
            }

        return ToolResponse(
            llm_response=response_text,
            rich_response=rich_response,
            citation_mapping=citation_mapping,
        )

        rich_response = {
            "chunks": selected_chunks,
            "sections": sections,
            "is_data_query": is_data_query,
        }

        if sql_result:
            rich_response["sql_result"] = {
                "sql": sql_result.sql,
                "data": sql_result.data,
                "row_count": sql_result.row_count,
                "intent": sql_result.intent.value if sql_result.intent else None,
            }

        return ToolResponse(
            llm_response=response_text,
            rich_response=rich_response,
            citation_mapping=citation_mapping,
        )

    def _sql_results_to_chunks(self, result: TextToSQLResult) -> list[Chunk]:
        """Convert SQL results to Chunk objects."""
        chunks = []

        intent_desc = result.intent.value if result.intent else "数据查询"
        query_desc = f"用户查询意图: {intent_desc}"

        for i, row in enumerate(result.data):
            content_parts = []
            for key, value in row.items():
                if isinstance(value, dict):
                    value_str = json.dumps(value, ensure_ascii=False, default=str)
                elif isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                else:
                    value_str = str(value)
                content_parts.append(f"{key}: {value_str}")
            content = " | ".join(content_parts)

            chunk = Chunk(
                document_id=f"sql_result_{i}",
                chunk_id=i,
                content=f"[数据库查询结果 - 第{i+1}行]\n{query_desc}\n查询详情: {content}",
                title=f"查询结果 #{i+1}",
                source_type="sql_database",
                embedding=None,
                metadata={
                    "sql_result": True,
                    "original_query": result.sql,
                    "intent": result.intent.value if result.intent else None,
                    "row_data": row,
                    "row_index": i,
                },
            )
            chunk.score = 1.0
            chunks.append(chunk)

        return chunks

    def _format_sql_results(self, data: list[dict]) -> str:
        """Format SQL results as readable text."""
        if not data:
            return "未找到相关数据。"

        lines = []

        if len(data) == 1:
            row = data[0]
            parts = []
            for key, value in row.items():
                if isinstance(value, dict):
                    value_str = json.dumps(value, ensure_ascii=False, default=str)
                elif isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                else:
                    value_str = str(value)
                parts.append(f"{key}: {value_str}")
            return "📊 查询结果:\n  • " + "\n  • ".join(parts)

        headers = list(data[0].keys())
        lines.append("┌ " + " │ ".join(headers) + " ┐")
        
        for row in data:
            values = []
            for h in headers:
                val = row.get(h, "")
                if isinstance(val, dict):
                    val = json.dumps(val, ensure_ascii=False, default=str)[:30]
                elif isinstance(val, list):
                    val = ", ".join(str(v) for v in val)[:30]
                else:
                    val = str(val)[:30]
                values.append(val)
            lines.append("│ " + " │ ".join(values) + " │")

        lines.append("└" + "─" * (len(headers) * 15) + "┘")

        max_rows = 10
        if len(data) > max_rows:
            lines.append(f"... 共 {len(data)} 条结果")

        return "📊 查询结果:\n" + "\n".join(lines)

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
        """Build query specs for document search."""
        from agent_rag.tools.builtin.search.query_expansion import (
            keyword_query_expansion,
            semantic_query_rephrase,
        )

        if skip_query_expansion or not llm:
            return [QuerySpec(query=query, weight=1.0)]

        logger.debug("[SearchTool] Running query expansion in parallel")

        from agent_rag.tools.builtin.search.prompts import (
            KEYWORD_REPHRASE_USER_PROMPT,
            KEYWORD_REPHRASE_SYSTEM_PROMPT,
            SEMANTIC_QUERY_REPHRASE_USER_PROMPT,
            SEMANTIC_QUERY_REPHRASE_SYSTEM_PROMPT,
        )
        from agent_rag.llm.providers.litellm_provider import LLMMessage

        current_date = "2025-01-26"

        semantic_task = (
            semantic_query_rephrase,
            (query, llm, history, user_info, memories),
        )

        keyword_task = (
            keyword_query_expansion,
            (query, llm, config.max_expanded_queries, history, user_info, memories),
        )

        expansion_results = run_in_parallel(
            [semantic_task, keyword_task],
            max_workers=config.query_expansion_workers,
            allow_failures=True,
        )

        semantic_query = expansion_results[0]
        keyword_queries = expansion_results[1] or []

        specs: list[QuerySpec] = [
            QuerySpec(query=query, weight=config.original_query_weight),
        ]

        if semantic_query and semantic_query.strip() != query.strip():
            specs.append(QuerySpec(
                query=semantic_query,
                weight=config.llm_semantic_query_weight,
            ))

        for kw in keyword_queries:
            if kw.strip() != query.strip():
                specs.append(QuerySpec(
                    query=kw,
                    weight=config.llm_keyword_query_weight,
                ))

        return specs

    def _select_documents(
        self,
        query: str,
        chunks: list[Chunk],
        llm: LLM,
        max_docs: int = 10,
    ) -> list[Chunk]:
        """Select most relevant documents using LLM."""
        from agent_rag.tools.builtin.search.search_utils import (
            dedupe_chunks_by_title,
            score_chunks_by_query,
        )

        if not chunks:
            return []

        chunks = score_chunks_by_query(query, chunks)
        chunks = chunks[:max_docs * 2]

        from agent_rag.ingestion.text_normalize import clean_markdown

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            source = clean_up_source(chunk.source_type)
            cleaned_content = clean_markdown(chunk.content, max_lines=5)
            summary = f"[{i}] {chunk.title or 'Untitled'}\nSource: {source}\nContent: {cleaned_content[:200]}..."
            chunk_summaries.append(summary)

        chunk_selection_prompt = f"""
Select the most relevant document chunks for answering: "{query}"

Summaries:
{chr(10).join(chunk_summaries)}

Respond with JSON:
{{"indices": [0, 1, ...]}}
"""

        from agent_rag.llm.providers.litellm_provider import LLMMessage

        messages = [
            LLMMessage(role="system", content="You are a document selection assistant."),
            LLMMessage(role="user", content=chunk_selection_prompt),
        ]

        try:
            response = llm.chat(messages, max_tokens=200)

            import json

            content = response.content.strip()
            json_match = content
            if content.startswith("```json"):
                json_match = content[7:]
            if content.startswith("```"):
                json_match = content[3:]
            if content.endswith("```"):
                json_match = json_match[:-3]

            result = json.loads(json_match)
            indices = result.get("indices", [])

            selected = [chunks[i] for i in indices if i < len(chunks)]
            return dedupe_chunks_by_title(selected)

        except Exception as e:
            logger.warning(f"Document selection failed: {e}")
            return chunks[:max_docs]

    def _build_response(
        self,
        query: str,
        chunks: list[Chunk],
        sections: list[Section],
        max_tokens: int,
        max_content_chars: int,
    ) -> tuple[str, dict]:
        """Build response text and citation mapping."""
        from agent_rag.citation.processor import CitationProcessor
        from agent_rag.ingestion.text_normalize import clean_markdown

        if sections:
            section_texts = []
            for i, section in enumerate(sections):
                source = clean_up_source(section.center_chunk.source_type)
                cleaned_content = clean_markdown(section.combined_content, max_lines=50)
                section_texts.append(f"[Section {i+1}] ({source})\n{cleaned_content}")
            content = "\n\n".join(section_texts)
        else:
            chunk_texts = []
            for i, chunk in enumerate(chunks):
                source = clean_up_source(chunk.source_type)
                cleaned_content = clean_markdown(chunk.content, max_lines=30)
                title = clean_markdown(chunk.title or '', max_lines=1) if chunk.title else ''
                chunk_texts.append(f"[{i+1}] ({source})\n{title}\n{cleaned_content}")
            content = "\n\n".join(chunk_texts)

        processor = CitationProcessor()
        content = processor.process(content, max_content_chars, max_tokens)

        return content, {}
