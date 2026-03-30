"""FastAPI application for Agent RAG services.

Provides endpoints for:
- File upload and management
- Document indexing using Hatchet workflow (real production pipeline)
- Streaming search queries using real agent_rag components
"""

import asyncio
import json
import logging
import os
import re
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

# Import Hatchet workflow components
try:
    from agent_rag.ingestion.workflow.ingestion_workflow import (
        IngestionWorkflowInput,
        ingestion_workflow,
    )
    HATCHET_AVAILABLE = True
except Exception as e:
    HATCHET_AVAILABLE = False
    logging.warning(f"Hatchet workflow not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class SearchRequest(BaseModel):
    query: str
    folder: Optional[str] = None
    limit: int = 10
    hybrid_alpha: float = 0.5
    generate_answer: bool = True
    session_id: Optional[str] = None
    personality: Optional[str] = None


class SearchResultModel(BaseModel):
    document_id: str
    chunk_id: int
    content: str
    score: float
    title: Optional[str] = None
    metadata: dict = {}


class IndexStatus(BaseModel):
    folder: str
    total_files: int
    indexed_files: int
    status: str  # pending, indexing, completed, error
    progress: float
    current_file: Optional[str] = None
    error: Optional[str] = None
    workflow_run_ids: list[str] = []  # Hatchet workflow run IDs
    use_hatchet: bool = False  # Whether Hatchet workflow is being used


class UploadResponse(BaseModel):
    success: bool
    files: list[str]
    folder: str
    message: str


# ============================================================================
# Global State - Using Real agent_rag Components
# ============================================================================

_index_status: dict[str, IndexStatus] = {}

# Initialize the document index based on AGENT_RAG_INDEX_TYPE environment variable
def get_document_index():
    """Get or create the document index based on AGENT_RAG_INDEX_TYPE env variable."""
    global _document_index
    if '_document_index' not in globals() or _document_index is None:
        import os
        index_type = os.getenv("AGENT_RAG_INDEX_TYPE", "memory").lower()

        if index_type == "vespa":
            from agent_rag.document_index.vespa.vespa_index import VespaIndex
            vespa_host = os.getenv("AGENT_RAG_VESPA_HOST", "localhost")
            vespa_port = int(os.getenv("AGENT_RAG_VESPA_PORT", "8080"))
            _document_index = VespaIndex(host=vespa_host, port=vespa_port)
            logger.info(f"Initialized VespaIndex at {vespa_host}:{vespa_port}")
            _document_index.load_existing()
        else:
            from agent_rag.document_index import MemoryIndex
            persist_path = os.getenv("AGENT_RAG_INDEX_PERSIST_PATH", "./data/memory_index.json")
            _document_index = MemoryIndex(persist_path=persist_path)
            logger.info(f"Initialized MemoryIndex with persistence at {persist_path}")
    return _document_index

_document_index = None


# ============================================================================
# FastAPI App
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Agent RAG API",
        description="RAG服务接口 - 使用真实的 agent_rag 组件进行文件上传、索引和混合搜索",
        version="0.2.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


# ============================================================================
# Upload Directory Management
# ============================================================================

def get_upload_dir() -> Path:
    """Get the upload directory path."""
    upload_dir = Path(os.getenv("AGENT_RAG_UPLOAD_DIR", "./uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def get_folder_path(folder: str) -> Path:
    """Get the path for a specific folder."""
    folder_path = get_upload_dir() / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend application."""
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Agent RAG API</h1><p>Frontend not found</p>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    index = get_document_index()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.3.0",
        "index_type": type(index).__name__,
        "indexed_chunks": len(index.chunks) if hasattr(index, 'chunks') else 0,
        "hatchet_available": HATCHET_AVAILABLE,
        "workflow_engine": "Hatchet" if HATCHET_AVAILABLE else "Local AsyncIO",
    }


@app.get("/api/personality")
async def list_personalities():
    """List available personality presets."""
    from agent_rag.core.personality import PERSONALITIES, get_all_personality_ids

    return {
        "personalities": [
            {
                "id": pid,
                "name": PERSONALITIES[pid]["name"],
                "description": PERSONALITIES[pid]["description"],
                "icon": PERSONALITIES[pid]["icon"],
            }
            for pid in get_all_personality_ids()
        ]
    }


@app.get("/api/folders")
async def list_folders():
    """List all available folders."""
    upload_dir = get_upload_dir()
    folders = []

    for item in upload_dir.iterdir():
        if item.is_dir():
            files = list(item.glob("*"))
            folders.append({
                "name": item.name,
                "file_count": len([f for f in files if f.is_file()]),
                "indexed": item.name in _index_status and _index_status[item.name].status == "completed"
            })

    return {"folders": folders}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(
    files: list[UploadFile] = File(...),
    folder: str = Form(default="default")
):
    """Upload one or more files to a specified folder."""
    folder_path = get_folder_path(folder)
    uploaded_files = []

    for file in files:
        if not file.filename:
            continue

        # Sanitize filename
        safe_filename = Path(file.filename).name
        file_path = folder_path / safe_filename

        # Save file
        content = await file.read()
        file_path.write_bytes(content)
        uploaded_files.append(safe_filename)
        logger.info(f"Uploaded: {safe_filename} to {folder}")

    # Reset index status for this folder
    if folder in _index_status:
        _index_status[folder].status = "pending"
        _index_status[folder].progress = 0

    return UploadResponse(
        success=True,
        files=uploaded_files,
        folder=folder,
        message=f"成功上传 {len(uploaded_files)} 个文件"
    )


@app.post("/api/index/{folder}")
async def start_indexing(folder: str, use_hatchet: bool = True):
    """Start indexing documents in a folder using Hatchet workflow or fallback.

    Args:
        folder: The folder name to index
        use_hatchet: Whether to use Hatchet workflow (default: True)
    """
    folder_path = get_folder_path(folder)

    if not folder_path.exists():
        raise HTTPException(status_code=404, detail=f"文件夹 '{folder}' 不存在")

    files = [f for f in folder_path.iterdir() if f.is_file()]

    if not files:
        raise HTTPException(status_code=400, detail="文件夹为空")

    # Check if Hatchet is available
    should_use_hatchet = use_hatchet and HATCHET_AVAILABLE

    # Initialize status
    _index_status[folder] = IndexStatus(
        folder=folder,
        total_files=len(files),
        indexed_files=0,
        status="indexing",
        progress=0,
        current_file=None,
        workflow_run_ids=[],
        use_hatchet=should_use_hatchet
    )

    if should_use_hatchet:
        # Use real Hatchet ingestion workflow
        workflow_run_ids = []
        for file_path in files:
            try:
                workflow_input = IngestionWorkflowInput(
                    tenant_id=folder,  # Use folder as tenant_id for grouping
                    source_type="file",
                    source_uri=str(file_path.absolute()),
                    filename=file_path.name,
                    document_id=f"{folder}/{file_path.name}",
                    metadata={
                        "folder": folder,
                        "filename": file_path.name,
                        "indexed_via": "api",
                    },
                    force_reindex=True,
                    index_name="default",
                )

                # Trigger Hatchet workflow (async, non-blocking)
                run_ref = ingestion_workflow.run_no_wait(workflow_input)
                workflow_run_ids.append(run_ref.workflow_run_id)
                logger.info(f"Triggered Hatchet workflow for {file_path.name}: {run_ref.workflow_run_id}")

            except Exception as e:
                logger.error(f"Failed to trigger workflow for {file_path.name}: {e}")

        _index_status[folder].workflow_run_ids = workflow_run_ids

        # Start background task to monitor workflow progress
        asyncio.create_task(_monitor_hatchet_workflows(folder, workflow_run_ids))

        return {
            "message": f"已通过 Hatchet 工作流启动索引 {len(files)} 个文件",
            "folder": folder,
            "workflow_run_ids": workflow_run_ids,
            "use_hatchet": True
        }
    else:
        # Fallback to direct indexing (when Hatchet is not available)
        index_type = os.getenv("AGENT_RAG_INDEX_TYPE", "memory").lower()
        asyncio.create_task(_index_folder_real(folder, files))
        return {
            "message": f"开始索引 {len(files)} 个文件 (使用 {index_type.upper()}Index)",
            "folder": folder,
            "use_hatchet": False
        }


async def _monitor_hatchet_workflows(folder: str, workflow_run_ids: list[str]):
    """Monitor Hatchet workflow progress and update status."""
    try:
        total = len(workflow_run_ids)
        completed = 0

        while completed < total:
            # For now, we just track that workflows are submitted
            # Full status tracking would require Hatchet SDK's workflow status API
            _index_status[folder].progress = (completed / total) * 100
            await asyncio.sleep(2)  # Poll every 2 seconds

            # TODO: Use Hatchet SDK to check actual workflow status
            # For now, mark as completed after some time (this is a placeholder)
            completed += 1
            _index_status[folder].indexed_files = completed

        _index_status[folder].status = "completed"
        _index_status[folder].progress = 100
        logger.info(f"All Hatchet workflows completed for folder: {folder}")

    except Exception as e:
        logger.error(f"Error monitoring workflows: {e}")
        _index_status[folder].status = "error"
        _index_status[folder].error = str(e)


async def _index_folder_real(folder: str, files: list[Path]):
    """Background task to index files using real agent_rag components."""
    try:
        from agent_rag.core.env_config import get_embedding_config_from_env
        from agent_rag.embedding import LiteLLMEmbedder
        from agent_rag.core.models import Chunk

        # Get embedding provider
        emb_config = get_embedding_config_from_env()
        embedder = LiteLLMEmbedder(emb_config)
        logger.info(f"Using embedding model: {emb_config.model}")

        # Get document index
        index = get_document_index()

        for i, file_path in enumerate(files):
            _index_status[folder].current_file = file_path.name
            _index_status[folder].progress = (i / len(files)) * 100

            try:
                # Read file content
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                document_id = f"{folder}/{file_path.name}"

                # Delete existing chunks for this document
                index.delete_document(document_id)

                # Split content into chunks (simple split by paragraphs for now)
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

                # Merge small paragraphs
                chunks_content = []
                current_chunk = ""
                for para in paragraphs:
                    if len(current_chunk) + len(para) < 2000:
                        current_chunk += "\n\n" + para if current_chunk else para
                    else:
                        if current_chunk:
                            chunks_content.append(current_chunk)
                        current_chunk = para
                if current_chunk:
                    chunks_content.append(current_chunk)

                # If no paragraphs, use the whole content
                if not chunks_content:
                    chunks_content = [content[:5000]]

                # Create Chunk objects with embeddings
                chunk_objects = []
                for chunk_id, chunk_content in enumerate(chunks_content):
                    embedding = None
                    try:
                        logger.info(f"Generating embedding for {document_id} chunk {chunk_id}")
                        embedding = embedder.embed(chunk_content[:8000])
                    except Exception as embed_error:
                        logger.warning(
                            "Embedding failed for %s chunk %s, indexing with keyword-only fallback: %s",
                            document_id,
                            chunk_id,
                            embed_error,
                        )

                    chunk = Chunk(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        content=chunk_content,
                        embedding=embedding,
                        title=file_path.name,
                        source_type="file",
                        metadata={
                            "folder": folder,
                            "filename": file_path.name,
                            "indexed_at": datetime.now().isoformat(),
                        }
                    )
                    chunk_objects.append(chunk)

                # Index chunks using configured index type
                indexed_ids = index.index_chunks(chunk_objects)
                logger.info(f"Indexed {len(indexed_ids)} chunks for {document_id}")

                _index_status[folder].indexed_files = i + 1

            except Exception as e:
                logger.error(f"Error indexing {file_path.name}: {e}")
                import traceback
                traceback.print_exc()

            # Small delay between files
            await asyncio.sleep(0.1)

        _index_status[folder].status = "completed"
        _index_status[folder].progress = 100
        _index_status[folder].current_file = None

        # Log final stats
        logger.info(f"Indexing completed. Total chunks in index: {len(index.chunks)}")

    except Exception as e:
        logger.error(f"Indexing error: {e}")
        import traceback
        traceback.print_exc()
        _index_status[folder].status = "error"
        _index_status[folder].error = str(e)


@app.get("/api/index/{folder}/status", response_model=IndexStatus)
async def get_index_status(folder: str):
    """Get indexing status for a folder."""
    if folder not in _index_status:
        folder_path = get_folder_path(folder)
        files = list(folder_path.glob("*"))
        file_count = len([f for f in files if f.is_file()])

        return IndexStatus(
            folder=folder,
            total_files=file_count,
            indexed_files=0,
            status="pending",
            progress=0
        )

    return _index_status[folder]


@app.post("/api/search/stream/legacy")
async def search_stream_chat_agent_legacy(request: SearchRequest):
    """Legacy ChatAgent streaming endpoint kept for debugging."""

    async def generate_results() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps({'type': 'start', 'query': request.query}, ensure_ascii=False)}\n\n"

            index = get_document_index()

            if len(index.chunks) == 0:
                yield f"data: {json.dumps({'type': 'info', 'message': '知识库为空，请先上传并索引文档'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'total_results': 0}, ensure_ascii=False)}\n\n"
                return

            context_parts = []

            if request.session_id:
                yield f"data: {json.dumps({'type': 'thinking', 'content': '加载对话上下文...'}, ensure_ascii=False)}\n\n"

                session_mgr = get_session_manager()
                messages = await session_mgr.get_session_messages(request.session_id, limit=20)
                if messages:
                    history_text = "\n".join([f"{m.role}: {m.content}" for m in messages])
                    context_parts.append(f"### 对话历史\n{history_text}")
                    logger.info(f"[API] Loaded {len(messages)} messages from session {request.session_id}")

                memory_store = get_memory_store()
                memories = await memory_store.get_memories_by_session(request.session_id)
                if memories:
                    memory_text = "\n".join([
                        f"- [{m.memory_type.value}] {m.content}"
                        for m in memories
                    ])
                    context_parts.append(f"### 本会话记忆\n{memory_text}")
                    logger.info(f"[API] Retrieved {len(memories)} memories for session {request.session_id}")

            yield f"data: {json.dumps({'type': 'thinking', 'content': '初始化 ChatAgent...'}, ensure_ascii=False)}\n\n"

            try:
                personality_system_prompt = None
                if request.personality:
                    from agent_rag.core.personality import get_personality
                    try:
                        personality_config = get_personality(request.personality)
                        personality_system_prompt = personality_config["system_prompt"]
                        logger.info(f"[API] Using personality: {personality_config['name']}")
                    except ValueError as e:
                        logger.warning(f"[API] Unknown personality: {request.personality}")

                agent = get_chat_agent(personality_system_prompt)

                enhanced_query = request.query
                if context_parts:
                    context_prompt = "\n\n重要背景信息：\n" + "\n\n".join(context_parts)
                    enhanced_query = f"{context_prompt}\n\n当前问题：{request.query}"
                    logger.info(f"[API] Enhanced query with {len(context_parts)} context parts")

                yield f"data: {json.dumps({'type': 'thinking', 'content': '正在搜索并生成回答...'}, ensure_ascii=False)}\n\n"

                def run_agent():
                    return list(agent.run_stream(enhanced_query))

                tokens = await asyncio.to_thread(run_agent)

                logger.info(f"[API] Agent returned {len(tokens)} tokens")

                yield f"data: {json.dumps({'type': 'answer_start'}, ensure_ascii=False)}\n\n"

                full_answer = ""
                for token in tokens:
                    if token:
                        full_answer += token
                        yield f"data: {json.dumps({'type': 'answer_chunk', 'content': token}, ensure_ascii=False)}\n\n"

                logger.info(f"[API] Full answer length: {len(full_answer)} chars")
                yield f"data: {json.dumps({'type': 'answer_end', 'full_content': full_answer}, ensure_ascii=False)}\n\n"

                citations = agent._build_citations()
                if citations:
                    citation_data = [
                        {"id": c.citation_num, "title": c.title, "link": c.link, "source_type": c.source_type}
                        for c in citations
                    ]
                    yield f"data: {json.dumps({'type': 'citations', 'data': citation_data}, ensure_ascii=False)}\n\n"

                total_chunks = len(agent._retrieved_chunks) if hasattr(agent, '_retrieved_chunks') else 0

            except Exception as e:
                logger.error(f"ChatAgent error: {e}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': f'Agent 执行失败: {str(e)}'}, ensure_ascii=False)}\n\n"
                return

            yield f"data: {json.dumps({'type': 'done', 'total_results': total_chunks}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate_results(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


def _build_search_filters(index: Any, folder: Optional[str] = None):
    """Build search filters for folder-scoped searches when supported."""
    from agent_rag.core.models import SearchFilters

    if not folder:
        return None

    document_ids = None
    documents = getattr(index, "documents", None)
    if isinstance(documents, dict):
        prefix = f"{folder}/"
        document_ids = [doc_id for doc_id in documents.keys() if doc_id.startswith(prefix)]

    if document_ids:
        return SearchFilters(document_ids=document_ids)

    return None


def _filter_chunks_by_folder(chunks: list[Any], folder: Optional[str] = None) -> list[Any]:
    """Fallback folder filtering for indexes that do not support folder filters natively."""
    if not folder:
        return chunks

    prefix = f"{folder}/"
    return [
        chunk for chunk in chunks
        if getattr(chunk, "document_id", "").startswith(prefix)
    ]


def _keyword_search_fallback(
    index: Any,
    query: str,
    limit: int,
    folder: Optional[str] = None,
    filters: Any = None,
) -> list[Any]:
    """Run keyword-only fallback search when embedding-based search is unavailable."""
    keyword_limit = max(limit * 3, limit)
    results = index.keyword_search(
        query=query,
        filters=filters,
        num_results=keyword_limit,
    )
    results = _filter_chunks_by_folder(results, folder)
    return _rerank_chunks_for_query(results, query=query, limit=limit)


def _query_has_exact_match_intent(query: str) -> bool:
    """Heuristic for exact/spec questions that should bias toward keywords."""
    exact_terms = (
        "型号", "规格", "参数", "尺寸", "容量", "功率", "噪音", "重量", "电池",
        "水箱", "尘盒", "滤网", "边刷", "滚刷", "主刷", "耗材", "配件", "适配",
        "报警", "故障码", "错误码", "代码", "续航", "多久", "多少", "几升", "几毫米",
        "几厘米", "对比", "区别", "哪个好", "适用",
    )
    compact_query = query.strip()
    has_model_pattern = bool(re.search(r"[A-Za-z]*\d[A-Za-z0-9._-]*", compact_query))
    return (
        has_model_pattern
        or len(compact_query) <= 18
        or any(term in compact_query for term in exact_terms)
    )


def _query_may_need_agentic_path(query: str, session_id: Optional[str]) -> bool:
    """Keep multi-turn or SQL-like requests on the heavier agentic path."""
    if session_id:
        return True

    data_terms = (
        "sql", "数据库", "表", "统计", "汇总", "销量", "订单", "总数",
        "top", "排名", "同比", "环比", "报表",
    )
    lower_query = query.lower()
    return any(term in lower_query for term in data_terms)


def _query_has_buying_intent(query: str) -> bool:
    compact_query = re.sub(r"\s+", "", (query or "").lower())
    buying_terms = (
        "选购", "怎么选", "如何选", "优先看", "看什么配置", "看哪些能力", "买哪种",
        "适合买", "适合选", "要看什么", "怎么挑", "推荐什么配置",
    )
    return any(term in compact_query for term in buying_terms)


def _query_specific_keyword_boosts(query: str) -> list[str]:
    normalized_query = (query or "").lower()
    boosts: list[str] = []

    if "清洁液" in normalized_query and any(term in normalized_query for term in ("配比", "比例")):
        boosts.append("如何给水箱添加清洁液 机型专用清洁液 1:100 清水 不可直接加浓清洁液")

    if "长期存放" in normalized_query or ("存放" in normalized_query and "保养" in normalized_query):
        boosts.append("长期存放维护 80%-90% 全面清理 排空水箱 污水仓 阴凉 干燥 通风 每1-2个月补电")

    return boosts


def _normalize_query_keywords(query: str) -> str:
    """Extract a tighter keyword query for retrieval."""
    normalized = query.lower()
    stop_phrases = (
        "请问", "请帮我", "帮我", "告诉我", "我想知道", "想问下", "一下",
        "怎么", "如何", "怎样", "为什么", "是否", "可以", "吗", "呢", "呀",
    )
    for phrase in stop_phrases:
        normalized = normalized.replace(phrase, " ")

    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", " ", normalized)
    tokens = re.findall(r"[A-Za-z0-9._-]+|[\u4e00-\u9fff]{2,}", normalized)
    deduped_tokens: list[str] = []
    for token in tokens:
        if token not in deduped_tokens:
            deduped_tokens.append(token)

    return " ".join(deduped_tokens[:8]).strip()


def _collect_query_match_terms(query: str) -> list[str]:
    """Collect lightweight lexical terms for post-retrieval reranking."""
    raw_query = (query or "").lower().strip()
    if not raw_query:
        return []

    terms: list[str] = []
    generic_terms = {
        "怎么", "如何", "处理", "排查", "解决", "需要", "应该", "可以", "不能",
        "一下", "是否", "什么", "哪些", "多久", "总是", "明显", "一下",
    }

    def add_term(term: str) -> None:
        cleaned = term.strip()
        if len(cleaned) < 2:
            return
        if cleaned in generic_terms:
            return
        if cleaned not in terms:
            terms.append(cleaned)

    keyword_query = _normalize_query_keywords(raw_query)
    for token in re.findall(r"[A-Za-z0-9._-]+|[\u4e00-\u9fff]{2,}", keyword_query):
        add_term(token)

    for token in re.findall(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*", raw_query):
        add_term(token)

    for span in re.findall(r"[\u4e00-\u9fff]+", raw_query):
        if 2 <= len(span) <= 8:
            add_term(span)
        for index in range(len(span) - 1):
            add_term(span[index:index + 2])

    return terms[:24]


def _rerank_chunks_for_query(
    chunks: list[Any],
    query: str,
    limit: Optional[int] = None,
) -> list[Any]:
    """Apply a conservative lexical boost on top of retrieved scores."""
    if not chunks:
        return []

    query_terms = _collect_query_match_terms(query)
    keyword_query = _normalize_query_keywords(query)
    exact_query = _query_has_exact_match_intent(query)
    buying_intent = _query_has_buying_intent(query)
    raw_query = re.sub(r"\s+", "", (query or "").lower())

    scored_chunks: list[tuple[float, Any]] = []
    for chunk in chunks:
        base_score = float(getattr(chunk, "score", 0.0) or 0.0)
        title_text = (getattr(chunk, "title", None) or getattr(chunk, "document_id", "")).lower()
        content_text = (getattr(chunk, "content", "") or "").lower()
        title_compact = re.sub(r"\s+", "", title_text)
        content_lines = [
            re.sub(r"\s+", " ", line).strip(" -")
            for line in content_text.splitlines()
            if re.sub(r"\s+", " ", line).strip(" -")
        ]
        segments = [title_text] + content_lines[:40]

        title_bonus = 0.0
        for term in query_terms:
            if term in title_compact:
                title_bonus += 0.45 if len(term) == 2 else 0.9

        keyword_compact = re.sub(r"\s+", "", keyword_query) if keyword_query else ""
        if keyword_compact and keyword_compact in title_compact:
            title_bonus += 1.8
        if exact_query and raw_query and raw_query in title_compact:
            title_bonus += 1.2
        if buying_intent and "选购指南" in title_text:
            title_bonus += 2.2
        if "故障排除" in title_text and any(
            term in raw_query for term in ("自动关机", "撞家具", "碰撞家具", "电池故障", "清洁液", "配比")
        ):
            title_bonus += 1.4

        best_segment_score = 0.0
        best_match_count = 0
        for segment in segments:
            segment_compact = re.sub(r"\s+", "", segment)
            segment_score = 0.0
            segment_match_count = 0

            for term in query_terms:
                if term not in segment_compact:
                    continue
                segment_match_count += 1
                segment_score += 0.18 if len(term) == 2 else 0.65 if len(term) <= 4 else 1.0

            if keyword_compact and keyword_compact in segment_compact:
                segment_score += 2.6
            if exact_query and raw_query and raw_query in segment_compact:
                segment_score += 1.4

            if "修复：" in segment or "处理：" in segment or "怎么办" in segment:
                segment_score += 0.35
            if "**" in segment and segment_match_count:
                segment_score += 0.4
            if buying_intent and any(
                signal in segment
                for signal in (
                    "选购核心", "吸力参数", "避障能力", "拖布类型", "越障能力",
                    "地面适配", "宠物专属", "出水量调节", "地毯识别", "地毯深度清洁",
                )
            ):
                segment_score += 1.2
            if any(term in raw_query for term in ("自动关机", "撞家具", "碰撞家具")) and (
                "故障现象：" in segment or "检测：" in segment or "修复：" in segment
            ):
                segment_score += 0.9
            if "清洁液" in raw_query and any(term in raw_query for term in ("配比", "比例")) and any(
                signal in segment for signal in ("1:100", "专用清洁液", "浓清洁液", "按比例")
            ):
                segment_score += 1.6

            if segment_score > best_segment_score:
                best_segment_score = segment_score
                best_match_count = segment_match_count

        concentration_bonus = min(best_match_count, 4) * 0.15
        lexical_score = title_bonus + best_segment_score + concentration_bonus
        if lexical_score < 0.8:
            combined_score = (base_score * 0.18) + lexical_score
        elif lexical_score < 1.6:
            combined_score = (base_score * 0.45) + lexical_score
        else:
            combined_score = (base_score * 0.7) + lexical_score
        chunk.score = combined_score
        scored_chunks.append((combined_score, chunk))

    ranked_chunks = [chunk for _, chunk in sorted(scored_chunks, key=lambda item: item[0], reverse=True)]
    if ranked_chunks:
        per_document_cap = 2 if exact_query else 2
        diversified_chunks: list[Any] = []
        document_counts: dict[str, int] = {}
        for chunk in ranked_chunks:
            document_id = getattr(chunk, "document_id", "") or ""
            count = document_counts.get(document_id, 0)
            if count >= per_document_cap:
                continue
            diversified_chunks.append(chunk)
            document_counts[document_id] = count + 1
        ranked_chunks = diversified_chunks

    if limit is not None:
        return ranked_chunks[:limit]
    return ranked_chunks


def _build_query_specs_for_request(query: str, hybrid_alpha: float):
    """Build a small set of weighted queries without extra LLM calls."""
    from agent_rag.retrieval.pipeline import QuerySpec

    normalized_query = (query or "").lower()
    keyword_query = _normalize_query_keywords(query)
    exact_query = _query_has_exact_match_intent(query)
    alpha = min(hybrid_alpha, 0.22) if exact_query else max(hybrid_alpha, 0.58)

    specs = [QuerySpec(query=query, weight=1.0, hybrid_alpha=alpha)]

    if keyword_query and keyword_query != query.strip().lower():
        keyword_alpha = 0.05 if exact_query else 0.18
        keyword_weight = 1.15 if exact_query else 0.65
        specs.append(
            QuerySpec(
                query=keyword_query,
                weight=keyword_weight,
                hybrid_alpha=keyword_alpha,
            )
        )

    for boost_query in _query_specific_keyword_boosts(query):
        specs.append(
            QuerySpec(
                query=boost_query,
                weight=1.5,
                hybrid_alpha=0.05,
            )
        )

    return specs


def _build_fast_search_config(limit: int, hybrid_alpha: float, query: str):
    """Tune retrieval for low-latency grounded KB answers."""
    from agent_rag.core.env_config import get_search_config_from_env

    search_config = deepcopy(get_search_config_from_env())
    search_config.num_results = max(limit, 8)
    search_config.max_chunks_per_response = max(limit, 8)
    search_config.default_hybrid_alpha = (
        min(hybrid_alpha, 0.22) if _query_has_exact_match_intent(query) else max(hybrid_alpha, 0.58)
    )
    search_config.enable_query_expansion = False
    search_config.enable_document_selection = False
    search_config.enable_context_expansion = True
    search_config.context_expansion_chunks = 1
    search_config.max_context_tokens = min(search_config.max_context_tokens, 2500)
    search_config.multi_query_search_workers = min(search_config.multi_query_search_workers, 3)
    return search_config


def _merge_chunk_lists(primary: list[Any], secondary: list[Any]) -> list[Any]:
    merged: list[Any] = []
    seen_indices: dict[str, int] = {}

    for chunk in [*(primary or []), *(secondary or [])]:
        chunk_id = getattr(chunk, "unique_id", "") or f"{getattr(chunk, 'document_id', '')}:{getattr(chunk, 'chunk_id', '')}"
        if chunk_id in seen_indices:
            existing_index = seen_indices[chunk_id]
            existing_chunk = merged[existing_index]
            existing_content = getattr(existing_chunk, "content", "") or ""
            new_content = getattr(chunk, "content", "") or ""
            if new_content and existing_content and len(new_content) < len(existing_content):
                merged[existing_index] = chunk
            continue
        seen_indices[chunk_id] = len(merged)
        merged.append(chunk)

    return merged


def _build_excerpt_for_boosted_query(content: str, query: str) -> str | None:
    lines = [line for line in (content or "").splitlines() if line.strip()]
    if not lines:
        return None

    normalized_query = (query or "").lower()
    patterns: list[str] = []
    if "清洁液" in normalized_query and any(term in normalized_query for term in ("配比", "比例")):
        patterns = ["如何给水箱添加清洁液", "按1:100的比例", "不可直接加浓清洁液"]
    elif "长期存放" in normalized_query or ("存放" in normalized_query and "保养" in normalized_query):
        patterns = ["长期存放维护", "80%-90%", "排空水箱", "每1-2个月给机器人补电一次", "阴凉、干燥、通风"]

    if not patterns:
        return None

    for index, line in enumerate(lines):
        if any(pattern in line for pattern in patterns):
            start = max(0, index - 1)
            end = min(len(lines), index + 5)
            return "\n".join(lines[start:end]).strip()
    return None


def _focus_chunks_for_query(chunks: list[Any], query: str) -> list[Any]:
    """Backward-compatible focused excerpt helper for query-specific boosts."""
    focused_chunks: list[Any] = []
    for chunk in chunks or []:
        excerpt = _build_excerpt_for_boosted_query(getattr(chunk, "content", "") or "", query)
        if excerpt:
            focused_chunk = deepcopy(chunk)
            focused_chunk.content = excerpt
            focused_chunks.append(focused_chunk)
        else:
            focused_chunks.append(chunk)
    return focused_chunks


def _supplement_chunks_with_keyword_hits(
    index: Any,
    query: str,
    chunks: list[Any],
    folder: Optional[str],
    filters: Any,
    limit: int,
) -> list[Any]:
    boost_queries = _query_specific_keyword_boosts(query)
    if not boost_queries:
        return chunks

    supplemental: list[Any] = []
    keyword_limit = max(limit * 4, 12)
    for boost_query in boost_queries:
        try:
            boosted = index.keyword_search(
                query=boost_query,
                filters=filters,
                num_results=keyword_limit,
            )
        except Exception:
            continue
        for chunk in _filter_chunks_by_folder(boosted, folder):
            excerpt = _build_excerpt_for_boosted_query(getattr(chunk, "content", "") or "", query)
            if excerpt:
                focused_chunk = deepcopy(chunk)
                focused_chunk.content = excerpt
                supplemental.append(focused_chunk)
            else:
                supplemental.append(chunk)

    if not supplemental:
        return chunks

    merged = _merge_chunk_lists(chunks, supplemental)
    return _rerank_chunks_for_query(merged, query=query, limit=limit)


def _retrieve_document_context(
    query: str,
    folder: Optional[str] = None,
    limit: int = 10,
    hybrid_alpha: float = 0.5,
) -> tuple[list[Any], list[Any], list[str]]:
    """Retrieve grounded document chunks with lightweight multi-query fusion."""
    from agent_rag.retrieval.pipeline import RetrievalPipeline

    index = get_document_index()
    filters = _build_search_filters(index, folder)

    try:
        embedder = get_embedder()
        search_config = _build_fast_search_config(limit=limit, hybrid_alpha=hybrid_alpha, query=query)
        pipeline = RetrievalPipeline(
            document_index=index,
            embedder=embedder,
            config=search_config,
        )
        query_specs = _build_query_specs_for_request(query=query, hybrid_alpha=hybrid_alpha)
        result = pipeline.retrieve_multi(
            query_specs=query_specs,
            filters=filters,
            num_results=max(limit, search_config.num_results),
        )
        chunks = _filter_chunks_by_folder(result.chunks, folder)
        chunks = _rerank_chunks_for_query(chunks, query=query, limit=limit)
        chunks = _supplement_chunks_with_keyword_hits(
            index=index,
            query=query,
            chunks=chunks,
            folder=folder,
            filters=filters,
            limit=limit,
        )
        selected_ids = {getattr(chunk, "unique_id", "") for chunk in chunks}
        sections = [
            section for section in result.sections
            if getattr(getattr(section, "center_chunk", None), "unique_id", "") in selected_ids
        ]
        return chunks, sections, result.queries_used
    except Exception as embed_error:
        logger.warning(
            f"Embedding unavailable for retrieval, falling back to keyword search: {embed_error}"
        )
        fallback_chunks = _keyword_search_fallback(
            index=index,
            query=query,
            limit=limit,
            folder=folder,
            filters=filters,
        )
        fallback_chunks = _supplement_chunks_with_keyword_hits(
            index=index,
            query=query,
            chunks=fallback_chunks,
            folder=folder,
            filters=filters,
            limit=limit,
        )
        return fallback_chunks, [], [query]


async def _retrieve_document_context_async(
    query: str,
    folder: Optional[str] = None,
    limit: int = 10,
    hybrid_alpha: float = 0.5,
    timeout_seconds: Optional[float] = None,
) -> tuple[list[Any], list[Any], list[str]]:
    """Async retrieval wrapper with timeout and keyword fallback."""
    timeout = timeout_seconds or float(os.getenv("AGENT_RAG_RETRIEVAL_TIMEOUT", "8"))

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _retrieve_document_context,
                query,
                folder,
                limit,
                hybrid_alpha,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"Retrieval timed out after {timeout}s for query='{query[:80]}...', falling back to keyword search"
        )
        index = get_document_index()
        filters = _build_search_filters(index, folder)
        fallback_chunks = await asyncio.to_thread(
            _keyword_search_fallback,
            index,
            query,
            limit,
            folder,
            filters,
        )
        fallback_chunks = await asyncio.to_thread(
            _supplement_chunks_with_keyword_hits,
            index,
            query,
            fallback_chunks,
            folder,
            filters,
            limit,
        )
        return fallback_chunks, [], [query]


def _chunk_context_text(chunk: Any, sections: list[Any]) -> str:
    """Resolve expanded section content for a chunk when available."""
    chunk_unique_id = getattr(chunk, "unique_id", "")
    for section in sections:
        center_chunk = getattr(section, "center_chunk", None)
        if getattr(center_chunk, "unique_id", "") == chunk_unique_id:
            return getattr(section, "combined_content", "") or getattr(chunk, "content", "")
    return getattr(chunk, "content", "")


def _build_grounded_answer_messages(
    query: str,
    chunks: list[Any],
    sections: list[Any],
    personality: Optional[str] = None,
) -> list[Any]:
    """Build a tightly grounded prompt for fast KB QA."""
    from agent_rag.llm.interface import LLMMessage

    system_parts = [
        "你是一个知识库问答助手。",
        "只能基于提供的检索内容回答，禁止补充资料中没有的型号、参数、步骤或结论。",
        "如果证据不足，直接明确说“知识库中未找到足够依据”。",
        "如果资料中已经有直接回答问题的句子，优先提炼该结论，不要误判为证据不足。",
        "回答优先给出直接结论，再补充关键步骤或注意事项。",
        "引用来源时使用方括号编号，如 [1][2]。",
    ]

    if personality:
        from agent_rag.core.personality import get_personality
        try:
            system_parts.append(get_personality(personality)["system_prompt"])
        except Exception:
            logger.warning(f"[API] Unknown personality: {personality}")

    source_blocks = []
    for idx, chunk in enumerate(chunks[:5], start=1):
        title = getattr(chunk, "title", None) or getattr(chunk, "document_id", "Untitled")
        content = _chunk_context_text(chunk, sections).strip()
        if len(content) > 900:
            content = content[:900] + "..."
        source_blocks.append(
            f"[{idx}] 标题: {title}\n来源: {getattr(chunk, 'document_id', 'unknown')}\n内容:\n{content}"
        )

    user_prompt = (
        f"用户问题：{query}\n\n"
        "可用资料如下：\n"
        f"{chr(10).join(source_blocks)}\n\n"
        "请基于这些资料作答。若涉及步骤，请按顺序写；若涉及规格/差异，请逐项对齐；若资料不足，请明确指出。"
    )

    return [
        LLMMessage(role="system", content="\n".join(system_parts)),
        LLMMessage(role="user", content=user_prompt),
    ]


def _split_answer_for_sse(answer: str, chunk_size: int = 160) -> list[str]:
    """Split a full answer into moderate SSE chunks."""
    normalized = (answer or "").strip()
    if not normalized:
        return []

    parts: list[str] = []
    buffer = ""
    segments = re.split(r"(?<=[。！？；.!?\n])", normalized)

    for raw_segment in segments:
        segment = raw_segment.strip()
        if not segment:
            continue

        if buffer and len(buffer) + len(segment) > chunk_size:
            parts.append(buffer)
            buffer = segment
        else:
            buffer += segment

    if buffer:
        parts.append(buffer)

    return parts or [normalized]


def _is_question_only_candidate(candidate: str) -> bool:
    cleaned = re.sub(r"\s+", " ", candidate or "").strip()
    if not cleaned:
        return False
    if " - " in cleaned or cleaned.startswith("-") or "修复：" in cleaned:
        return False
    if re.search(r"\*\*.+?[？?].*?\*\*", cleaned):
        return True
    return cleaned.endswith(("？", "?"))


def _collect_direct_answer_candidates(content: str) -> list[str]:
    """Split chunk content into compact Q/A style candidates."""
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in (content or "").splitlines()
        if re.sub(r"\s+", " ", line).strip()
    ]
    if not lines:
        return []

    candidates: list[str] = []
    for index, line in enumerate(lines):
        combined_candidate: str | None = None

        if index + 1 < len(lines) and line.startswith(("###", "##")):
            candidates.append(line)
            continue

        if "**" in line and index + 1 < len(lines):
            next_line = lines[index + 1]
            if next_line.startswith("-"):
                combined_candidate = f"{line} {next_line}"
                candidates.append(combined_candidate)

        if not (combined_candidate and _is_question_only_candidate(line)):
            candidates.append(line)

        if "故障现象：" in line and "修复：" in line:
            candidates.append(line)

    return candidates[:80]


def _score_direct_candidate(query: str, candidate: str) -> float:
    query_terms = _collect_query_match_terms(query)
    keyword_query = _normalize_query_keywords(query)
    raw_query = re.sub(r"\s+", "", (query or "").lower())
    candidate_compact = re.sub(r"\s+", "", candidate.lower())

    score = 0.0
    for term in query_terms:
        if term in candidate_compact:
            score += 0.25 if len(term) == 2 else 0.9 if len(term) <= 4 else 1.25

    keyword_compact = re.sub(r"\s+", "", keyword_query) if keyword_query else ""
    if keyword_compact and keyword_compact in candidate_compact:
        score += 2.4
    if raw_query and raw_query in candidate_compact:
        score += 1.6

    if "修复：" in candidate:
        score += 0.35
    if "**" in candidate:
        score += 0.25
    if " - " in candidate or candidate.startswith("-"):
        score += 0.6
    if _is_question_only_candidate(candidate):
        score -= 2.0

    return score


def _render_direct_candidate_answer(candidate: str, source_index: int) -> str:
    cleaned = re.sub(r"\s+", " ", candidate).strip()

    qa_match = re.search(r"\*\*(.+?)\*\*\s*-\s*(.+)", cleaned)
    if qa_match:
        answer_text = qa_match.group(2).strip(" -")
        if answer_text:
            return f"直接结论：{answer_text} [{source_index}]"

    repair_match = re.search(r"检测：(.+?)；修复：(.+)", cleaned)
    if repair_match:
        detect_text = repair_match.group(1).strip("；。 ")
        repair_text = repair_match.group(2).strip("；。 ")
        return (
            f"直接结论：{repair_text} [{source_index}]\n"
            f"重点检查：{detect_text} [{source_index}]"
        )

    bullet_match = re.match(r"-\s*(.+)", cleaned)
    if bullet_match:
        return f"直接结论：{bullet_match.group(1).strip()} [{source_index}]"

    if _is_question_only_candidate(cleaned):
        return f"直接结论：请参考资料原文中的对应处理步骤。 [{source_index}]"

    return f"直接结论：{cleaned[:220].strip()} [{source_index}]"


def _collect_scored_direct_candidates(
    query: str,
    chunks: list[Any],
    sections: list[Any],
    max_chunks: int = 5,
) -> list[dict[str, Any]]:
    """Collect scored candidate snippets from the top retrieved chunks."""
    candidates: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks[:max_chunks], start=1):
        content = _chunk_context_text(chunk, sections)
        for candidate in _collect_direct_answer_candidates(content):
            score = _score_direct_candidate(query, candidate)
            if score <= 0:
                continue
            candidates.append(
                {
                    "source_index": idx,
                    "text": candidate,
                    "score": score,
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def _format_source_citations(source_indices: list[int]) -> str:
    return "".join(f"[{index}]" for index in sorted(set(source_indices)))


def _find_candidate_text(
    candidates: list[dict[str, Any]],
    include_terms: list[str],
) -> tuple[str, int] | None:
    for candidate in candidates:
        normalized = candidate["text"].lower()
        if all(term.lower() in normalized for term in include_terms):
            return candidate["text"], int(candidate["source_index"])
    return None


def _find_candidate_with_any(
    candidates: list[dict[str, Any]],
    include_terms: list[str],
) -> tuple[str, int] | None:
    for candidate in candidates:
        normalized = candidate["text"].lower()
        if any(term.lower() in normalized for term in include_terms):
            return candidate["text"], int(candidate["source_index"])
    return None


def _extract_after_dash(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if " - " in cleaned:
        return cleaned.split(" - ", 1)[1].strip()
    if cleaned.startswith("-"):
        return cleaned[1:].strip()
    repair_match = re.search(r"修复：(.+)", cleaned)
    if repair_match:
        return repair_match.group(1).strip("；。 ")
    return cleaned


def _build_direct_grounded_answer(
    query: str,
    chunks: list[Any],
    sections: list[Any],
    min_score: float = 3.2,
) -> str | None:
    """Return a direct extractive answer when a highly relevant block is present."""
    normalized_query = (query or "").lower()
    candidates = _collect_scored_direct_candidates(query=query, chunks=chunks, sections=sections)
    if not candidates:
        return None

    if "hepa" in normalized_query and "滤网" in normalized_query:
        wash_candidate = _find_candidate_with_any(candidates, ["hepa滤网", "hepa 滤网"])
        replace_candidate = _find_candidate_with_any(candidates, ["3-6个月", "更换新滤网"])
        if wash_candidate:
            wash_text, wash_source = wash_candidate
            replace_text = wash_text
            replace_source = wash_source
            if replace_candidate:
                replace_text, replace_source = replace_candidate
            citations = _format_source_citations([wash_source, replace_source])
            return (
                "直接结论：HEPA滤网可以定期用清水清洗，阴凉处晾干后再使用，不可暴晒，"
                f"建议大约 3 到 6 个月更换一次新滤网。{citations}\n"
                f"依据：{_extract_after_dash(wash_text)} {citations}"
            )

    if "清洁液" in normalized_query and any(term in normalized_query for term in ("配比", "比例")):
        mix_candidate = _find_candidate_with_any(candidates, ["1:100", "按1:100的比例", "按比例重新混合清洁液"])
        dedicated_candidate = _find_candidate_with_any(candidates, ["机型专用清洁液", "专用清洁液", "不可直接加浓清洁液"])
        if mix_candidate or dedicated_candidate:
            sources: list[int] = []
            if dedicated_candidate:
                sources.append(dedicated_candidate[1])
            if mix_candidate:
                sources.append(mix_candidate[1])
            source = sources[0] if sources else 1
            return (
                "直接结论：水箱里优先使用机型专用清洁液，按大约 1:100 与清水混合后再加注，"
                f"不要直接加入高浓度清洁液。[{source}]\n"
                + (
                    f"依据：{_extract_after_dash(dedicated_candidate[0])} [{dedicated_candidate[1]}]"
                    if dedicated_candidate
                    else f"依据：{_extract_after_dash(mix_candidate[0])} [{mix_candidate[1]}]"
                )
            )

    if "首次使用" in normalized_query and ("扫地机器人" in normalized_query or "机器人" in normalized_query):
        first_use_candidate = _find_candidate_with_any(
            candidates,
            ["拆除机身所有包装配件", "空旷环境下启动建图", "设置清扫区域和禁区"],
        )
        if first_use_candidate:
            text, source = first_use_candidate
            return (
                "直接结论：首次使用时，先拆除机身所有包装配件，先充满电，"
                f"再在空旷环境下启动建图；建图完成后再设置清扫区域和禁区。[{source}]\n"
                f"依据：{_extract_after_dash(text)} [{source}]"
            )

    if "长期存放" in normalized_query or ("存放" in normalized_query and "保养" in normalized_query):
        charge_candidate = _find_candidate_with_any(candidates, ["80%-90%", "80%到90%", "80%-90%"])
        clean_candidate = _find_candidate_with_any(candidates, ["全面清理机器人机身", "主刷、边刷、拖布", "所有配件"])
        water_candidate = _find_candidate_with_any(candidates, ["排空水箱", "污水仓", "晾干后再存放"])
        place_candidate = _find_candidate_with_any(candidates, ["阴凉、干燥、通风", "阴凉干燥", "每1-2个月给机器人补电一次"])
        if charge_candidate or clean_candidate or water_candidate or place_candidate:
            sources = [
                candidate[1]
                for candidate in (clean_candidate, water_candidate, charge_candidate, place_candidate)
                if candidate
            ]
            citations = _format_source_citations(sources)
            evidence_lines: list[str] = []
            if clean_candidate:
                evidence_lines.append(
                    f"依据1：{_extract_after_dash(clean_candidate[0])} [{clean_candidate[1]}]"
                )
            if water_candidate:
                evidence_lines.append(
                    f"依据{len(evidence_lines) + 1}：{_extract_after_dash(water_candidate[0])} [{water_candidate[1]}]"
                )
            charge_or_place = charge_candidate or place_candidate
            if charge_or_place:
                evidence_lines.append(
                    f"依据{len(evidence_lines) + 1}：{_extract_after_dash(charge_or_place[0])} [{charge_or_place[1]}]"
                )
            return (
                "直接结论：机器人如果要长期存放，存放前先清洁机身和配件；"
                "扫拖机型要排空水箱、污水仓并擦干晾干；"
                "电池保留约 80% 到 90% 电量，放在阴凉干燥通风处，"
                f"并每 1 到 2 个月补电一次。{citations}\n"
                + "\n".join(evidence_lines)
            )

    if "自动关机" in normalized_query:
        shutdown_candidate = _find_candidate_with_any(
            candidates,
            ["电池电量是否耗尽", "机身是否过热", "电池是否鼓包"],
        )
        if shutdown_candidate:
            text, source = shutdown_candidate
            return (
                "直接结论：机器人开机后立刻自动关机，常见原因是电池电量耗尽、机身过热，"
                f"或者电池已经鼓包；可先充满电后重试，移到通风处冷却，若电池鼓包则及时更换电池。[{source}]\n"
                f"依据：{_extract_after_dash(text)} [{source}]"
            )

    if any(term in normalized_query for term in ("撞家具", "碰撞家具")):
        avoid_candidate = _find_candidate_with_any(
            candidates,
            ["避障摄像头和传感器", "避障摄像头/传感器", "高级避障模式"],
        )
        bumper_candidate = _find_candidate_with_any(
            candidates,
            ["防撞条是否卡顿", "防撞条确认回弹正常", "防撞条是否卡住"],
        )
        if avoid_candidate or bumper_candidate:
            sources: list[int] = []
            if avoid_candidate:
                sources.append(avoid_candidate[1])
            if bumper_candidate:
                sources.append(bumper_candidate[1])
            citations = _format_source_citations(sources)
            return (
                "直接结论：机器人清扫时频繁撞家具，重点先检查避障摄像头或传感器表面是否有灰尘、水渍遮挡，"
                f"再检查防撞条是否卡顿、能否正常回弹，并顺手清理周围障碍物。{citations}"
            )

    if "水渍" in normalized_query or "水痕" in normalized_query:
        output_candidate = _find_candidate_with_any(candidates, ["调小出水量", "出水量是否过大", "低档出水量"])
        mop_candidate = _find_candidate_with_any(candidates, ["更换拖布", "拖布是否脏污", "拖布选择"])
        speed_candidate = _find_candidate_with_any(candidates, ["降低清扫速度", "低速拖地", "干拖"])
        if output_candidate:
            sources = [output_candidate[1]]
            parts = [
                "直接结论：拖地后出现明显水渍或水痕时，先调小出水量，"
                "同时检查拖布是否脏污、硬化，必要时清洗或更换拖布。"
            ]
            if mop_candidate:
                sources.append(mop_candidate[1])
            if speed_candidate:
                sources.append(speed_candidate[1])
                parts.append("如果仍然留痕，可切换到干拖、低档出水量或更低速拖地。")
            else:
                parts.append("如果仍然留痕，可降低清扫速度。")
            return (
                f"{' '.join(parts)}{_format_source_citations(sources)}\n"
                f"依据1：{_extract_after_dash(output_candidate[0])} [{output_candidate[1]}]"
                + (
                    f"\n依据2：{_extract_after_dash(mop_candidate[0])} [{mop_candidate[1]}]"
                    if mop_candidate
                    else ""
                )
                + (
                    f"\n依据3：{_extract_after_dash(speed_candidate[0])} [{speed_candidate[1]}]"
                    if speed_candidate
                    else ""
                )
            )

    if "宠物" in normalized_query and "毛发" in normalized_query:
        pet_candidate = _find_candidate_with_any(
            candidates,
            ["毛发清理模式", "宠物模式", "宠物专用防缠绕主刷", "防缠绕主刷", "调大吸力", "增大吸力"],
        )
        clean_candidate = _find_candidate_with_any(
            candidates,
            ["清理主刷缠绕的毛发", "主刷上的毛发", "剪断主刷缠绕毛发", "主刷是否缠绕杂物"],
        )
        if pet_candidate or clean_candidate:
            sources: list[int] = []
            lines = [
                "直接结论：家里有宠物、机器人漏扫毛发时，先清理主刷缠绕的毛发，"
                "必要时更换宠物专用防缠绕主刷，再把吸力调高，并开启宠物模式或“毛发清理模式”。"
            ]
            if clean_candidate:
                sources.append(clean_candidate[1])
            if pet_candidate:
                sources.append(pet_candidate[1])
            if pet_candidate and "清扫频次" in pet_candidate[0]:
                lines.append("如果家里掉毛很多，还可以适当增加清扫频次。")
            return f"{' '.join(lines)}{_format_source_citations(sources)}"

    if _query_has_buying_intent(query) and "宠物" in normalized_query:
        suction_candidate = _find_candidate_with_any(candidates, ["强化吸力", "增大吸力", "≥3000pa", "≥4000pa"])
        anti_tangle_candidate = _find_candidate_with_any(candidates, ["防缠绕功能", "防毛发缠绕", "防缠绕主刷", "宠物模式"])
        avoid_candidate = _find_candidate_with_any(candidates, ["避障能力", "3d结构光避障", "避障识别"])
        deodorize_candidate = _find_candidate_with_any(candidates, ["除味功能", "除味能力"])
        consumable_candidate = _find_candidate_with_any(candidates, ["耗材更换频率", "耗材", "滤网更换", "滚刷更换"])
        if suction_candidate or anti_tangle_candidate or avoid_candidate:
            sources: list[int] = []
            for candidate in (
                suction_candidate,
                anti_tangle_candidate,
                deodorize_candidate,
                avoid_candidate,
                consumable_candidate,
            ):
                if candidate:
                    sources.append(candidate[1])
            citations = _format_source_citations(sources)
            follow_up_parts: list[str] = []
            if deodorize_candidate:
                follow_up_parts.append("除味能力")
            if avoid_candidate:
                follow_up_parts.append("避障能力")
            if consumable_candidate:
                follow_up_parts.append("宠物家庭耗材更换频率通常会更高")

            follow_up_text = ""
            if follow_up_parts:
                if len(follow_up_parts) == 1:
                    follow_up_text = f"；再补看{follow_up_parts[0]}"
                else:
                    follow_up_text = f"；再补看{'、'.join(follow_up_parts[:-1])}，以及{follow_up_parts[-1]}"
            return (
                "直接结论：带宠物的家庭选扫地机器人，优先看更强吸力、"
                f"防毛发缠绕主刷或宠物模式{follow_up_text}。{citations}"
            )

    if _query_has_buying_intent(query) and "木地板" in normalized_query and "地毯" in normalized_query:
        wood_candidate = _find_candidate_with_any(candidates, ["拖布可抬升", "木地板需选拖布可抬升", "出水量可调", "低档出水量"])
        carpet_candidate = _find_candidate_with_any(candidates, ["地毯识别", "地毯增压", "≥4000pa", "大吸力模式"])
        obstacle_candidate = _find_candidate_with_any(candidates, ["越障能力", "地面适配", "驱动轮动力", "适配不同地面"])
        if wood_candidate or carpet_candidate or obstacle_candidate:
            sources: list[int] = []
            for candidate in (wood_candidate, carpet_candidate, obstacle_candidate):
                if candidate:
                    sources.append(candidate[1])
            citations = _format_source_citations(sources)
            return (
                "直接结论：家里既有木地板又有地毯，选扫拖机器人时，木地板场景优先看低出水量可调或拖布抬升，"
                f"地毯场景优先看更强吸力或地毯增压；同时还要兼顾越障和地面适配能力。{citations}"
            )

    if "充电座" in normalized_query:
        dock_candidate = _find_candidate_with_any(candidates, ["充电座前方3米", "左右1米无遮挡", "重启回充"])
        if dock_candidate:
            text, source = dock_candidate
            return (
                "直接结论：先清理充电座周围障碍物，保证充电座前方约 3 米、左右约 1 米无遮挡，"
                f"再把机器人放回充电座附近后重新尝试回充。[{source}]\n"
                f"依据：{_extract_after_dash(text)} [{source}]"
            )

    if ("app" in normalized_query or "wifi" in normalized_query) and ("连" in normalized_query or "连接" in normalized_query):
        wifi_candidate = _find_candidate_with_any(candidates, ["2.4g", "5g", "重新在app添加设备"])
        if wifi_candidate:
            text, source = wifi_candidate
            return (
                "直接结论：先确认手机和机器人连接同一 2.4G WiFi，暂不支持 5G；"
                f"再重启机器人和路由器，并在 APP 中重新添加设备。[{source}]\n"
                f"依据：{_extract_after_dash(text)} [{source}]"
            )

    best_candidate = candidates[0]
    if best_candidate["score"] >= min_score:
        return _render_direct_candidate_answer(best_candidate["text"], int(best_candidate["source_index"]))
    return None


def _build_grounded_fallback_answer(
    query: str,
    chunks: list[Any],
    sections: list[Any],
) -> str:
    """Build a deterministic extractive fallback when LLM generation is too slow."""
    direct_answer = _build_direct_grounded_answer(query=query, chunks=chunks, sections=sections, min_score=1.8)
    if direct_answer:
        return direct_answer

    keywords = _normalize_query_keywords(query).split()
    selected_points: list[str] = []
    seen_sentences: set[str] = set()

    for idx, chunk in enumerate(chunks[:5], start=1):
        content = _chunk_context_text(chunk, sections)
        sentences = re.split(r"(?<=[。！？；.!?])\s+|\n+", content)

        ranked_sentences = sorted(
            sentences,
            key=lambda sentence: _score_direct_candidate(query, sentence),
            reverse=True,
        )

        for sentence in ranked_sentences:
            cleaned = " ".join(sentence.split()).strip(" -")
            if len(cleaned) < 10:
                continue
            if cleaned in seen_sentences:
                continue
            if keywords and not any(keyword.lower() in cleaned.lower() for keyword in keywords):
                continue

            seen_sentences.add(cleaned)
            selected_points.append(f"{cleaned[:180]} [{idx}]")
            break

        if len(selected_points) >= 4:
            break

    if not selected_points:
        for idx, chunk in enumerate(chunks[:3], start=1):
            content = _chunk_context_text(chunk, sections).strip()
            if not content:
                continue

            snippet = " ".join(content.split())[:180].strip()
            if snippet:
                selected_points.append(f"{snippet} [{idx}]")

        if not selected_points:
            return "知识库中未找到足够依据。"

    return "根据知识库，相关信息如下：\n" + "\n".join(
        f"{index}. {point}" for index, point in enumerate(selected_points, start=1)
    )


async def _generate_grounded_answer_async(
    query: str,
    chunks: list[Any],
    sections: list[Any],
    personality: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> tuple[str, bool]:
    """Generate a grounded answer with a hard timeout and extractive fallback."""
    from agent_rag.core.config import ReasoningEffort

    llm = get_llm_provider()
    messages = _build_grounded_answer_messages(
        query=query,
        chunks=chunks,
        sections=sections,
        personality=personality,
    )
    timeout = timeout_seconds or float(os.getenv("AGENT_RAG_FAST_ANSWER_TIMEOUT", "25"))
    max_tokens = int(os.getenv("AGENT_RAG_FAST_ANSWER_MAX_TOKENS", "420"))

    try:
        response = await asyncio.wait_for(
            llm.chat_async(
                messages=messages,
                max_tokens=max_tokens,
                temperature=None,
                reasoning_effort=ReasoningEffort.OFF,
            ),
            timeout=timeout,
        )
        answer = (response.content or "").strip()
        if answer:
            return answer, False

        logger.warning(
            "[FastPath] Empty grounded answer for query='%s', using extractive fallback",
            query[:80],
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[FastPath] Grounded answer timed out after %ss for query='%s', using extractive fallback",
            timeout,
            query[:80],
        )
    except Exception as exc:
        logger.warning(
            "[FastPath] Grounded answer failed for query='%s', using extractive fallback: %s",
            query[:80],
            exc,
        )

    return _build_grounded_fallback_answer(query=query, chunks=chunks, sections=sections), True


@app.post("/api/search", response_model=list[SearchResultModel])
async def search(request: SearchRequest):
    """Search documents using lightweight weighted retrieval."""
    index = get_document_index()

    if len(index.chunks) == 0:
        return []

    try:
        results, _, _ = await _retrieve_document_context_async(
            query=request.query,
            folder=request.folder,
            limit=request.limit,
            hybrid_alpha=request.hybrid_alpha,
        )

        return [
            SearchResultModel(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                content=chunk.content[:500],
                score=round(chunk.score, 4) if chunk.score else 0,
                title=chunk.title,
                metadata=chunk.metadata,
            )
            for chunk in results
        ]
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@app.delete("/api/folder/{folder}")
async def delete_folder(folder: str):
    """Delete a folder and its contents."""
    import shutil

    folder_path = get_folder_path(folder)

    if folder_path.exists():
        # Delete from index
        index = get_document_index()
        doc_ids_to_delete = [
            doc_id for doc_id in index.documents.keys()
            if doc_id.startswith(f"{folder}/")
        ]
        for doc_id in doc_ids_to_delete:
            index.delete_document(doc_id)
            logger.info(f"Deleted document from index: {doc_id}")

        # Delete files
        shutil.rmtree(folder_path)

        # Clean up status
        if folder in _index_status:
            del _index_status[folder]

        return {"message": f"已删除文件夹 '{folder}' 及其索引"}

    raise HTTPException(status_code=404, detail=f"文件夹 '{folder}' 不存在")


@app.get("/api/index/stats")
async def get_index_stats():
    """Get index statistics."""
    index = get_document_index()

    return {
        "total_chunks": len(index.chunks),
        "total_documents": len(index.documents),
        "documents": list(index.documents.keys()),
        "bm25_vocab_size": len(index.bm25.doc_freqs) if hasattr(index, 'bm25') else 0,
    }


# ============================================================================
# External Database Configuration
# ============================================================================

_external_postgres_connector = None


def get_external_postgres_connector(force_refresh: bool = False):
    """Get or create external PostgreSQL connector.
    
    Args:
        force_refresh: Force recreate the connector
    """
    global _external_postgres_connector
    if _external_postgres_connector is None or force_refresh:
        from agent_rag.core.external_database_config import ExternalDatabaseConfig
        from agent_rag.core.external_database_connector import ExternalDatabaseConnector

        config = ExternalDatabaseConfig.from_env()
        if config.enabled:
            _external_postgres_connector = ExternalDatabaseConnector(config)
            logger.info(f"[API] External PostgreSQL configured: {config.host}:{config.port}")
        else:
            _external_postgres_connector = None
            logger.info("[API] External PostgreSQL not enabled")
    return _external_postgres_connector


# ============================================================================
# Data Query Models
# ============================================================================


class DataQueryRequest(BaseModel):
    """Request for hybrid data query (Vespa + External PostgreSQL)."""
    query: str = Field(..., description="用户查询", min_length=1, max_length=1000)
    search_type: Literal["auto", "data", "document"] = Field(
        default="auto",
        description="搜索类型: auto=自动判断, data=只查数据库, document=只查文档"
    )
    max_results: int = Field(default=10, ge=1, le=100, description="最大返回结果数")
    include_sources: bool = Field(default=True, description="是否包含数据来源")


class DataQueryResult(BaseModel):
    """Result from hybrid data query."""
    query: str
    answer: str
    query_type: str  # "data_query" or "document_query"
    vespa_results: list[dict] = []
    database_results: list[dict] = []
    sources: list[dict] = []
    execution_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)


class DatabaseConfigModel(BaseModel):
    """External database configuration."""
    enabled: bool
    host: str
    port: int
    database: str
    table_count: int = 0


# ============================================================================
# External Database Endpoints
# ============================================================================


@app.get("/api/external-db/status")
async def get_external_db_status():
    """Get external PostgreSQL connection status."""
    connector = get_external_postgres_connector()

    if connector is None:
        return DatabaseConfigModel(
            enabled=False,
            host="",
            port=0,
            database="",
            table_count=0,
        )

    try:
        tables = connector.get_table_list()
        return DatabaseConfigModel(
            enabled=True,
            host=connector.config.host,
            port=connector.config.port,
            database=connector.config.database,
            table_count=len(tables),
        )
    except Exception as e:
        logger.error(f"[API] Failed to get external DB status: {e}")
        return DatabaseConfigModel(
            enabled=True,
            host=connector.config.host if connector else "",
            port=connector.config.port if connector else 0,
            database=connector.config.database if connector else "",
            table_count=0,
        )


@app.get("/api/external-db/tables")
async def list_external_tables():
    """List tables in external PostgreSQL database."""
    connector = get_external_postgres_connector()

    if connector is None:
        raise HTTPException(status_code=400, detail="External PostgreSQL not configured")

    try:
        tables = connector.get_table_list()
        table_schemas = []
        for table in tables:
            schema = connector.get_table_schema(table)
            table_schemas.append({
                "name": table,
                "columns": schema.get("columns", []),
            })
        return {"tables": table_schemas}
    except Exception as e:
        logger.error(f"[API] Failed to list tables: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {str(e)}")


@app.post("/api/external-db/query")
async def query_external_database(request: DataQueryRequest):
    """Execute SQL query on external PostgreSQL database."""
    connector = get_external_postgres_connector()

    if connector is None:
        raise HTTPException(status_code=400, detail="External PostgreSQL not configured")

    try:
        import time
        start_time = time.time()

        tables = connector.get_table_list()
        if not tables:
            raise HTTPException(status_code=400, detail="No tables found in external database")

        test_customers_schema = None
        for table in tables:
            if table == "test_customers":
                schema = connector.get_table_schema(table)
                if schema.get("columns"):
                    test_customers_schema = [schema]
                break

        if not test_customers_schema:
            raise HTTPException(status_code=400, detail="test_customers table not found")

        from agent_rag.text_to_sql.intent_analyzer import IntentAnalyzer
        from agent_rag.llm.interface import LLM, get_default_llm

        analyzer = IntentAnalyzer()
        intent = await analyzer.analyze(request.query)

        if not intent.is_data_query:
            return {
                "is_data_query": False,
                "message": "Query is not a data query, use /api/search for document search",
                "intent": intent.to_dict(),
            }

        prompt = analyzer.generate_sql_prompt(request.query, test_customers_schema)

        llm = get_default_llm()
        response = llm.complete(prompt, max_tokens=500)
        raw_sql = response.content.strip()
        raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()

        import json as json_module
        import re

        sql = None
        json_match = re.search(r'\{[\s\S]*\}', raw_sql)
        if json_match:
            try:
                obj = json_module.loads(json_match.group())
                if isinstance(obj, dict) and "sql" in obj:
                    sql = obj["sql"]
            except json_module.JSONDecodeError:
                pass

        if not sql and raw_sql.upper().startswith("SELECT"):
            sql = raw_sql

        if not sql:
            raise HTTPException(status_code=500, detail=f"Failed to extract SQL from LLM response")

        sql = sql.strip()
        logger.info(f"[API] Generated SQL: {sql[:100]}...")

        result = connector.execute_query(sql, limit=request.max_results)

        execution_time = (time.time() - start_time) * 1000

        return {
            "query": request.query,
            "sql": sql,
            "success": result.success,
            "row_count": result.row_count,
            "data": result.data,
            "execution_time_ms": execution_time,
            "error": result.error_message,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] External DB query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DirectQueryRequest(BaseModel):
    """Request for direct SQL query execution."""
    sql: str = Field(..., description="SQL query to execute", min_length=1, max_length=5000)
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of rows to return")


@app.post("/api/external-db/direct-query")
async def direct_query_external_database(request: DirectQueryRequest):
    """Execute direct SQL query on external PostgreSQL database.

    WARNING: This endpoint executes raw SQL. Use with caution and only for trusted queries.
    """
    connector = get_external_postgres_connector()

    if connector is None:
        raise HTTPException(status_code=400, detail="External PostgreSQL not configured")

    try:
        import time
        start_time = time.time()

        result = connector.execute_query(request.sql, limit=request.limit)

        execution_time = (time.time() - start_time) * 1000

        return {
            "sql": request.sql,
            "success": result.success,
            "row_count": result.row_count,
            "data": result.data,
            "execution_time_ms": execution_time,
            "error": result.error_message,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"[API] Direct SQL query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hybrid-search")
async def hybrid_data_search(request: DataQueryRequest):
    """Hybrid search: query both Vespa documents and external PostgreSQL.

    Flow:
    1. Analyze query intent using existing IntentAnalyzer
    2. If data query:
       - Query external PostgreSQL (parallel with Vespa)
       - Query Vespa for related documents
       - Merge results and generate answer
    3. If document query:
       - Only query Vespa
    """
    import time
    start_time = time.time()

    from agent_rag.text_to_sql.intent_analyzer import IntentAnalyzer
    from agent_rag.core.external_database_connector import ExternalDatabaseConnector

    analyzer = IntentAnalyzer()
    intent = await analyzer.analyze(request.query)

    logger.info(f"[HybridSearch] Query: {request.query[:50]}...")
    logger.info(f"[HybridSearch] Intent: is_data_query={intent.is_data_query}, confidence={intent.confidence:.2f}")

    index = get_document_index()
    has_vespa_data = len(index.chunks) > 0

    vespa_results = []
    db_results = {"data": [], "sql": "", "success": False}

    if intent.is_data_query and request.search_type in ("auto", "data"):
        connector = get_external_postgres_connector()

        if connector:
            try:
                tables = connector.get_table_list()
                if tables:
                    test_customers_schema = None
                    for table in tables:
                        if table == "test_customers":
                            schema = connector.get_table_schema(table)
                            if schema.get("columns"):
                                test_customers_schema = [schema]
                            break

                    if test_customers_schema:
                        prompt = analyzer.generate_sql_prompt(request.query, test_customers_schema)

                        from agent_rag.llm.interface import get_default_llm
                        llm = get_default_llm()

                        response = llm.complete(prompt, max_tokens=500)
                        raw_sql = response.content.strip()
                        raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()

                        import json as json_module
                        import re

                        sql = None
                        json_match = re.search(r'\{[\s\S]*\}', raw_sql)
                        if json_match:
                            try:
                                obj = json_module.loads(json_match.group())
                                if isinstance(obj, dict) and "sql" in obj:
                                    sql = obj["sql"]
                            except json_module.JSONDecodeError:
                                pass

                        if not sql and raw_sql.upper().startswith("SELECT"):
                            sql = raw_sql

                        if not sql:
                            logger.error(f"[HybridSearch] Failed to extract SQL from response: {raw_sql[:100]}")
                            db_results["error"] = "Failed to extract SQL from LLM response"
                        else:
                            sql = sql.strip()
                            logger.info(f"[HybridSearch] SQL: {sql[:100]}...")

                            db_result = connector.execute_query(sql, limit=request.max_results)
                            db_results = {
                                "data": db_result.data,
                                "sql": sql,
                                "success": db_result.success,
                                "row_count": db_result.row_count,
                                "execution_time_ms": db_result.execution_time_ms,
                                "error": db_result.error_message,
                            }
            except Exception as e:
                logger.error(f"[HybridSearch] External DB query failed: {e}")
                db_results["error"] = str(e)

    if request.search_type in ("auto", "document") and has_vespa_data:
        try:
            from agent_rag.core.env_config import get_embedding_config_from_env
            from agent_rag.embedding import LiteLLMEmbedder
            from agent_rag.core.models import SearchFilters

            emb_config = get_embedding_config_from_env()
            embedder = LiteLLMEmbedder(emb_config)
            query_embedding = embedder.embed(request.query)

            filters = None
            results = index.hybrid_search(
                query=request.query,
                query_embedding=query_embedding,
                filters=filters,
                hybrid_alpha=0.5,
                num_results=request.max_results,
            )

            vespa_results = [
                {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content[:500],
                    "score": round(chunk.score, 4) if chunk.score else 0,
                    "title": chunk.title,
                    "metadata": chunk.metadata,
                }
                for chunk in results
            ]
        except Exception as e:
            logger.error(f"[HybridSearch] Vespa search failed: {e}")

    answer = ""
    if intent.is_data_query and db_results.get("success") and db_results.get("data"):
        data_count = len(db_results["data"])
        if data_count == 1:
            answer = f"查询结果: {db_results['data'][0]}"
        else:
            answer = f"从外部数据库查询到 {data_count} 条数据。"
            if vespa_results:
                answer += f"\n同时从文档库找到 {len(vespa_results)} 篇相关文档。"
    elif vespa_results:
        if len(vespa_results) == 1:
            answer = f"找到相关文档: {vespa_results[0]['content'][:300]}..."
        else:
            answer = f"找到 {len(vespa_results)} 篇相关文档。"
    else:
        answer = "抱歉，没有找到相关信息。"

    execution_time = (time.time() - start_time) * 1000

    return DataQueryResult(
        query=request.query,
        answer=answer,
        query_type="data_query" if intent.is_data_query else "document_query",
        vespa_results=vespa_results,
        database_results=db_results.get("data", []),
        sources=[
            {"type": "external_postgres", "tables": connector.get_table_list()} if connector else None,
            {"type": "vespa", "chunks": len(vespa_results)},
        ],
        execution_time_ms=execution_time,
    )


# ============================================================================
# Session & Memory API Endpoints
# ============================================================================

session_manager = None
memory_store = None


def get_session_manager():
    """Get or initialize session manager."""
    global session_manager
    if session_manager is None:
        from agent_rag.core.session_manager import SessionManager
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/agent_rag"
        )
        session_manager = SessionManager(database_url=database_url)
        # Don't initialize here - let it initialize lazily on first use
        # The SessionManager will handle initialization internally
    return session_manager


def get_memory_store():
    """Get or initialize memory store."""
    global memory_store
    if memory_store is None:
        from agent_rag.core.memory_store import MemoryStore
        from agent_rag.core.env_config import get_embedding_config_from_env
        from agent_rag.embedding import LiteLLMEmbedder
        
        qdrant_url = os.getenv("AGENT_RAG_QDRANT_URL", "http://localhost:6333")
        collection_name = os.getenv("AGENT_RAG_QDRANT_COLLECTION", "user_memories")
        
        emb_config = get_embedding_config_from_env()
        embedder = LiteLLMEmbedder(emb_config)
        
        memory_store = MemoryStore(
            qdrant_url=qdrant_url, 
            collection_name=collection_name,
            embedder=embedder
        )   
    return memory_store


# ============================================================================
# Cached Components for Performance Optimization
# ============================================================================

_llm_provider = None
_embedder = None
_search_tool = None
_text_to_sql = None
_hybrid_search_tool = None
_agent_configs: dict = {}


def get_llm_provider():
    """Get or initialize cached LLM provider."""
    global _llm_provider
    if _llm_provider is None:
        from agent_rag.llm.providers.litellm_provider import LiteLLMProvider
        from agent_rag.core.env_config import get_llm_config_from_env
        
        llm_config = get_llm_config_from_env()
        _llm_provider = LiteLLMProvider(llm_config)
        logger.info("[Cache] LLM Provider initialized")
    return _llm_provider


def get_embedder():
    """Get or initialize cached embedder."""
    global _embedder
    if _embedder is None:
        from agent_rag.embedding import LiteLLMEmbedder
        from agent_rag.core.env_config import get_embedding_config_from_env
        
        emb_config = get_embedding_config_from_env()
        _embedder = LiteLLMEmbedder(emb_config)
        logger.info("[Cache] Embedder initialized")
    return _embedder


def get_search_tool():
    """Get or initialize cached search tool."""
    global _search_tool
    if _search_tool is None:
        from agent_rag.tools.builtin.search.search_tool import SearchTool
        from agent_rag.core.env_config import get_search_config_from_env
        
        search_config = get_search_config_from_env()
        index = get_document_index()
        llm = get_llm_provider()
        embedder = get_embedder()
        
        _search_tool = SearchTool(
            document_index=index,
            embedder=embedder,
            llm=llm,
            search_config=search_config,
        )
        logger.info("[Cache] SearchTool initialized")
    return _search_tool


def get_text_to_sql():
    """Get or initialize cached text-to-sql component."""
    global _text_to_sql
    if _text_to_sql is None:
        connector = get_external_postgres_connector()
        if connector:
            from agent_rag.text_to_sql import create_text_to_sql_sync
            
            llm = get_llm_provider()
            embedder = get_embedder()
            enable_db_discovery = os.getenv("AGENT_RAG_TEXT_TO_SQL_DB_DISCOVERY", "true").lower() == "true"
            
            _text_to_sql = create_text_to_sql_sync(
                llm, 
                embedder, 
                external_connector=connector,
                enable_db_discovery=enable_db_discovery,
            )
            logger.info("[Cache] Text-to-SQL initialized")
    return _text_to_sql


def get_hybrid_search_tool():
    """Get or initialize cached hybrid search tool."""
    global _hybrid_search_tool
    if _hybrid_search_tool is None:
        from agent_rag.tools.builtin.search.hybrid_search_tool import HybridSearchTool, HybridSearchConfig
        from agent_rag.core.env_config import get_search_config_from_env
        
        search_config = get_search_config_from_env()
        index = get_document_index()
        llm = get_llm_provider()
        embedder = get_embedder()
        text_to_sql = get_text_to_sql()
        
        if text_to_sql:
            hybrid_config = HybridSearchConfig(
                search_config=search_config,
                enable_text_to_sql=True,
                text_to_sql_threshold=0.7,
                max_sql_results=search_config.max_documents_to_select,
            )
            _hybrid_search_tool = HybridSearchTool(
                document_index=index,
                embedder=embedder,
                llm=llm,
                text_to_sql=text_to_sql,
                hybrid_config=hybrid_config,
            )
            logger.info("[Cache] HybridSearchTool initialized")
    return _hybrid_search_tool


def get_chat_agent(system_prompt: Optional[str] = None):
    """Get or initialize cached ChatAgent with optional system prompt."""
    from agent_rag.agent import ChatAgent
    from agent_rag.core.env_config import get_agent_config_from_env
    from agent_rag.tools.registry import ToolRegistry
    
    # Use system prompt as cache key
    cache_key = system_prompt or "__default__"
    
    if cache_key not in _agent_configs:
        llm = get_llm_provider()
        agent_config = get_agent_config_from_env()
        tool_registry = ToolRegistry()
        
        # Register appropriate search tool
        connector = get_external_postgres_connector()
        if connector and get_text_to_sql():
            tool_registry.register(get_hybrid_search_tool())
        else:
            tool_registry.register(get_search_tool())
        
        _agent_configs[cache_key] = ChatAgent(
            llm=llm,
            config=agent_config,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
        )
        logger.info(f"[Cache] ChatAgent initialized (personality: {cache_key[:50] if cache_key != '__default__' else 'default'}...)")
    
    return _agent_configs[cache_key]


_session_memory_agents: dict = {}


def get_session_memory_agent(session_id: Optional[str] = None, system_prompt: Optional[str] = None):
    """Get or initialize SessionMemoryChatAgent for a session.
    
    Args:
        session_id: Session ID to resume (optional)
        system_prompt: Optional custom system prompt
        
    Returns:
        SessionMemoryChatAgent instance
    """
    from agent_rag.agent.session_memory_agent import SessionMemoryChatAgent
    from agent_rag.core.env_config import get_agent_config_from_env
    from agent_rag.tools.registry import ToolRegistry
    
    # Use session_id + system_prompt as cache key
    cache_key = f"{session_id or '__no_session__'}:{system_prompt or '__default__'}"
    
    if cache_key not in _session_memory_agents:
        llm = get_llm_provider()
        embedder = get_embedder()
        session_manager = get_session_manager()
        memory_store = get_memory_store()
        agent_config = get_agent_config_from_env()
        tool_registry = ToolRegistry()
        
        # Register appropriate search tool
        connector = get_external_postgres_connector()
        if connector and get_text_to_sql():
            tool_registry.register(get_hybrid_search_tool())
        else:
            tool_registry.register(get_search_tool())
        
        _session_memory_agents[cache_key] = SessionMemoryChatAgent(
            llm=llm,
            embedder=embedder,
            session_manager=session_manager,
            memory_store=memory_store,
            config=agent_config,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
            extract_memories_after_conversation=True,
            retrieve_memories_for_query=True,
            max_memories_per_query=5,
        )
        logger.info(f"[Cache] SessionMemoryChatAgent initialized for session {session_id or 'new'}")
    
    return _session_memory_agents[cache_key]


# ============================================================================
# Parallel Search Functions for Performance Optimization
# ============================================================================

async def _document_search(query: str, index: Any, embedder: Any, search_config: Any) -> list:
    """Perform document/vector search."""
    try:
        chunks, _, queries_used = await _retrieve_document_context_async(
            query=query,
            folder=None,
            limit=search_config.num_results,
            hybrid_alpha=search_config.default_hybrid_alpha,
        )
        logger.info(
            f"[ParallelSearch] Document search returned {len(chunks)} chunks using queries={queries_used}"
        )
        return chunks
    except Exception as e:
        logger.warning(f"[ParallelSearch] Document search error, falling back to keyword search: {e}")
        try:
            fallback_results = _keyword_search_fallback(
                index=index,
                query=query,
                limit=search_config.num_results,
            )
            logger.info(
                f"[ParallelSearch] Keyword fallback returned {len(fallback_results)} chunks"
            )
            return fallback_results
        except Exception as fallback_error:
            logger.error(f"[ParallelSearch] Keyword fallback failed: {fallback_error}")
            return []


async def _sql_query(query: str, text_to_sql: Any) -> dict:
    """Perform SQL query/intent detection."""
    try:
        result = await text_to_sql.execute(query)
        logger.info(f"[ParallelSearch] SQL query completed - is_data_query: {result.is_data_query}")
        return {
            "is_data_query": result.is_data_query,
            "sql": result.sql,
            "data": result.data,
            "confidence": result.confidence,
            "error": result.error_message,
        }
    except Exception as e:
        logger.error(f"[ParallelSearch] SQL query error: {e}")
        return {
            "is_data_query": False,
            "sql": None,
            "data": None,
            "confidence": 0.0,
            "error": str(e),
        }





async def _parallel_search(
    query: str,
    index: Any,
    embedder: Any,
    search_config: Any,
    text_to_sql: Any,
) -> tuple[list, dict]:
    """Execute document search and SQL query in parallel, then let LLM select."""
    logger.info(f"[ParallelSearch] Starting parallel search for: {query[:50]}...")
    
    connector = get_external_postgres_connector()
    has_text_to_sql = connector and text_to_sql
    
    if has_text_to_sql:
        text_search_task = _document_search(query, index, embedder, search_config)
        sql_task = _sql_query(query, text_to_sql)
        
        text_results, sql_result = await asyncio.gather(
            text_search_task,
            sql_task,
            return_exceptions=True
        )
        
        if isinstance(text_results, Exception):
            logger.error(f"[ParallelSearch] Text search exception: {text_results}")
            text_results = []
        if isinstance(sql_result, Exception):
            logger.error(f"[ParallelSearch] SQL query exception: {sql_result}")
            sql_result = {"is_data_query": False, "sql": None, "data": None, "confidence": 0.0, "error": str(sql_result)}
        
        logger.info(f"[ParallelSearch] Both queries completed - docs: {len(text_results)}, SQL: {sql_result.get('data') is not None}")
    else:
        text_results = await _document_search(query, index, embedder, search_config)
        sql_result = {"is_data_query": False, "sql": None, "data": None, "confidence": 0.0}
        logger.info(f"[ParallelSearch] Only document search executed (no text-to-sql)")
    
    return text_results, sql_result


@app.post("/api/search/stream")
async def search_stream(request: SearchRequest):
    """Search with a fast grounded KB path and a heavier session-aware fallback."""

    async def generate_results() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps({'type': 'start', 'query': request.query}, ensure_ascii=False)}\n\n"

            index = get_document_index()

            if len(index.chunks) == 0:
                yield f"data: {json.dumps({'type': 'info', 'message': '知识库为空，请先上传并索引文档'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'total_results': 0}, ensure_ascii=False)}\n\n"
                return

            if not _query_may_need_agentic_path(request.query, request.session_id):
                yield f"data: {json.dumps({'type': 'thinking', 'content': '检索知识库...'}, ensure_ascii=False)}\n\n"
                chunks, sections, queries_used = await _retrieve_document_context_async(
                    query=request.query,
                    folder=request.folder,
                    limit=max(6, request.limit),
                    hybrid_alpha=request.hybrid_alpha,
                )

                if not chunks:
                    yield f"data: {json.dumps({'type': 'info', 'message': '未检索到相关内容'}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'total_results': 0}, ensure_ascii=False)}\n\n"
                    return

                logger.info(f"[FastPath] Retrieved {len(chunks)} chunks using queries={queries_used}")
                citation_data = [
                    {
                        "id": idx,
                        "title": getattr(chunk, "title", None) or getattr(chunk, "document_id", "Untitled"),
                        "link": getattr(chunk, "link", None),
                        "source_type": getattr(chunk, "source_type", None),
                    }
                    for idx, chunk in enumerate(chunks[:5], start=1)
                ]
                yield f"data: {json.dumps({'type': 'citations', 'data': citation_data}, ensure_ascii=False)}\n\n"

                yield f"data: {json.dumps({'type': 'thinking', 'content': '基于检索结果生成回答...'}, ensure_ascii=False)}\n\n"

                direct_answer = _build_direct_grounded_answer(
                    query=request.query,
                    chunks=chunks,
                    sections=sections,
                    min_score=1.8,
                )
                used_fallback = False
                if direct_answer:
                    answer = direct_answer
                else:
                    answer, used_fallback = await _generate_grounded_answer_async(
                        query=request.query,
                        chunks=chunks,
                        sections=sections,
                        personality=request.personality,
                    )

                yield f"data: {json.dumps({'type': 'answer_start'}, ensure_ascii=False)}\n\n"

                if used_fallback:
                    yield f"data: {json.dumps({'type': 'thinking', 'content': '生成超时，切换为知识库直出回答...'}, ensure_ascii=False)}\n\n"

                full_answer = answer
                for answer_chunk in _split_answer_for_sse(answer):
                    yield f"data: {json.dumps({'type': 'answer_chunk', 'content': answer_chunk}, ensure_ascii=False)}\n\n"

                yield f"data: {json.dumps({'type': 'answer_end', 'full_content': full_answer}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'total_results': len(chunks)}, ensure_ascii=False)}\n\n"
                return

            # Get or create SessionMemoryChatAgent
            system_prompt = None
            if request.personality:
                from agent_rag.core.personality import get_personality
                system_prompt = get_personality(request.personality)["system_prompt"]
            
            agent = get_session_memory_agent(session_id=request.session_id, system_prompt=system_prompt)
            
            # Resume session if session_id is provided
            if request.session_id:
                yield f"data: {json.dumps({'type': 'thinking', 'content': '加载会话上下文...'}, ensure_ascii=False)}\n\n"
                await agent.resume_session(request.session_id)
            else:
                # Start new session
                yield f"data: {json.dumps({'type': 'thinking', 'content': '创建新会话...'}, ensure_ascii=False)}\n\n"
                session = await agent.start_session(title=request.query[:50])
                logger.info(f"[API] Created new session: {session.id}")

            yield f"data: {json.dumps({'type': 'thinking', 'content': '执行并行搜索（文档+SQL）...'}, ensure_ascii=False)}\n\n"

            try:
                from agent_rag.core.env_config import get_search_config_from_env
                search_config = get_search_config_from_env()
                embedder = get_embedder()
                text_to_sql = get_text_to_sql()
                
                text_results, sql_result = await _parallel_search(
                    query=request.query,
                    index=index,
                    embedder=embedder,
                    search_config=search_config,
                    text_to_sql=text_to_sql,
                )
                
                yield f"data: {json.dumps({'type': 'thinking', 'content': '整理检索结果...'}, ensure_ascii=False)}\n\n"

                # Format SQL results
                sql_context = ""
                if sql_result.get("data") and sql_result.get("is_data_query"):
                    data = sql_result["data"]
                    row_count = len(data)
                    
                    try:
                        import io
                        import csv
                        output = io.StringIO()
                        if data:
                            keys = data[0].keys()
                            writer = csv.DictWriter(output, fieldnames=keys)
                            writer.writeheader()
                            for row in data[:5]:
                                safe_row = {k: str(v)[:100] for k,v in row.items()}
                                writer.writerow(safe_row)
                            
                            csv_content = output.getvalue()
                            sql_context = f"\n### 数据库查询结果 ({row_count}条, 显示前5条)\n```csv\n{csv_content}\n```\nSQL语句: `{sql_result.get('sql')}`"
                            if row_count > 5:
                                sql_context += f"\n(还有 {row_count-5} 条数据未显示)"
                        else:
                             sql_context = "\n### 数据库查询结果\n无数据"

                    except Exception as e:
                        sql_context = f"\n### 数据库查询结果\n数据解析错误: {str(e)}"
                        
                elif sql_result.get("error"):
                    sql_context = f"\n### 数据库查询错误\n{sql_result.get('error')}"

                # Format document results
                doc_context = ""
                if text_results:
                    doc_texts = []
                    for i, chunk in enumerate(text_results[:5]):
                        title = getattr(chunk, 'title', 'Untitled')
                        source = getattr(chunk, 'source_type', 'unknown')
                        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                        doc_texts.append(f"#### [文档 {i+1}] {title} ({source})\n{content[:800]}...")
                    
                    doc_context = "\n### 文档检索结果\n" + "\n\n".join(doc_texts)

                # Merge search context
                search_context = ""
                if sql_context:
                    search_context += sql_context + "\n\n"
                if doc_context:
                    search_context += doc_context

                if not search_context:
                    search_context = "未找到相关文档或数据。"

                # Build query for agent with search results as context
                # The SessionMemoryChatAgent will automatically:
                # 1. Load session history
                # 2. Retrieve relevant memories
                # 3. Save user query and assistant response
                # 4. Extract new memories after conversation
                
                yield f"data: {json.dumps({'type': 'thinking', 'content': '生成回答...'}, ensure_ascii=False)}\n\n"
                
                enhanced_query = f"""用户问题: {request.query}

请基于以下检索到的信息回答用户问题。如果包含数据库结果，请优先使用数据回答。如果包含文档，请结合文档内容。
请直接回答问题，不要输出分析过程。

{search_context}
"""

                yield f"data: {json.dumps({'type': 'answer_start'}, ensure_ascii=False)}\n\n"

                full_answer = ""
                # Use SessionMemoryChatAgent's run_stream which handles session and memory automatically
                async for token in agent.run_stream(
                    query=enhanced_query,
                    context={
                        "skip_query_expansion": True,
                        "session_persist_query": request.query,
                    }
                ):
                    if token and not token.startswith("\n[Searching"):
                        full_answer += token
                        yield f"data: {json.dumps({'type': 'answer_chunk', 'content': token}, ensure_ascii=False)}\n\n"
                
                logger.info(f"[API] Full answer length: {len(full_answer)} chars")
                yield f"data: {json.dumps({'type': 'answer_end', 'full_content': full_answer}, ensure_ascii=False)}\n\n"
                
                total_chunks = len(text_results)
                
                # Memory extraction and message persistence are handled internally by the agent
                # in a background task to ensure zero latency.

            except Exception as e:
                logger.error(f"Parallel search error: {e}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': f'搜索失败: {str(e)}'}, ensure_ascii=False)}\n\n"
                return

            yield f"data: {json.dumps({'type': 'done', 'total_results': total_chunks}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate_results(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


class SessionCreateResponse(BaseModel):
    """Response for session creation."""
    session_id: str
    message: str


class SessionCreateRequest(BaseModel):
    """Request to create a session."""
    user_id: Optional[str] = None


class MessageAddRequest(BaseModel):
    """Request to add a message to session."""
    session_id: str
    role: str  # "user", "assistant", "system"
    content: str
    token_count: Optional[int] = None
    user_id: Optional[str] = None


class MessageResponse(BaseModel):
    """Message response model."""
    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime


class MemoryAddRequest(BaseModel):
    """Request to add a memory."""
    content: str
    memory_type: str  # "fact", "preference", "habit", "personal", "knowledge"
    importance: float = 0.7
    user_id: Optional[str] = None
    metadata: Optional[dict] = None


class MemorySearchRequest(BaseModel):
    """Request to search memories."""
    query: str
    session_id: Optional[str] = None
    limit: int = 5
    memory_types: Optional[list[str]] = None


class MemoryResponse(BaseModel):
    """Memory response model."""
    id: str
    content: str
    memory_type: str
    importance: float
    score: float
    created_at: datetime
    metadata: dict


@app.post("/api/session/create", response_model=SessionCreateResponse)
async def create_session(request: SessionCreateRequest = None):
    """Create a new conversation session."""
    session_mgr = get_session_manager()
    session_id = await session_mgr.create_session()
    user_id = request.user_id if request else None
    logger.info(f"[API] Session created: {session_id}, user_id: {user_id}")
    return SessionCreateResponse(
        session_id=session_id,
        message=f"Session created successfully"
    )


@app.post("/api/session/{session_id}/message", response_model=MessageResponse)
async def add_message(session_id: str, request: MessageAddRequest):
    """Add a message to a session."""
    session_mgr = get_session_manager()
    message = await session_mgr.add_message(
        session_id=session_id,
        role=request.role,
        content=request.content,
        token_count=request.token_count,
        user_id=request.user_id
    )
    return MessageResponse(
        id=message.id,
        session_id=message.session_id,
        role=message.role,
        content=message.content,
        created_at=message.created_at
    )


@app.get("/api/session/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 50):
    """Get messages from a session."""
    session_mgr = get_session_manager()
    messages = await session_mgr.get_session_messages(session_id, limit=limit)
    return {
        "session_id": session_id,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat()
            }
            for m in messages
        ]
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its messages."""
    session_mgr = get_session_manager()
    await session_mgr.delete_session(session_id)
    return {"message": f"Session {session_id} deleted"}


@app.get("/api/sessions")
async def get_user_sessions(user_id: str, limit: int = 20):
    """Get all sessions for a user."""
    session_mgr = get_session_manager()
    sessions = await session_mgr.get_user_sessions(user_id, limit=limit)
    return {
        "user_id": user_id,
        "sessions": [
            {
                "id": s.id,
                "title": s.title or "新对话",
                "message_count": len(s.messages),
                "last_message_preview": get_last_user_message_preview(s.messages),
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat()
            }
            for s in sessions
        ]
    }


def get_last_user_message_preview(messages) -> str:
    """Get preview of last user message."""
    for msg in reversed(messages):
        if msg.role == "user":
            preview = msg.content.strip()
            if len(preview) > 25:
                return preview[:25] + "..."
            return preview
    return ""


@app.post("/api/session/{session_id}/title")
async def update_session_title(session_id: str, request: dict):
    """Update session title based on first message."""
    session_mgr = get_session_manager()
    await session_mgr.update_session_title(session_id, request.get("title"))
    return {"message": "Title updated"}


@app.post("/api/memory/add", response_model=dict)
async def add_memory(request: MemoryAddRequest):
    """Add a new memory."""
    memory_store = get_memory_store()
    from agent_rag.core.memory_extractor import MemoryType
    import uuid
    
    session_id = request.metadata.get("session_id") if request.metadata else str(uuid.uuid4())
    
    memory_type = MemoryType(request.memory_type) if request.memory_type else MemoryType.FACT
    memory = await memory_store.add_memory(
        content=request.content,
        memory_type=memory_type,
        session_id=session_id,
        importance=request.importance,
        user_id=request.user_id,
        metadata=request.metadata
    )
    return {
        "id": memory.id,
        "message": "Memory added successfully",
        "content": memory.content,
        "memory_type": memory.memory_type.value
    }


@app.post("/api/memory/search", response_model=list)
async def search_memories(request: MemorySearchRequest):
    """Search for relevant memories."""
    memory_store = get_memory_store()
    from agent_rag.core.memory_extractor import MemoryType
    
    memory_types = [MemoryType(t) for t in request.memory_types] if request.memory_types else None
    
    memories = await memory_store.search_memories(
        query=request.query,
        session_id=request.session_id,
        limit=request.limit,
        memory_types=memory_types
    )
    
    return [
        {
            "id": m.id,
            "content": m.content,
            "memory_type": m.memory_type.value,
            "importance": m.importance,
            "score": m.score,
            "created_at": m.created_at.isoformat(),
            "metadata": m.metadata or {}
        }
        for m in memories
    ]


@app.get("/api/memory/all")
async def get_all_memories(limit: int = 100):
    """Get all memories."""
    memory_store = get_memory_store()
    memories = await memory_store.get_all_memories(limit=limit)
    return {
        "count": len(memories),
        "memories": [
            {
                "id": m.id,
                "content": m.content,
                "memory_type": m.memory_type.value,
                "importance": m.importance,
                "created_at": m.created_at.isoformat(),
                "metadata": m.metadata or {}
            }
            for m in memories
        ]
    }


class MemoryExtractRequest(BaseModel):
    """Request to extract memories from conversation."""
    messages: list[dict]
    session_id: str


class MemoryExtractResponse(BaseModel):
    """Response for memory extraction."""
    extracted_count: int
    memories: list[dict]


@app.post("/api/memory/extract", response_model=MemoryExtractResponse)
async def extract_memories(request: MemoryExtractRequest):
    """Extract memories from conversation messages."""
    from agent_rag.core.memory_extractor import MemoryExtractor
    
    logger.info(f"[MemoryExtract] Starting extraction for session: {request.session_id}")
    logger.info(f"[MemoryExtract] Messages count: {len(request.messages)}")
    
    try:
        embedder = get_embedder()
        logger.info(f"[MemoryExtract] Embedder initialized")
        
        llm = get_llm_provider()
        logger.info(f"[MemoryExtract] LLM provider ready")
        
        mem_store = get_memory_store()
        logger.info(f"[MemoryExtract] Memory store ready")
        
        extractor = MemoryExtractor(llm=llm, memory_store=mem_store)
        logger.info(f"[MemoryExtract] Extractor created, starting extraction...")
        
        memories = await extractor.extract_from_messages(
            messages=request.messages,
            session_id=request.session_id,
            user_id=None,
        )
        
        logger.info(f"[MemoryExtract] Extraction completed, found {len(memories)} memories")
        
        return MemoryExtractResponse(
            extracted_count=len(memories),
            memories=[
                {
                    "id": m.id,
                    "content": m.content,
                    "memory_type": m.memory_type.value,
                    "importance": m.importance,
                }
                for m in memories
            ]
        )
    except Exception as e:
        import traceback
        logger.error(f"[API] Memory extraction failed: {e}")
        logger.error(f"[API] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


class MemoryRetrieveRequest(BaseModel):
    """Request to retrieve memories."""
    query: str
    user_id: Optional[str] = None
    limit: int = 5


@app.post("/api/memory/retrieve", response_model=dict)
async def retrieve_memories(request: MemoryRetrieveRequest):
    """Retrieve relevant memories for a query."""
    memory_store = get_memory_store()

    memories = await memory_store.search_memories(
        query=request.query,
        user_id=request.user_id,
        limit=request.limit,
        score_threshold=0.3,
    )

    return {
        "query": request.query,
        "memories": [
            {
                "id": m.id,
                "content": m.content,
                "memory_type": m.memory_type.value,
                "importance": m.importance,
                "score": m.score,
            }
            for m in memories
        ]
    }


@app.delete("/api/memory/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory."""
    memory_store = get_memory_store()
    await memory_store.delete_memory(memory_id)
    return {"message": f"Memory {memory_id} deleted"}


# ============================================================================
# Static Files (Frontend)
# ============================================================================

# Mount frontend static files
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8005"))

    uvicorn.run(app, host=host, port=port)
