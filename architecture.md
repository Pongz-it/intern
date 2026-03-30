# Architecture Design

Agent RAG system architecture and component design.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Agent Layer                                     │
│  ┌─────────────────┐                    ┌─────────────────────────────┐     │
│  │   ChatAgent     │                    │   DeepResearchOrchestrator  │     │
│  │  (Conversational)│                    │   (Multi-step Research)     │     │
│  └────────┬────────┘                    └──────────────┬──────────────┘     │
│           │                                            │                     │
│           └──────────────────┬─────────────────────────┘                     │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Tool System                                    │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌─────────────┐                      │  │
│  │  │SearchTool│  │WebSearchTool │  │ OpenURLTool │  ...                 │  │
│  │  └────┬─────┘  └──────┬───────┘  └──────┬──────┘                      │  │
│  └───────┼───────────────┼─────────────────┼─────────────────────────────┘  │
└──────────┼───────────────┼─────────────────┼────────────────────────────────┘
           │               │                 │
           ▼               ▼                 ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Document Index  │ │   Web APIs   │ │   URL Fetcher    │
│  (Vespa/Memory)  │ │  (Tavily)    │ │  (HTTP Client)   │
└────────┬─────────┘ └──────────────┘ └──────────────────┘
         │
         │  ◄──── Indexed Chunks
         │
┌────────┴───────────────────────────────────────────────────────────────────┐
│                          Ingestion Pipeline                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Hatchet DAG Workflow                              │   │
│  │                                                                      │   │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐         │   │
│  │   │ Receive │───►│  Parse  │───►│  Chunk  │───►│  Embed  │──┐      │   │
│  │   │Document │    │   +OCR  │    │         │    │         │  │      │   │
│  │   └─────────┘    └─────────┘    └─────────┘    └─────────┘  │      │   │
│  │        │                                                     │      │   │
│  │        ▼                                                     ▼      │   │
│  │   ┌─────────┐                                          ┌─────────┐ │   │
│  │   │  Dedup  │                                          │  Index  │ │   │
│  │   │  Check  │                                          │ (Vespa) │ │   │
│  │   └─────────┘                                          └─────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                   │
│  │  PostgreSQL   │  │     MinIO     │  │   Hatchet     │                   │
│  │  (Metadata)   │  │   (Storage)   │  │  (Orchestrate)│                   │
│  └───────────────┘  └───────────────┘  └───────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agent Layer

#### ChatAgent
Single-turn and multi-turn conversational RAG.

```
User Query → [LLM] → Tool Calls? ─Yes→ Execute Tools → [LLM] → Response
                            │
                            └─No→ Direct Response
```

**Flow:**
1. Receive user query
2. LLM decides: answer directly or call tools
3. If tools needed: execute and collect results
4. Generate final response with citations
5. Repeat until done or max_cycles reached

#### DeepResearchOrchestrator
Multi-step research with parallel agent execution.

```
Question → Clarification? → Sub-Questions → Parallel Agents → Synthesis → Report
              │                    │                │              │
              ▼                    ▼                ▼              ▼
         [Optional]          [Planning]      [Execution]     [Generation]
```

**Phases:**
1. **Clarification** (optional): Ask user for context
2. **Planning**: Generate sub-questions and research plan
3. **Execution**: Spawn parallel research agents
4. **Synthesis**: Collect and merge findings
5. **Report**: Generate comprehensive report

### 2. Tool System

#### Architecture
```
ToolRegistry ──► ToolRunner ──► Tool.execute()
     │                              │
     │                              ▼
     │                       ToolResponse
     ▼
Tool Definitions (for LLM)
```

#### Built-in Tools

| Tool | Purpose | Backend |
|------|---------|---------|
| SearchTool | Internal document search | DocumentIndex |
| WebSearchTool | Web search | Tavily API |
| OpenURLTool | URL content extraction | HTTP/HTML |

#### Custom Tool Example
```python
from agent_rag.tools.interface import Tool, ToolResponse

class MyTool(Tool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "Description for LLM"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "..."}
            },
            "required": ["query"]
        }

    def execute(self, query: str) -> ToolResponse:
        result = do_something(query)
        return ToolResponse(success=True, data=result)
```

### 3. Document Index

#### Interface
```python
class DocumentIndex(Protocol):
    def index(chunks) -> IndexingResult
    def search(query, filters, embedding, hybrid_alpha, limit) -> list[Chunk]
    def get_chunk(document_id, chunk_id) -> Optional[Chunk]
    def id_based_retrieval(chunk_requests) -> list[Chunk]
    def delete_document(document_id) -> bool
```

#### Implementations

**MemoryIndex**
- In-memory vector search
- FAISS-based similarity
- Good for development/testing

**VespaIndex**
- Production-grade search
- Hybrid (semantic + keyword)
- Multi-tenant support

**EnhancedVespaIndex**
- Parallel indexing
- Visit API for bulk operations
- Chunk cleanup
- Knowledge graph fields

### 4. Retrieval Pipeline

```
Query → Query Expansion → Multi-Query Search → RRF Fusion → Section Selection → Context Expansion
           │                    │                  │               │                    │
           ▼                    ▼                  ▼               ▼                    ▼
      [Semantic]           [Parallel]         [Ranking]       [LLM Filter]         [Expand]
      [Keyword]             Search             Merge           Relevance           Sections
```

#### Query Expansion
1. **Semantic Query**: Natural language rephrasing
2. **Keyword Query**: Extract key terms

#### Fusion Strategies
- **RRF**: Reciprocal Rank Fusion with weights
- **Linear**: Weighted score combination

#### Section Expansion
1. Retrieve center chunks
2. LLM classifies relevance level
3. Expand to adjacent chunks or full document

### 5. Citation System

```
Chunks → Citation Map → LLM Response → Citation Processor → Renumbered Output
                             │                  │
                             │                  ▼
                             │           Used Citations
                             ▼
                      [1], [2], [3] in text
```

**Components:**
- `DynamicCitationProcessor`: Stream processing
- `GlobalCitationAccumulator`: Cross-agent citation tracking
- Citation utilities for formatting

### 6. LLM Integration

```
LLMConfig → LLM Provider → LLM Interface
                │
                ▼
         LiteLLMProvider
                │
                ▼
    OpenAI / Anthropic / Azure / etc.
```

**Features:**
- Streaming support
- Tool calling
- Reasoning model support (o1, etc.)
- Configurable timeouts and retries

---

## Data Flow

### Search Query Flow

```
1. User Query
       │
       ▼
2. Query Expansion (optional)
   ├── Semantic Query
   └── Keyword Queries
       │
       ▼
3. Parallel Search
   ├── Semantic Search (embedding)
   └── Keyword Search (BM25)
       │
       ▼
4. Result Fusion (RRF)
       │
       ▼
5. Section Building
       │
       ▼
6. LLM Selection (optional)
       │
       ▼
7. Context Expansion
       │
       ▼
8. Final Sections for LLM
```

### Indexing Flow (Legacy)

> **注意**: 完整的摄取流程请参见下方 [Ingestion Pipeline](#7-ingestion-pipeline) 章节。

```
1. Documents
       │
       ▼
2. Chunking
   ├── Content chunks
   ├── Title embedding
   └── KG extraction (optional)
       │
       ▼
3. Embedding Generation
       │
       ▼
4. Index to Vespa
   ├── Batch operations
   └── Parallel threads
       │
       ▼
5. Indexing Result
```

---

## 7. Ingestion Pipeline

完整的文档摄取管道，使用 Hatchet DAG 编排多阶段任务。

> 详细文档: [parsing.md](./parsing.md) | [chunking.md](./chunking.md) | [indexing.md](./indexing.md)

### 架构概述

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Ingestion API (FastAPI)                          │
│  POST /ingest/file  POST /ingest/url  POST /ingest/text  POST /ingest/batch │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      Hatchet DAG Workflow                                 │
│                                                                           │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │  Task 1: fetch-content                                          │     │
│   │  - 保存原始文件到 MinIO                                          │     │
│   │  - 创建 IngestionItem 记录                                       │     │
│   └──────────────────────┬─────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │  Task 2: dedup-check                                            │     │
│   │  - 计算内容哈希 (SHA-256)                                        │     │
│   │  - 检测重复文档                                                   │     │
│   └──────────────────────┬─────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │  Task 3: parse-document                                         │     │
│   │  - 文本提取 (pypdf, python-docx, openpyxl, python-pptx)         │     │
│   │  - 结构保留 (标题、段落、表格)                                     │     │
│   │  - Unstructured API 回退                                         │     │
│   └──────────────────────┬─────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │  Task 3.5: extract-images                                       │     │
│   │  - 从解析结果中提取图片                                            │     │
│   │  - 保存到 MinIO                                                  │     │
│   └──────────────────────┬─────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │  Task 4: ocr-images  (条件执行: 仅当有图片时)                     │     │
│   │  - 图片文字识别 (Tesseract/Google Vision/AWS Textract/LLM)       │     │
│   │  - OCR 结果合并到文本                                             │     │
│   └──────────────────────┬─────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │  Task 5: chunk-document                                         │     │
│   │  - 语义分块 (chonkie SentenceChunker)                            │     │
│   │  - Mini-chunks (多遍索引)                                        │     │
│   │  - Large chunks (上下文扩展)                                     │     │
│   │  - Contextual RAG (文档摘要 + 块上下文)                          │     │
│   └──────────────────────┬─────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │  Task 6: embed-chunks                                           │     │
│   │  - Title embedding 缓存                                          │     │
│   │  - 批量嵌入处理                                                   │     │
│   │  - 失败块隔离 (不阻塞成功块)                                       │     │
│   └──────────────────────┬─────────────────────────────────────────┘     │
│                          │                                                │
│                          ▼                                                │
│   ┌────────────────────────────────────────────────────────────────┐     │
│   │  Task 7: index-chunks                                           │     │
│   │  - Vespa 批量写入 (hybrid search: BM25 + 向量)                   │     │
│   │  - 事务回滚支持                                                   │     │
│   │  - 租户隔离索引                                                   │     │
│   └────────────────────────────────────────────────────────────────┘     │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 存储架构

| 组件 | 用途 | 数据类型 |
|------|------|----------|
| **PostgreSQL** | 摄取状态跟踪 | IngestionItem, IngestionBatch |
| **MinIO** | 原始文件存储 | 上传文件, 解析产物, OCR结果 |
| **Vespa** | 向量索引 | Chunks with embeddings |

### 数据模型

#### IngestionItem（完整字段）

```python
class IngestionItem(Base):
    """摄取任务跟踪表 - 完整定义"""
    __tablename__ = "ingestion_items"

    # === 主键 ===
    id: UUID                          # 任务唯一标识

    # === 租户隔离 ===
    tenant_id: str                    # 租户 ID（索引）

    # === 来源信息 ===
    source_type: SourceType           # FILE, URL, TEXT, SLACK, etc.
    source_uri: str                   # 原始来源 URI
    file_name: str                    # 原始文件名（最大 512 字符）
    mime_type: str | None             # MIME 类型（如 application/pdf）
    size_bytes: int                   # 文件大小（字节）

    # === 去重与存储 ===
    content_hash: str                 # SHA-256 内容哈希（索引）
    content_ref: str | None           # MinIO 原始文件路径
                                      # 格式: raw/{tenant_id}/{item_id}/{filename}
    parsed_ref: str | None            # MinIO 解析结果路径
                                      # 格式: parsed/{tenant_id}/{item_id}/text.md

    # === 处理状态 ===
    status: IngestionStatus           # 当前状态（索引）
    error: str | None                 # 错误信息（失败时）
    retry_count: int                  # 重试次数（默认 0）
    last_attempt_at: datetime | None  # 最后尝试时间

    # === 索引关联 ===
    document_id: str | None           # Vespa 文档 ID（索引）
    chunk_count: int                  # 生成的文本分块数
    image_count: int                  # 提取的图片数
    table_count: int                  # 提取的表格数

    # === 元数据与回调 ===
    metadata_json: dict               # JSONB 扩展元数据
                                      # 示例: {"title": "...", "author": "..."}
    webhook_url: str | None           # 完成/失败回调 URL

    # === 时间戳 ===
    created_at: datetime              # 创建时间
    updated_at: datetime              # 更新时间（自动更新）
    completed_at: datetime | None     # 完成时间

    # === 数据库索引 ===
    # Index: (tenant_id, status) - 查询租户下各状态任务
    # Index: (tenant_id, content_hash) - 去重检查
    # Index: (document_id) - 反向查找摄取任务
```

#### IngestionStatus 枚举

```python
class IngestionStatus(Enum):
    """摄取任务状态流转"""
    PENDING = "pending"               # 等待处理
    PARSING = "parsing"               # 解析中
    CHUNKING = "chunking"             # 分块中
    EMBEDDING = "embedding"           # 嵌入生成中
    INDEXING = "indexing"             # 索引写入中
    INDEXED = "indexed"               # 已完成索引
    FAILED = "failed"                 # 完全失败
    FAILED_PARTIAL = "failed_partial" # 部分失败（部分 chunks 索引成功）
```

#### 状态流转图

```
PENDING → PARSING → CHUNKING → EMBEDDING → INDEXING → INDEXED
    ↓         ↓          ↓           ↓           ↓
  FAILED   FAILED    FAILED      FAILED      FAILED
                                              ↓
                                      FAILED_PARTIAL
                                    (部分 chunks 成功)
```

### OCR 子系统

```
┌─────────────────────────────────────────────────────────┐
│                    OCR Registry                          │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Tesseract  │  │   Google    │  │     AWS     │     │
│  │    OCR      │  │   Vision    │  │   Textract  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                          │
│  ┌─────────────┐                                        │
│  │   LLM OCR   │  ← 使用 Vision Model (GPT-4V, etc.)    │
│  └─────────────┘                                        │
└─────────────────────────────────────────────────────────┘
```

### API 端点

> **说明**: 所有端点前缀为 `/api/v1/ingestion`

#### 摄取端点

```
POST /api/v1/ingestion/ingest/file
  - 单文件摄取
  - 支持本地文件和 URL 引用

POST /api/v1/ingestion/ingest/url
  - URL 内容抓取和摄取
  - 支持网页、API 响应等

POST /api/v1/ingestion/ingest/text
  - 直接摄取文本内容
  - 支持 Markdown、纯文本

POST /api/v1/ingestion/ingest/batch
  - 批量摄取多个文件或 URL
  - 支持混合来源类型
```

#### 状态查询

```
GET /api/v1/ingestion/status/{item_id}
  - 查询单个摄取任务状态
  - 返回详细的处理进度

GET /api/v1/ingestion/stats
  - 获取摄取统计信息
  - 按租户、状态聚合数据
```

#### 管理端点

> **注意**: 删除端点暂未实现。删除文档需通过 DocumentIndex 接口直接操作。

```
# DELETE /api/v1/ingestion/documents/{document_id}
#   - 删除已索引文档及相关 chunks
#   - 同时清理 MinIO 存储和数据库记录
#   状态: 计划中，尚未实现
```

### 配置

> **说明**: Agent RAG 使用环境变量进行配置，无需编程式配置类。

#### 环境变量配置

```bash
# === 数据库配置 ===
DATABASE_URL="postgresql+asyncpg://user:pass@localhost/agent_rag"

# === MinIO 存储配置 ===
MINIO_ENDPOINT="localhost:9000"
MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"
MINIO_BUCKET="agent-rag"
MINIO_SECURE=false  # 使用 HTTPS: true, HTTP: false

# === Hatchet 工作流配置 ===
HATCHET_API_KEY="your-hatchet-api-key"
HATCHET_ADDRESS="localhost:7077"  # Hatchet 服务器地址

# === Vespa 索引配置 ===
VESPA_APP_URL="http://localhost:8080"
VESPA_SCHEMA_NAME="agent_rag_chunk"

# === 摄取处理配置 ===
MAX_FILE_SIZE=100000000  # 最大文件大小（字节），默认 100MB

# === 分块配置 ===
CHUNK_SIZE=512            # 目标分块大小（tokens）
CHUNK_OVERLAP=50          # 分块重叠（tokens）
MIN_CHUNK_SIZE=128        # 最小分块大小
ENABLE_MINI_CHUNKS=true   # 启用 mini-chunks（多遍索引）
ENABLE_LARGE_CHUNKS=true  # 启用 large chunks（上下文扩展）

# === 嵌入配置 ===
EMBEDDING_MODEL="text-embedding-3-small"
EMBEDDING_DIMENSION=1536
EMBEDDING_BATCH_SIZE=128
EMBEDDING_API_KEY="your-openai-api-key"  # 如使用 OpenAI

# === OCR 配置 ===
OCR_PROVIDER="tesseract"  # tesseract | google_vision | aws_textract | llm
OCR_ENABLED=true
# Google Vision (if using)
# GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
# AWS Textract (if using)
# AWS_ACCESS_KEY_ID="..."
# AWS_SECRET_ACCESS_KEY="..."

# === Unstructured API（回退选项）===
UNSTRUCTURED_API_KEY=""  # 留空则不使用回退
UNSTRUCTURED_API_URL="https://api.unstructured.io/general/v0/general"
```

#### 配置加载

配置通过 `agent_rag.core.env_config` 模块加载：

```python
from agent_rag.core.env_config import load_dotenv, get_config

# 加载 .env 文件
load_dotenv()

# 获取完整配置
config = get_config()
```

详细配置说明参见：[INSTALLATION.md](../INSTALLATION.md) 和 [configuration.md](./configuration.md)

---

## Configuration Hierarchy

```
AgentRAGConfig
├── LLMConfig
│   ├── model, provider
│   ├── api_key, api_base
│   └── max_tokens, temperature
├── EmbeddingConfig
│   ├── model, dimension
│   └── batch_size
├── DocumentIndexConfig
│   ├── type (memory/vespa)
│   └── vespa settings
└── AgentConfig
    ├── mode (chat/deep_research)
    ├── max_cycles
    ├── SearchConfig
    │   ├── hybrid_alpha
    │   ├── query expansion
    │   └── context expansion
    └── DeepResearchConfig
        ├── orchestrator cycles
        └── research agents
```

---

## Vespa Schema Design

### Core Fields
```
document agent_rag_chunk {
    field document_id: string
    field chunk_id: int
    field content: string
    field title: string
    field embeddings: tensor<float>(x[1536])
}
```

### Optional Features

**Title Embedding**
```
field title_embedding: tensor<float>(x[1536])
```

**Knowledge Graph**
```
field kg_entities: array<string>
field kg_relationships: array<kg_relationship>
field kg_terms: array<string>
```

**Multi-Tenant**
```
field tenant_id: string
```

**Access Control**
```
field access_control_list: weightedset<string>
```

### Ranking Profile
```
rank-profile hybrid {
    inputs {
        query(query_embedding): tensor<float>(x[1536])
        query(alpha): float
    }
    first-phase {
        expression: query(alpha) * closeness(field, embeddings) +
                   (1 - query(alpha)) * bm25(content)
    }
}
```

---

## Extension Points

### Custom Tools
Implement `Tool` protocol and register with `ToolRegistry`.

### Custom Document Index
Implement `DocumentIndex` protocol.

### Custom LLM Provider
Implement `LLM` protocol.

### Custom Embedder
Implement `Embedder` protocol.

### MCP Integration
Use `MCPProvider` for Model Context Protocol tools.

---

## Performance Considerations

### Indexing
- Batch size: 128 chunks
- Parallel threads: 32
- Retry with exponential backoff

### Search
- Limit: 10-20 results
- Hybrid alpha: 0.5 (balanced)
- RRF k: 50-60

### Agent
- Max cycles: 6 for chat
- Max orchestrator cycles: 8 for deep research
- Streaming for responsiveness

### Memory
- Chunk cleanup after retrieval
- Citation deduplication
- Session-based caching
