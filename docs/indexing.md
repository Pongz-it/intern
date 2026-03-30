# 文档索引系统 (Document Indexing)

Agent RAG 索引系统负责将分块后的文档向量化并存储到 Vespa 向量数据库，支持混合搜索（语义+关键词）。

## 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                   Ingestion Workflow                         │
│            (Hatchet DAG 工作流编排)                          │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ chunk_document│    │ embed_chunks  │    │ index_chunks  │
│    _task      │───▶│    _task      │───▶│    _task      │
│  (分块任务)    │    │  (向量化任务)  │    │  (索引任务)   │
└───────────────┘    └───────────────┘    └───────────────┘
                              │                     │
                              ▼                     ▼
                    ┌───────────────┐    ┌───────────────┐
                    │   Embedder    │    │  VespaIndex   │
                    │ (OpenAI/...)  │    │  (向量数据库)  │
                    └───────────────┘    └───────────────┘
```

## 索引流程

### 完整 Pipeline

```
1. chunk_document_task
   ├── 加载配置和解析文本
   ├── 合并 OCR 文本（如有）
   ├── 选择合适的 Chunker
   ├── 生成分块（含 mini-chunks, large-chunks）
   └── 存储分块元数据
          │
          ▼
2. embed_chunks_task
   ├── 加载 Embedding 配置
   ├── 初始化 Embedder (OpenAI/自定义)
   ├── 批量向量化
   ├── 失败处理和隔离
   └── 存储向量数据
          │
          ▼
3. index_chunks_task
   ├── 加载向量化后的分块
   ├── 构建完整 Chunk 对象
   ├── 写入 VespaIndex
   ├── 更新状态为 INDEXED
   └── 失败时回滚
```

## DocumentIndex 接口

抽象基类定义了所有索引后端必须实现的方法：

```python
from abc import ABC, abstractmethod
from agent_rag.core.models import Chunk, SearchFilters

class DocumentIndex(ABC):
    @abstractmethod
    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        filters: Optional[SearchFilters] = None,
        hybrid_alpha: float = 0.5,
        num_results: int = 10,
    ) -> list[Chunk]:
        """混合搜索（语义 + 关键词）"""
        pass

    @abstractmethod
    def semantic_search(
        self,
        query_embedding: list[float],
        filters: Optional[SearchFilters] = None,
        num_results: int = 10,
    ) -> list[Chunk]:
        """纯语义搜索"""
        pass

    @abstractmethod
    def keyword_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        num_results: int = 10,
    ) -> list[Chunk]:
        """纯关键词搜索 (BM25)"""
        pass

    @abstractmethod
    def index_chunks(self, chunks: list[Chunk]) -> list[str]:
        """索引分块"""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """删除文档所有分块"""
        pass
```

## VespaIndex 实现

生产环境使用 Vespa 作为向量数据库后端。

### 配置

```python
from agent_rag.core.config import DocumentIndexConfig

config = DocumentIndexConfig(
    vespa_host="localhost",
    vespa_port=8080,
    vespa_app_name="agent_rag",
    vespa_timeout=30,
    vespa_schema_name="agent_rag_chunk",
    vespa_title_content_ratio=0.2,
    vespa_decay_factor=0.5,
)
```

环境变量：

| 环境变量 | 描述 | 默认值 |
|---------|------|--------|
| `AGENT_RAG_VESPA_HOST` | Vespa 服务地址 | localhost |
| `AGENT_RAG_VESPA_PORT` | Vespa 端口 | 8080 |
| `AGENT_RAG_VESPA_APP_NAME` | 应用名称 | agent_rag |
| `AGENT_RAG_VESPA_TIMEOUT` | 请求超时（秒） | 30 |
| `AGENT_RAG_VESPA_SCHEMA_NAME` | Schema 名称 | agent_rag_chunk |

### 搜索示例

```python
from agent_rag.document_index.vespa import VespaIndex
from agent_rag.core.models import SearchFilters

# 初始化
index = VespaIndex()

# 混合搜索
results = index.hybrid_search(
    query="如何配置认证",
    query_embedding=embedder.embed("如何配置认证"),
    filters=SearchFilters(
        source_types=["file", "url"],
        document_sets=["engineering"],
    ),
    hybrid_alpha=0.6,  # 60% 语义，40% 关键词
    num_results=10,
)

# 获取文档所有分块
chunks = index.get_chunks_by_document("doc_123")

# 获取特定分块范围
chunks = index.get_chunks_by_document("doc_123", chunk_range=(0, 5))

# 获取周围上下文
section = index.get_section(
    document_id="doc_123",
    chunk_id=5,
    expand_before=2,
    expand_after=2,
)
```

## 搜索过滤器

`SearchFilters` 支持多种过滤条件：

```python
from agent_rag.core.models import SearchFilters

filters = SearchFilters(
    # 来源类型过滤
    source_types=["file", "url", "text"],

    # 文档 ID 过滤
    document_ids=["doc_1", "doc_2"],

    # 知识库/文档集过滤
    document_sets=["engineering", "product"],

    # 用户文件夹过滤
    user_folder=123,

    # 用户项目过滤
    user_project=[1, 2, 3],

    # 时间过滤
    time_cutoff=datetime(2024, 1, 1),

    # 自定义过滤器
    custom_filters={"category": "internal"},
)
```

## Embedding 流程

### EmbeddingConfig

```python
from agent_rag.ingestion.embeddings.config import EmbeddingConfig

config = EmbeddingConfig(
    model_name="text-embedding-3-small",
    provider="openai",
    batch_size=32,
    max_retries=3,
    retry_delay=1.0,
)
```

### 失败处理

系统提供完善的失败隔离机制：

```python
from agent_rag.ingestion.embeddings.failure_handler import (
    embed_chunks_with_failure_handling,
)

result = await embed_chunks_with_failure_handling(
    chunks=chunks,
    embedder=embedder,
    config=config,
)

print(f"成功: {result.successful_chunks}/{result.total_chunks}")
print(f"成功率: {result.success_rate:.1%}")

if result.has_failures:
    summary = result.to_summary()
    print(f"失败文档: {summary['failed_documents']}")
    print(f"失败分块: {summary['failed_chunks']}")
```

特性：
- 按文档隔离失败（一个文档失败不影响其他）
- 批量处理优化
- 自动重试机制
- 标题向量缓存

## Vespa Schema

Vespa 使用的文档 schema 定义了索引字段和排名配置：

### 核心字段

```
document agent_rag_chunk {
    field document_id type string
    field chunk_id type int
    field content type string
    field title type string
    field source_type type string
    field source_links type string

    # 向量字段
    field embeddings type tensor<float>(x[1536])
    field title_embedding type tensor<float>(x[1536])

    # 增强字段
    field semantic_identifier type string
    field metadata_suffix type string
    field blurb type string
    field doc_summary type string
    field chunk_context type string

    # 访问控制字段
    field document_sets type array<string>
    field user_folder type int
    field user_project type array<int>

    # 多租户
    field tenant_id type string
}
```

### 排名配置

```
rank-profile hybrid_search_semantic_base_1536 {
    inputs {
        query(query_embedding) tensor<float>(x[1536])
        query(alpha) double: 0.5
        query(title_content_ratio) double: 0.2
        query(decay_factor) double: 0.5
    }

    first-phase {
        expression {
            query(alpha) * (
                closeness(field, embeddings) * (1 - query(title_content_ratio)) +
                closeness(field, title_embedding) * query(title_content_ratio)
            ) +
            (1 - query(alpha)) * bm25(content)
        }
    }
}
```

## 重新索引

### 触发条件

1. **强制重新索引**: `force_reindex=True`
2. **内容变更**: 相同 `document_id`，不同 `content_hash`
3. **失败重试**: `AGENT_RAG_DEDUP_REPROCESS_FAILED=true`

### 重新索引流程

```python
# 在 ingestion_workflow.py 中
if dedup_output.action in ("reindex", "update"):
    # 1. 删除旧分块
    await delete_old_chunks(
        tenant_id=input.tenant_id,
        existing_item_id=dedup_output.existing_item_id,
        existing_document_id=dedup_output.existing_document_id,
    )

    # 2. 继续正常索引流程
    ...
```

### 相关环境变量

| 环境变量 | 描述 | 默认值 |
|---------|------|--------|
| `AGENT_RAG_DEDUP_REPROCESS_FAILED` | 重试失败文档 | false |
| `AGENT_RAG_DEDUP_CROSS_TENANT` | 跨租户去重 | false |

## 状态管理

索引过程中的状态变化：

```
PENDING → PROCESSING → INDEXED
                    ↘ FAILED
                    ↘ FAILED_PARTIAL (部分索引后失败)
                    ↘ DUPLICATE (重复内容)
```

状态更新点：
- `fetch_content_task`: PENDING → PROCESSING
- `index_chunks_task`: PROCESSING → INDEXED / FAILED
- `dedup_check_task`: → DUPLICATE (如果是重复)

## 目录结构

```
agent_rag/
├── document_index/
│   ├── __init__.py
│   ├── interface.py       # DocumentIndex 抽象接口
│   ├── memory/            # 内存索引（测试用）
│   │   └── memory_index.py
│   └── vespa/
│       ├── __init__.py
│       ├── vespa_index.py # Vespa 实现
│       ├── schema_config.py
│       └── app_config/    # Vespa 应用配置
│
├── ingestion/
│   ├── embeddings/
│   │   ├── config.py      # Embedding 配置
│   │   ├── embedder.py    # 索引专用 Embedder
│   │   └── failure_handler.py  # 失败处理
│   │
│   └── workflow/
│       └── tasks/
│           └── indexing_tasks.py  # 索引任务
│
└── embedding/
    ├── interface.py       # Embedder 接口
    └── providers/
        ├── openai.py      # OpenAI Embedder
        └── ...
```

## 初始化 Vespa

使用提供的脚本初始化 Vespa：

```bash
# 部署 Vespa schema
agent-rag-vespa --host localhost --port 19071

# 或使用 Python
python -m agent_rag.scripts.deploy_vespa
```

详见 [初始化指南](./initialization.md)。

## 最佳实践

### 批量索引

```python
# 使用批量 API 提升性能
chunks = [...]  # 大量分块
batch_size = 100

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    index.index_chunks(batch)
```

### 租户隔离

```python
# 确保每个分块都有 tenant_id
for chunk in chunks:
    chunk.tenant_id = "tenant_123"

# 搜索时自动过滤
results = index.hybrid_search(
    query="...",
    query_embedding=...,
    filters=SearchFilters(
        custom_filters={"tenant_id": "tenant_123"}
    ),
)
```

### 错误处理

```python
try:
    indexed_ids = index.index_chunks(chunks)
except Exception as e:
    # 回滚已索引的部分
    for chunk in chunks:
        try:
            index.delete_chunk(chunk.document_id, chunk.chunk_id)
        except:
            pass
    raise
```
