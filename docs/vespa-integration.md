# Vespa Integration Guide

Vespa 文档索引的完整集成指南。

## Overview

Agent RAG 提供两种 Vespa 索引实现：

| 实现 | 用途 | 特性 |
|------|------|------|
| `VespaIndex` | 基础使用 | 混合搜索、基本 CRUD |
| `EnhancedVespaIndex` | 生产环境 | 并行索引、Visit API、知识图谱、Chunk 清理 |

---

## Quick Start

### 基础使用

```python
from agent_rag.document_index.vespa import VespaIndex
from agent_rag.core.models import Chunk

# 创建索引
index = VespaIndex(
    host="localhost",
    port=8080,
    schema_name="agent_rag_chunk",
)

# 索引文档
chunks = [
    Chunk(
        document_id="doc_1",
        chunk_id=0,
        content="Agent RAG 是一个独立的 RAG 组件...",
        title="Agent RAG 介绍",
        source_type="confluence",
    )
]
indexed_ids = index.index_chunks(chunks)

# 混合搜索
results = index.hybrid_search(
    query="RAG 组件",
    query_embedding=embedding_vector,
    hybrid_alpha=0.5,
    num_results=10,
)
```

### 生产环境

```python
from agent_rag.document_index.vespa import EnhancedVespaIndex
from agent_rag.document_index.vespa.schema_config import VespaSchemaConfig

# 创建配置
schema_config = VespaSchemaConfig(
    dim=1536,
    enable_title_embedding=True,
    enable_knowledge_graph=True,
    multi_tenant=True,
)

# 创建增强索引
index = EnhancedVespaIndex(
    host="localhost",
    port=8080,
    schema_config=schema_config,
    num_threads=32,
)

# 并行索引
result = index.index_chunks_parallel(chunks, batch_size=128)
print(f"Indexed: {len(result.indexed_ids)}, Failed: {len(result.failed_ids)}")
```

---

## Schema Configuration

### 预设配置

```python
from agent_rag.document_index.vespa.schema_config import get_schema_preset

# 可用预设: minimal, standard, enterprise
config = get_schema_preset("standard")
```

| 预设 | 特性 |
|------|------|
| `minimal` | 基础功能，无标题嵌入、无知识图谱 |
| `standard` | 标题嵌入、大块引用，适合大多数场景 |
| `enterprise` | 多租户、知识图谱、访问控制、高 rerank |

### 自定义配置

```python
from agent_rag.document_index.vespa.schema_config import VespaSchemaConfig

config = VespaSchemaConfig(
    # Schema 标识
    schema_name="my_chunk",

    # Embedding 配置
    dim=1536,                          # 嵌入维度
    embedding_precision="float",       # float 或 bfloat16

    # 功能开关
    enable_title_embedding=True,       # 标题嵌入
    enable_large_chunks=True,          # 大块引用
    enable_knowledge_graph=True,       # 知识图谱字段
    multi_tenant=True,                 # 多租户支持
    enable_access_control=False,       # 访问控制

    # 排序配置
    default_decay_factor=0.5,          # 时间衰减因子
    rerank_count=2000,                 # 全局重排数量

    # 服务配置
    redundancy=1,                      # 副本数
    searchable_copies=1,               # 可搜索副本
    search_threads=4,                  # 搜索线程
    summary_threads=2,                 # 摘要线程
)
```

### 生成 Application Package

```python
from pathlib import Path
from agent_rag.document_index.vespa.schema_config import (
    VespaSchemaConfig,
    VespaSchemaRenderer,
)

# 创建配置
config = VespaSchemaConfig(
    schema_name="my_chunk",
    dim=1536,
    enable_knowledge_graph=True,
)

# 生成 schema 文件
renderer = VespaSchemaRenderer()
files = renderer.generate_application_package(
    config=config,
    output_dir=Path("./vespa-app"),
)

print(f"Schema: {files['schema']}")
print(f"Services: {files['services']}")
```

生成的目录结构：

```
vespa-app/
├── schemas/
│   └── my_chunk.sd
└── services.xml
```

---

## Search Operations

### 混合搜索

结合语义搜索和关键词搜索：

```python
results = index.hybrid_search(
    query="machine learning",           # 关键词查询
    query_embedding=embedding_vector,   # 语义嵌入
    hybrid_alpha=0.5,                   # 0=纯关键词, 1=纯语义
    num_results=10,
    include_highlights=True,            # 包含高亮
    tenant_id="tenant_123",             # 多租户过滤
)

for chunk in results:
    print(f"Score: {chunk.score}")
    print(f"Content: {chunk.content[:200]}")
    print(f"Highlights: {chunk.match_highlights}")
```

### 语义搜索

纯向量相似度搜索：

```python
results = index.semantic_search(
    query_embedding=embedding_vector,
    num_results=10,
)
```

### 关键词搜索

纯 BM25 关键词搜索：

```python
results = index.keyword_search(
    query="RAG architecture",
    num_results=10,
)
```

### 管理员搜索

包含隐藏文档的搜索：

```python
results = index.admin_search(
    query="internal document",
    num_results=10,
    include_hidden=True,
)
```

### 搜索过滤

```python
from agent_rag.core.models import SearchFilters
from datetime import datetime, timedelta

filters = SearchFilters(
    source_types=["confluence", "github"],
    document_ids=["doc_1", "doc_2"],
    tags=["important", "reviewed"],
    time_cutoff=datetime.now() - timedelta(days=30),
)

results = index.hybrid_search(
    query="search term",
    query_embedding=embedding,
    filters=filters,
)
```

---

## Document Operations

### 按文档获取 Chunks

```python
# 获取文档的所有 chunks
chunks = index.get_chunks_by_document("doc_123")

# 获取特定范围
chunks = index.get_chunks_by_document(
    document_id="doc_123",
    chunk_range=(0, 10),  # chunk_id 0-9
)
```

### ID-Based Retrieval

批量获取指定 ID 范围的 chunks：

```python
from agent_rag.document_index.interface import ChunkRequest

requests = [
    ChunkRequest(document_id="doc_1", min_chunk_id=0, max_chunk_id=5),
    ChunkRequest(document_id="doc_2", min_chunk_id=3, max_chunk_id=8),
]

chunks = index.id_based_retrieval(requests)
```

### 获取单个 Chunk

```python
chunk = index.get_chunk(document_id="doc_123", chunk_id=0)
if chunk:
    print(chunk.content)
```

### Doc ID 清理

Vespa 对 document ID 有字符限制，使用清理函数：

```python
from agent_rag.document_index.vespa.vespa_index import replace_invalid_doc_id_characters

# 清理无效字符
clean_id = replace_invalid_doc_id_characters("doc/with:special@chars")
# 结果: "doc_with_special_chars"
```

---

## Indexing

### 基础索引

```python
chunks = [
    Chunk(
        document_id="doc_1",
        chunk_id=0,
        content="...",
        title="Title",
        embedding=[0.1, 0.2, ...],
    )
]

indexed_ids = index.index_chunks(chunks)
```

### 并行索引 (EnhancedVespaIndex)

```python
from agent_rag.document_index.vespa import EnhancedVespaIndex

index = EnhancedVespaIndex(num_threads=32)

result = index.index_chunks_parallel(chunks, batch_size=128)

print(f"成功: {len(result.indexed_ids)}")
print(f"失败: {len(result.failed_ids)}")
for doc_id, error in result.error_messages.items():
    print(f"  {doc_id}: {error}")
```

特性：
- 自动重试（指数退避）
- 速率限制处理
- 并行 ThreadPool 执行

---

## Visit API

大规模文档遍历，适合数据迁移或批量处理：

### 分页遍历

```python
# 获取一批文档
result = index.visit_documents(
    selection='source_type == "confluence"',
    fields_to_include=["document_id", "content", "title"],
    wanted_document_count=1000,
)

print(f"获取 {len(result.chunks)} 个 chunks")
print(f"继续 token: {result.continuation_token}")

# 获取下一批
if result.continuation_token:
    next_result = index.visit_documents(
        continuation_token=result.continuation_token,
        wanted_document_count=1000,
    )
```

### 生成器遍历

```python
# 遍历所有文档
for chunk in index.visit_all_documents(batch_size=1000):
    process(chunk)

# 带过滤条件
for chunk in index.visit_all_documents(
    selection='source_type == "github"',
    fields_to_include=["document_id", "content"],
):
    process(chunk)
```

---

## Chunk Cleanup

从 Vespa 检索的 chunks 包含索引时添加的增强内容，需要清理后返回给用户：

### 增强内容说明

索引时添加的内容：
1. **Title Prefix**: 标题前置到 content 以提高搜索匹配
2. **Metadata Suffix**: 元数据后缀
3. **Doc Summary**: 文档摘要（Contextual RAG）
4. **Chunk Context**: 块上下文（Contextual RAG）

### 清理使用

```python
# 搜索获取原始结果
raw_results = index.hybrid_search(query, embedding)

# 清理增强内容
clean_results = index.cleanup_chunks(raw_results)

# 返回给用户
for chunk in clean_results:
    print(chunk.content)  # 原始文档内容
```

### 清理逻辑

```python
# 内部清理过程
def cleanup_chunks(chunks):
    for chunk in chunks:
        content = chunk.content

        # 1. 移除标题前缀
        if chunk.title and content.startswith(chunk.title):
            content = content[len(chunk.title):].lstrip()

        # 2. 移除元数据后缀
        if chunk.metadata_suffix and content.endswith(chunk.metadata_suffix):
            content = content.removesuffix(chunk.metadata_suffix)

        # 3. 移除 Contextual RAG 添加
        if chunk.doc_summary and content.startswith(chunk.doc_summary):
            content = content[len(chunk.doc_summary):].lstrip()
        if chunk.chunk_context and content.endswith(chunk.chunk_context):
            content = content[:-len(chunk.chunk_context)].rstrip()

        yield cleaned_chunk
```

---

## Knowledge Graph Fields

### 更新 KG 字段

```python
from agent_rag.core.models import KGRelationship

# 更新实体和关系
success = index.update_knowledge_graph_fields(
    document_id="doc_123",
    chunk_id=0,
    kg_entities=["Entity1", "Entity2", "Entity3"],
    kg_relationships=[
        KGRelationship(source="Entity1", rel_type="relates_to", target="Entity2"),
        KGRelationship(source="Entity2", rel_type="contains", target="Entity3"),
    ],
    kg_terms=["term1", "term2"],
)
```

### 索引时包含 KG 字段

```python
chunk = Chunk(
    document_id="doc_1",
    chunk_id=0,
    content="...",
    kg_entities=["Python", "Machine Learning"],
    kg_relationships=[
        KGRelationship(source="Python", rel_type="used_for", target="Machine Learning")
    ],
    kg_terms=["programming", "AI"],
)

index.index_chunks([chunk])
```

---

## Update Operations

### 更新 Boost

```python
# 提高文档相关性
success = index.update_chunk_boost(
    document_id="doc_123",
    chunk_id=0,
    boost=2.0,  # 默认 0.0
)
```

### 更新隐藏状态

```python
# 隐藏文档
success = index.update_chunk_hidden(
    document_id="doc_123",
    chunk_id=0,
    hidden=True,
)
```

---

## Delete Operations

### 删除文档

```python
# 删除文档及其所有 chunks
success = index.delete_document("doc_123")
```

### 删除单个 Chunk

```python
success = index.delete_chunk(
    document_id="doc_123",
    chunk_id=0,
)
```

---

## Multi-Tenant Support

### 配置多租户

```python
config = VespaSchemaConfig(
    multi_tenant=True,
)

index = EnhancedVespaIndex(schema_config=config)
```

### 租户隔离搜索

```python
# 搜索特定租户的文档
results = index.hybrid_search(
    query="search term",
    query_embedding=embedding,
    tenant_id="tenant_123",
)
```

### 索引时指定租户

```python
chunk = Chunk(
    document_id="doc_1",
    chunk_id=0,
    content="...",
    tenant_id="tenant_123",
)

index.index_chunks([chunk])
```

---

## Ranking Profiles

可用的排序 profile：

| Profile | 说明 |
|---------|------|
| `hybrid_search_semantic_base_{dim}` | 混合搜索（语义为主） |
| `hybrid_search_keyword_base_{dim}` | 混合搜索（关键词为主） |
| `bm25_only` | 纯 BM25 关键词 |
| `admin_search` | 管理员搜索（标题加权） |
| `random_rank` | 随机排序 |

### 自定义排序参数

```python
body = index._build_yql_query(
    query="search",
    query_embedding=embedding,
    hybrid_alpha=0.7,              # 语义权重
    title_content_ratio=0.2,       # 标题/内容比例
    decay_factor=0.5,              # 时间衰减
)
```

---

## Environment Configuration

通过环境变量配置 Vespa：

```bash
# Vespa 连接
AGENT_RAG_VESPA_HOST=localhost
AGENT_RAG_VESPA_PORT=8080
AGENT_RAG_VESPA_APP_NAME=agent_rag
AGENT_RAG_VESPA_TIMEOUT=30

# Schema 配置
AGENT_RAG_VESPA_SCHEMA_NAME=agent_rag_chunk
AGENT_RAG_VESPA_SCHEMA_DIM=1536
AGENT_RAG_VESPA_ENABLE_TITLE_EMBEDDING=true
AGENT_RAG_VESPA_ENABLE_KNOWLEDGE_GRAPH=true
AGENT_RAG_VESPA_MULTI_TENANT=false

# 排序配置
AGENT_RAG_VESPA_TITLE_CONTENT_RATIO=0.2
AGENT_RAG_VESPA_DEFAULT_DECAY_FACTOR=0.5
AGENT_RAG_VESPA_DECAY_FACTOR=0.5
```

加载配置：

```python
from agent_rag.core.env_config import (
    get_document_index_config_from_env,
    get_vespa_schema_config_from_env,
)

index_config = get_document_index_config_from_env()
schema_config = get_vespa_schema_config_from_env()

index = EnhancedVespaIndex(
    config=index_config,
    schema_config=schema_config,
)
```

---

## Error Handling

```python
from agent_rag.core.exceptions import DocumentIndexError

try:
    results = index.hybrid_search(query, embedding)
except DocumentIndexError as e:
    print(f"搜索失败: {e}")
    print(f"索引类型: {e.index_type}")
```

---

## Best Practices

### 1. 连接管理

```python
# 使用 context manager 或确保关闭
index = EnhancedVespaIndex()
try:
    # 操作...
    pass
finally:
    index.close()
```

### 2. 批量索引

```python
# 使用并行索引处理大量数据
result = index.index_chunks_parallel(
    chunks,
    batch_size=128,  # 调整批次大小
)

# 处理失败
if result.failed_ids:
    for doc_id in result.failed_ids:
        logger.error(f"索引失败: {doc_id}: {result.error_messages.get(doc_id)}")
```

### 3. 搜索优化

```python
# 使用过滤减少搜索范围
filters = SearchFilters(
    source_types=["relevant_type"],
    time_cutoff=recent_date,
)

# 适当设置 hybrid_alpha
# 0.3-0.5: 平衡模式
# 0.7-1.0: 语义为主（适合概念搜索）
# 0.0-0.3: 关键词为主（适合精确匹配）
```

### 4. 内存管理

```python
# 大规模遍历时使用生成器
for chunk in index.visit_all_documents(batch_size=500):
    process(chunk)
    # 不要在内存中累积所有结果
```

---

## Troubleshooting

### 连接问题

```python
# 检查 Vespa 是否可访问
import httpx

try:
    response = httpx.get("http://localhost:8080/state/v1/health")
    print(f"Vespa 状态: {response.status_code}")
except httpx.ConnectError:
    print("无法连接到 Vespa")
```

### 索引问题

```python
# 检查 schema 是否正确部署
response = httpx.get("http://localhost:8080/application/v2/tenant/default/application/default")
print(response.json())
```

### 搜索无结果

1. 检查文档是否已索引
2. 检查 embedding 维度是否匹配
3. 检查过滤条件是否过严
4. 检查 tenant_id 是否正确

```python
# 获取文档数量
result = index.visit_documents(wanted_document_count=1)
print(f"总文档数: {result.total_count}")
```
