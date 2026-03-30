# 文档分块系统 (Document Chunking)

Agent RAG 分块系统提供语义感知的文档分块能力，支持多种高级特性如 mini-chunks、large chunks 和 Contextual RAG。

## 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                    SemanticChunker                          │
│          (基于 chonkie SentenceChunker 实现)                 │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Text Chunks  │    │ Image Chunks  │    │ Large Chunks  │
│   (常规块)     │    │   (图片块)     │    │   (组合块)     │
└───────────────┘    └───────────────┘    └───────────────┘
        │
        ▼
┌───────────────────┐
│   Mini-Chunks     │
│  (子块/多遍索引)   │
└───────────────────┘
```

## 核心功能

### 优先级功能 [P0]

| 功能 | 描述 | 配置 |
|-----|------|------|
| SentenceChunker | 语义感知的句子级分割 | 使用 chonkie 库 |
| Image Chunks | 为图片创建独立分块 | `create_image_chunks=True` |
| Mini-Chunks | 多遍索引的子分块 | `enable_multipass=True` |
| Large Chunks | 合并多个分块为大块 | `enable_large_chunks=True` |

### 优先级功能 [P1]

| 功能 | 描述 | 配置 |
|-----|------|------|
| Dual Metadata Suffix | 语义+关键词双后缀 | `include_metadata=True` |
| Contextual RAG | 文档摘要+分块上下文 | `enable_contextual_rag=True` |
| Strict Token Limit | 严格的 token 限制 | `strict_chunk_token_limit=True` |

### 优先级功能 [P2]

| 功能 | 描述 | 配置 |
|-----|------|------|
| Section Continuation | 跟踪分块是否跨越章节 | `track_section_continuation=True` |
| Source Link Mapping | 链接偏移量映射 | `preserve_source_links=True` |

## 配置参数

### 基础参数

```python
from agent_rag.ingestion.chunking.config import ChunkingConfig

config = ChunkingConfig(
    # 基础分块参数
    chunk_token_limit=512,      # 每块最大 token 数
    chunk_overlap=0,            # 相邻块的重叠 token
    blurb_size=128,             # 预览文本大小
    chunk_min_content=256,      # 最小内容 token 数
    strict_chunk_token_limit=True,  # 严格执行 token 限制
)
```

### Multipass 索引

```python
config = ChunkingConfig(
    enable_multipass=True,      # 启用 mini-chunk 生成
    mini_chunk_size=64,         # mini-chunk 大小 (tokens)
)
```

Mini-chunks 用于多遍检索：
1. 第一遍：使用 mini-chunks 进行快速语义匹配
2. 第二遍：返回完整的父分块

### Large Chunks

```python
config = ChunkingConfig(
    enable_large_chunks=True,   # 启用大块生成
    large_chunk_ratio=4,        # 4 个常规块合并为 1 个大块
)
```

Large chunks 用于：
- 提供更广泛的上下文
- 支持需要长文本的 LLM
- 减少上下文碎片化

### Contextual RAG

```python
config = ChunkingConfig(
    enable_contextual_rag=True,     # 启用上下文增强
    use_doc_summary=True,           # 包含文档摘要
    use_chunk_context=True,         # 包含分块上下文
    max_context_tokens=512,         # 上下文预留 token 数
    contextual_rag_llm_name="gpt-4",
    contextual_rag_llm_provider="openai",
)
```

Contextual RAG 为每个分块添加：
- `doc_summary`: 文档级摘要
- `chunk_context`: 分块在文档中的位置描述

## 使用方式

### 基本使用

```python
from agent_rag.ingestion.chunking.chunker import SemanticChunker
from agent_rag.ingestion.chunking.config import ChunkingConfig
from agent_rag.ingestion.parsing.base import ParsedDocument

# 创建分块器
chunker = SemanticChunker()

# 配置
config = ChunkingConfig(
    chunk_token_limit=512,
    enable_multipass=True,
    enable_large_chunks=True,
)

# 解析后的文档
document = ParsedDocument(
    text="文档内容...",
    metadata={"title": "示例文档"}
)

# 生成分块
chunks = chunker.chunk(document, ingestion_item, config)

for chunk in chunks:
    print(f"Chunk {chunk.chunk_id}: {len(chunk.content)} chars")
    if chunk.large_chunk_reference_ids:
        print(f"  -> 包含分块: {chunk.large_chunk_reference_ids}")
```

### 从环境变量配置

```python
from agent_rag.ingestion.chunking.config import ChunkingConfig

# 从环境变量创建配置
config = ChunkingConfig.from_env()
```

支持的环境变量：

| 环境变量 | 描述 | 默认值 |
|---------|------|--------|
| `AGENT_RAG_CHUNK_TOKEN_LIMIT` | 最大 token 数 | 512 |
| `AGENT_RAG_CHUNK_OVERLAP` | 重叠 token 数 | 0 |
| `AGENT_RAG_BLURB_SIZE` | 预览大小 | 128 |
| `AGENT_RAG_CHUNK_MIN_CONTENT` | 最小内容 token | 256 |
| `AGENT_RAG_STRICT_CHUNK_TOKEN_LIMIT` | 严格限制 | true |
| `AGENT_RAG_ENABLE_MULTIPASS` | 启用多遍索引 | false |
| `AGENT_RAG_MINI_CHUNK_SIZE` | Mini-chunk 大小 | 64 |
| `AGENT_RAG_ENABLE_LARGE_CHUNKS` | 启用大块 | false |
| `AGENT_RAG_LARGE_CHUNK_RATIO` | 大块合并比例 | 4 |
| `AGENT_RAG_ENABLE_CONTEXTUAL_RAG` | 启用上下文 RAG | false |
| `AGENT_RAG_USE_DOC_SUMMARY` | 使用文档摘要 | true |
| `AGENT_RAG_USE_CHUNK_CONTEXT` | 使用分块上下文 | true |
| `AGENT_RAG_MAX_CONTEXT_TOKENS` | 上下文 token 预留 | 512 |

## Chunk 数据结构

分块后生成的 `Chunk` 对象包含：

```python
@dataclass
class Chunk:
    # 核心字段
    document_id: str              # 文档 ID
    chunk_id: int                 # 分块序号
    content: str                  # 分块内容
    embedding: list[float]        # 向量嵌入 (后续生成)

    # 元数据
    title: str                    # 文档标题
    source_type: str              # 来源类型
    link: str                     # 来源链接
    metadata: dict                # 额外元数据

    # 增强字段
    semantic_identifier: str      # 语义标识符
    metadata_suffix: str          # 元数据后缀
    blurb: str                    # 预览文本
    section_continuation: bool    # 是否跨章节

    # Multi-embedding 支持
    title_embedding: list[float]  # 标题向量
    embeddings: dict              # 多向量映射

    # Large chunk 支持
    large_chunk_reference_ids: list[int]  # 包含的分块 ID

    # Contextual RAG
    doc_summary: str              # 文档摘要
    chunk_context: str            # 分块上下文
```

## 分块流程

```
1. 验证配置参数
      │
      ▼
2. 构建 title_prefix 和 metadata_suffix
      │
      ▼
3. 计算有效内容 token 限制
      │
      ▼
4. 使用 SentenceChunker 分割文本
      │
      ├──[P0]──▶ 5. 处理图片，生成 image chunks
      │
      ├──[P0]──▶ 6. 生成 mini-chunks (multipass)
      │
      ├──[P0]──▶ 7. 生成 large chunks
      │
      └──[P1]──▶ 8. 添加 contextual RAG 内容
              │
              ▼
      9. 转换为最终 Chunk 模型
```

## 依赖库

- **chonkie**: 语义感知的句子分割
- **tiktoken**: OpenAI 的 tokenizer (GPT-4/cl100k_base)

如果 chonkie 不可用，系统会回退到简单的句子分割。

## Token 计算

系统使用 `tiktoken` 的 `cl100k_base` 编码（GPT-4 使用的编码）进行 token 计数：

```python
from agent_rag.ingestion.chunking.base import count_tokens, truncate_to_tokens

# 计算 token 数
tokens = count_tokens("Hello, world!")

# 截断到指定 token 数
truncated = truncate_to_tokens(long_text, max_tokens=256)
```

如果 tiktoken 不可用，会使用 `word_count * 1.3` 近似计算。

## 目录结构

```
agent_rag/ingestion/chunking/
├── __init__.py
├── base.py          # 基础接口和工具函数
├── config.py        # 配置参数定义
├── chunker.py       # SemanticChunker 实现
└── registry.py      # 分块器注册表 (可选)
```

## 最佳实践

### 通用文档

```python
config = ChunkingConfig(
    chunk_token_limit=512,
    chunk_overlap=50,
    enable_multipass=True,
)
```

### 长文档 / 技术文档

```python
config = ChunkingConfig(
    chunk_token_limit=768,
    enable_large_chunks=True,
    large_chunk_ratio=3,
    enable_contextual_rag=True,
)
```

### 精确检索场景

```python
config = ChunkingConfig(
    chunk_token_limit=256,
    enable_multipass=True,
    mini_chunk_size=48,
    strict_chunk_token_limit=True,
)
```
