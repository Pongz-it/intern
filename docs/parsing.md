# 文档解析系统 (Document Parsing)

Agent RAG 文档解析系统提供可扩展的文档解析能力，支持多种文件格式和自动解析器选择。

## 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                      ParserRegistry                         │
│  (自动解析器选择，基于 source_type/extension/mime_type)      │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  PDFParser    │    │  DOCXParser   │    │  XLSXParser   │
│  (priority=0) │    │  (priority=0) │    │  (priority=0) │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌───────────────────┐
                    │  PlainTextParser  │
                    │  (priority=-100)  │
                    │     (fallback)    │
                    └───────────────────┘
```

## 核心组件

### ParsedDocument

解析结果的标准数据结构：

```python
@dataclass
class ParsedDocument:
    text: str                           # 标准化文本内容
    metadata: dict[str, Any]            # 文档元数据
    images: list[ParsedImage]           # 提取的图片
    links: list[str]                    # 提取的链接
    tables: list[dict[str, Any]]        # 提取的表格
```

### ParsedImage

图片数据结构：

```python
@dataclass
class ParsedImage:
    image_id: str                       # 图片唯一标识
    content: bytes                      # 原始图片字节
    mime_type: str                      # MIME 类型 (image/png, image/jpeg)
    page_number: Optional[int]          # 所在页码
    caption: Optional[str]              # 图片说明/alt文本
```

## 内置解析器

| 解析器 | 支持格式 | 优先级 | 描述 |
|-------|---------|--------|------|
| `PDFParser` | .pdf | 0 | PDF 文档解析，支持图片提取 |
| `DOCXParser` | .docx | 0 | Word 文档解析 |
| `PPTXParser` | .pptx | 0 | PowerPoint 演示文稿解析 |
| `XLSXParser` | .xlsx, .xls, .csv | 0 | Excel 表格解析 |
| `MarkdownParser` | .md, .markdown | 0 | Markdown 文件解析，提取链接 |
| `URLParser` | source_type=url | 0 | 网页内容解析 (使用 trafilatura) |
| `PlainTextParser` | .txt, fallback | -100 | 纯文本解析，作为最终回退 |

## 使用方式

### 基本使用

```python
from agent_rag.ingestion.parsing.registry import get_parser_registry

# 获取全局注册表实例
registry = get_parser_registry()

# 自动选择解析器并解析
document = registry.parse(
    content=file_bytes,
    filename="document.pdf",
    source_type="file",
    mime_type="application/pdf"
)

print(f"文本长度: {len(document.text)}")
print(f"图片数量: {len(document.images)}")
print(f"表格数量: {len(document.tables)}")
```

### 手动选择解析器

```python
from agent_rag.ingestion.parsing.registry import get_parser_registry

registry = get_parser_registry()

# 获取特定解析器
parser = registry.get_parser(
    source_type="file",
    extension="pdf",
    mime_type="application/pdf"
)

# 直接使用解析器
document = parser.parse(file_bytes, "document.pdf")
```

### 注册自定义解析器

```python
from agent_rag.ingestion.parsing.base import BaseParser, ParsedDocument
from agent_rag.ingestion.parsing.registry import register_parser

class CustomParser(BaseParser):
    def supports(self, source_type: str, extension: str, mime_type: str) -> bool:
        return extension.lower() == "custom"

    def parse(self, content: bytes, filename: str) -> ParsedDocument:
        text = content.decode("utf-8")
        return ParsedDocument(
            text=text,
            metadata={"parser": "custom", "filename": filename}
        )

    @property
    def priority(self) -> int:
        return 10  # 高优先级

# 注册到全局注册表
register_parser(CustomParser())
```

## Unstructured API 回退

当没有找到合适的解析器或解析失败时，系统会自动回退到 Unstructured API：

```bash
# 环境变量配置
export AGENT_RAG_UNSTRUCTURED_API_KEY="your-api-key"
export AGENT_RAG_UNSTRUCTURED_API_URL="https://api.unstructured.io/general/v0/general"
```

回退逻辑：
1. 优先使用内置解析器
2. 如果解析失败且配置了 Unstructured API，使用 API 解析
3. 如果都失败，抛出异常

## 文本标准化

所有解析器输出的文本都经过标准化处理：

- 移除控制字符（保留 `\n` 和 `\t`）
- 合并重复空白字符
- 减少过多的空行
- 提取并存储结构化元数据

## 解析器优先级

解析器按优先级从高到低排序：
- 优先级相同时，后注册的优先
- `PlainTextParser` 优先级最低 (-100)，作为通用回退

推荐优先级范围：`-100` 到 `100`

## 目录结构

```
agent_rag/ingestion/parsing/
├── __init__.py
├── base.py              # 基础接口和数据模型
├── registry.py          # 解析器注册表
├── unstructured.py      # Unstructured API 集成
├── utils.py             # 文本标准化工具
└── parsers/
    ├── __init__.py
    ├── pdf_parser.py    # PDF 解析器
    ├── docx_parser.py   # DOCX 解析器
    ├── pptx_parser.py   # PPTX 解析器
    └── xlsx_parser.py   # XLSX 解析器
```

## 扩展指南

### 实现新解析器

1. 继承 `BaseParser` 基类
2. 实现 `supports()` 方法判断是否支持文件类型
3. 实现 `parse()` 方法返回 `ParsedDocument`
4. 可选：覆盖 `priority` 属性设置优先级
5. 使用 `register_parser()` 注册到全局注册表

### 最佳实践

- 解析器应该是无状态的
- 解析失败时抛出明确的异常
- 提取尽可能多的结构化元数据
- 图片应该转换为常见格式 (PNG/JPEG)
- 大文件考虑流式处理
