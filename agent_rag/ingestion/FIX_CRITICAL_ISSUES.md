# 关键问题修复清单

## 问题 1: 解析器未注册 ✅ 已修复

**问题**: docx/pdf/xlsx 解析器实现存在但未在 registry.py 中注册

**修复**:
- 文件: `/agent_rag/ingestion/parsing/registry.py`
- 已添加导入和注册逻辑

```python
from agent_rag.ingestion.parsing.parsers.docx_parser import DOCXParser
from agent_rag.ingestion.parsing.parsers.pdf_parser import PDFParser
from agent_rag.ingestion.parsing.parsers.xlsx_parser import XLSXParser

def _register_builtin_parsers(self):
    ...
    self.register(PDFParser())
    self.register(DOCXParser())
    self.register(XLSXParser())
```

---

## 问题 2: 字段命名不一致 (Critical)

### 2.1 IngestionItem 字段不匹配

**错误使用** (在 tasks 中):
- `item.filename` → 应为 `item.file_name`
- `item.metadata_` → 应为 `item.metadata_json`
- `item.parsed_text_path` → 应为 `item.parsed_ref`
- `item.error_message` → 应为 `item.error`
- `item.image_count` → ✅ 正确 (models.py:119)
- `item.table_count` → ❌ **字段不存在!**

**models.py 实际定义**:
```python
file_name = Column(String(512), nullable=False)  # NOT filename
metadata_json = Column(JSONB, ...)                # NOT metadata_
parsed_ref = Column(Text, nullable=True)          # NOT parsed_text_path
error = Column(Text, nullable=True)               # NOT error_message
chunk_count = Column(Integer, ...)                # EXISTS
```

**修复清单**:
1. `ingestion_tasks.py:148` - `filename` → `file_name`
2. `ingestion_tasks.py:150` - `metadata_` → `metadata_json`
3. `ingestion_tasks.py:352-354` - `parsed_text_path/image_count/table_count` 字段修复
4. `ingestion_tasks.py:382` - `error_message` → `error`
5. `indexing_tasks.py:140` - `item.metadata_` → `item.metadata_json`
6. `indexing_tasks.py:192` - `item.metadata_["chunks"]` → `item.metadata_json.setdefault("chunks", [])`
7. `indexing_tasks.py:249` - `item.metadata_.get("chunks")` → `item.metadata_json.get("chunks")`
8. `indexing_tasks.py:259/296/305` - 所有 `item.metadata_` 替换为 `item.metadata_json`

### 2.2 ParsedImage 字段不匹配

**错误使用** (ingestion_tasks.py:334-340):
```python
if image.image_content:  # ❌ 应为 image.content
    image_content=image.image_content,  # ❌
    extension=image.extension or "png",  # ❌ 应从 mime_type 提取
```

**models.py 实际定义** (base.py:8-23):
```python
@dataclass
class ParsedImage:
    content: bytes        # NOT image_content
    mime_type: str        # NOT extension
```

**修复**:
```python
if image.content:
    extension = image.mime_type.split("/")[-1] if "/" in image.mime_type else "png"
    await storage.store_image(
        image_content=image.content,
        extension=extension,
    )
```

---

## 问题 3: Indexing 任务是 stub

**文件**: `indexing_tasks.py:328-411`

**当前代码** (line 377-379):
```python
# For now, log success (actual indexing would happen here)
logger.info(f"Would index {len(indexed_chunks_data)} chunks to {input.index_name}")
```

**修复方案**:
```python
# Get document index
doc_index: DocumentIndex = VespaIndex()

# Reconstruct Chunk objects with embeddings
from agent_rag.core.models import Chunk

chunks_to_index = []
for chunk_data in indexed_chunks_data:
    chunk = Chunk(
        document_id=chunk_data["document_id"],
        chunk_id=chunk_data["chunk_id"],
        content="",  # Content not needed for indexing (already embedded)
        embedding=chunk_data["full_embedding"],
        # ... additional fields
    )
    chunks_to_index.append(chunk)

# Actually index chunks
await doc_index.index_chunks(chunks_to_index, index_name=input.index_name)

logger.info(f"Chunks indexed successfully: {len(chunks_to_index)} chunks to {input.index_name}")
```

---

## 问题 4: OCR 不是 LLM OCR

**设计要求**: LLM OCR + 可插拔接口

**当前实现**: 只有 tesseract/google_vision/aws_textract

**修复方案**: 创建 LLM OCR Provider

### 4.1 创建 LLM OCR Provider 接口

```python
# File: agent_rag/ingestion/ocr/providers/llm_ocr.py

from agent_rag.llm.interface import LLMProvider

class LLMOCRProvider(BaseOCRProvider):
    """LLM-based OCR using vision models (GPT-4V, Claude 3, Gemini Pro Vision)."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def extract_text(self, image_data: bytes, mime_type: str) -> OCRResult:
        # Encode image to base64
        import base64
        image_b64 = base64.b64encode(image_data).decode()

        # Call LLM with vision capabilities
        prompt = "Extract all text from this image. Return only the text content, maintaining original layout."

        response = await self.llm.generate(
            prompt=prompt,
            images=[{"data": image_b64, "mime_type": mime_type}]
        )

        return OCRResult(
            text=response.content,
            confidence=0.95,  # LLM OCR typically high quality
            provider="llm_ocr",
        )
```

### 4.2 更新 OCR Provider Registry

```python
# File: agent_rag/ingestion/ocr/providers/__init__.py

from agent_rag.ingestion.ocr.providers.llm_ocr import LLMOCRProvider
from agent_rag.llm.providers.openai import OpenAIProvider

def get_ocr_provider(provider_name: str) -> BaseOCRProvider:
    if provider_name == "llm_ocr":
        llm = OpenAIProvider(model="gpt-4-vision-preview")
        return LLMOCRProvider(llm)
    elif provider_name == "tesseract":
        return TesseractOCRProvider()
    # ... existing providers
```

### 4.3 更新 .env 配置

```bash
# LLM OCR 配置
AGENT_RAG_OCR_PROVIDER=llm_ocr  # tesseract, google_vision, aws_textract, llm_ocr
AGENT_RAG_LLM_OCR_MODEL=gpt-4-vision-preview  # 或 claude-3-opus-20240229
```

---

## 问题 5: URL 抓取缺少安全策略

**文件**: `ingestion_tasks.py:401-408`

**当前代码**:
```python
async def _fetch_url_content(url: str) -> bytes:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.read()
```

**安全问题**:
1. ❌ 无 SSRF 防护 (可访问内网)
2. ❌ 无限流
3. ❌ 无大小限制
4. ❌ 无超时控制
5. ❌ 无 User-Agent

**修复方案**:

```python
import ipaddress
from urllib.parse import urlparse

async def _fetch_url_content(url: str) -> bytes:
    """
    Fetch content from URL with security controls.

    Security measures:
    - SSRF protection: Block private/local IPs
    - Size limit: Max 100MB
    - Timeout: 30 seconds
    - Rate limiting: Handled by Hatchet workflow rate_limits
    """
    import aiohttp

    # 1. SSRF Protection: Validate URL
    parsed_url = urlparse(url)

    # Block non-HTTP(S) schemes
    if parsed_url.scheme not in ["http", "https"]:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}")

    # Resolve hostname to IP
    import socket
    try:
        ip = socket.gethostbyname(parsed_url.hostname)
        ip_obj = ipaddress.ip_address(ip)

        # Block private/internal IPs
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
            raise ValueError(f"SSRF attempt detected: {url} resolves to private IP {ip}")
    except socket.gaierror:
        raise ValueError(f"Cannot resolve hostname: {parsed_url.hostname}")

    # 2. Fetch with limits
    timeout = aiohttp.ClientTimeout(total=env_config.url_fetch_timeout)  # 30s from .env
    max_size = env_config.max_document_chars  # 10MB from .env

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(
            url,
            headers={"User-Agent": env_config.url_user_agent},  # From .env
        ) as response:
            response.raise_for_status()

            # 3. Check Content-Length before reading
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_size:
                raise ValueError(f"Content too large: {content_length} bytes (max {max_size})")

            # 4. Read with size limit
            content = bytearray()
            async for chunk in response.content.iter_chunked(8192):
                content.extend(chunk)
                if len(content) > max_size:
                    raise ValueError(f"Content exceeds {max_size} bytes")

            return bytes(content)
```

---

## 问题 6: chunk 存储到 metadata_ 字段

**文件**: `indexing_tasks.py:192`

**错误代码**:
```python
item.metadata_["chunks"] = chunks_data  # ❌ metadata_ 不存在
```

**修复**:
```python
# metadata_json 是 JSONB 类型，需要使用 setdefault 或直接赋值
if item.metadata_json is None:
    item.metadata_json = {}
item.metadata_json["chunks"] = chunks_data

# 或使用 SQLAlchemy 的 flag_modified
from sqlalchemy.orm.attributes import flag_modified
item.metadata_json = {**(item.metadata_json or {}), "chunks": chunks_data}
flag_modified(item, "metadata_json")
```

---

## 问题 7: IngestionItem 缺少 table_count 字段

**当前使用** (ingestion_tasks.py:354):
```python
item.table_count = len(parsed_doc.tables)  # ❌ 字段不存在
```

**models.py 定义**:
```python
chunk_count = Column(Integer, nullable=True, default=0)  # ✅ EXISTS
# table_count 不存在！
```

**修复方案**:

### 方案 A: 添加字段到 models.py (推荐)
```python
class IngestionItem(Base):
    ...
    chunk_count = Column(Integer, nullable=True, default=0)
    image_count = Column(Integer, nullable=True, default=0)  # 已存在但未显示
    table_count = Column(Integer, nullable=True, default=0)  # 新增
```

### 方案 B: 存储到 metadata_json
```python
# ingestion_tasks.py:354
if item.metadata_json is None:
    item.metadata_json = {}
item.metadata_json["table_count"] = len(parsed_doc.tables)
item.metadata_json["link_count"] = len(parsed_doc.links)
```

---

## 修复优先级

### P0 - Critical (阻塞运行)
1. ✅ **解析器注册** - 已修复
2. **字段命名不一致** - 必须修复，否则运行时错误
3. **ParsedImage 字段错误** - 必须修复
4. **metadata_ 字段错误** - 必须修复

### P1 - Important (功能不完整)
5. **Indexing stub** - 索引不落地
6. **URL 安全** - 安全风险

### P2 - Enhancement (设计改进)
7. **LLM OCR** - 设计文档要求但非阻塞

---

## 下一步行动

1. 修复所有 P0 问题 (字段命名)
2. 实现 index_chunks 真正写入逻辑
3. 添加 URL 安全验证
4. 可选: 添加 LLM OCR provider

