# Agent RAG 安装指南

完整的依赖安装和环境配置说明。

## 快速安装

### 1. 基础安装（仅 Agent 功能）

```bash
cd agent_rag
pip install -e .
```

**包含的依赖：**
- `litellm` - LLM 调用
- `httpx` - HTTP 客户端
- `pydantic` - 数据验证
- `tiktoken` - Token 计数
- `numpy` - 数组操作
- `jinja2` - 模板引擎

### 2. 完整安装（所有功能）

```bash
pip install -e ".[all]"
```

这会安装所有可选依赖组：database + vespa + ingestion + dev

---

## 按需安装

### Database（数据库支持）

**用途：** PostgreSQL 元数据存储、摄取状态跟踪

```bash
pip install -e ".[database]"
```

**包含：**
- `sqlalchemy[asyncio]>=2.0.0` - ORM 和异步数据库
- `asyncpg>=0.29.0` - PostgreSQL 异步驱动
- `alembic>=1.13.0` - 数据库迁移
- `greenlet>=3.0.0` - 异步支持

**初始化数据库：**
```bash
# 创建数据库表
agent-rag-init

# 运行迁移
agent-rag-migrate
```

### Vespa（向量索引）

**用途：** 混合搜索（语义 + 关键词）

```bash
pip install -e ".[vespa]"
```

**包含：**
- `pyvespa>=0.45.0` - Vespa Python 客户端

**部署 Vespa Schema：**
```bash
agent-rag-vespa
```

### Parsing（文档解析）

**用途：** PDF、DOCX、XLSX、PPTX 文件解析

```bash
pip install -e ".[parsing]"
```

**包含：**
- `pypdf>=3.0.0` - PDF 文本提取
- `pdfplumber>=0.10.0` - PDF 表格提取
- `python-docx>=1.0.0` - Word 文档解析
- `openpyxl>=3.1.0` - Excel 解析
- `python-pptx>=0.6.0` - PowerPoint 解析
- `unstructured-client>=0.20.0` - Unstructured API 回退

**支持的文件类型：**
- PDF (.pdf)
- Word (.docx)
- Excel (.xlsx)
- PowerPoint (.pptx)

### Chunking（文档分块）

**用途：** 语义感知的文档分块

```bash
pip install -e ".[chunking]"
```

**包含：**
- `chonkie>=0.1.0` - 语义分块（SentenceChunker）

**特性：**
- 语义边界感知
- Mini-chunks（多遍索引）
- Large chunks（上下文扩展）
- Contextual RAG 支持

### API（FastAPI 服务）

**用途：** 摄取 API 端点

```bash
pip install -e ".[api]"
```

**包含：**
- `fastapi>=0.100.0` - Web 框架
- `uvicorn[standard]>=0.23.0` - ASGI 服务器
- `python-multipart>=0.0.6` - 文件上传

**启动 API 服务：**
```bash
uvicorn agent_rag.ingestion.api.ingestion_api:app --reload
```

### Ingestion（完整摄取管道）

**用途：** 文档摄取、Hatchet DAG 编排、MinIO 存储

```bash
pip install -e ".[ingestion]"
```

**包含：**
- `database` 组的所有依赖
- `parsing` 组的所有依赖
- `chunking` 组的所有依赖
- `api` 组的所有依赖
- `minio>=7.2.0` - 对象存储客户端
- `hatchet-sdk>=0.30.0` - 工作流编排

**环境变量：**
```bash
# PostgreSQL
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost/agent_rag"

# MinIO
export MINIO_ENDPOINT="localhost:9000"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin"
export MINIO_BUCKET="agent-rag"

# Hatchet
export HATCHET_API_KEY="your-hatchet-key"
export HATCHET_ADDRESS="localhost:7077"
```

### Dev（开发工具）

**用途：** 测试、代码质量、类型检查

```bash
pip install -e ".[dev]"
```

**包含：**
- `pytest>=7.0.0` - 测试框架
- `pytest-asyncio>=0.21.0` - 异步测试
- `pytest-cov>=4.0.0` - 测试覆盖率
- `black>=23.0.0` - 代码格式化
- `ruff>=0.1.0` - 快速 linter
- `mypy>=1.0.0` - 类型检查

**运行测试：**
```bash
pytest tests/
pytest tests/ --cov=agent_rag --cov-report=html
```

**代码质量检查：**
```bash
black agent_rag/
ruff check agent_rag/
mypy agent_rag/
```

---

## 组合安装示例

### 仅 Agent + Vespa（无摄取）

```bash
pip install -e ".[vespa]"
```

适用场景：使用预索引数据，只需查询功能

### Agent + 摄取（无 Vespa，使用 MemoryIndex）

```bash
pip install -e ".[ingestion]"
```

适用场景：开发测试，使用内存索引

### 生产环境完整安装

```bash
pip install -e ".[all]"
```

适用场景：完整功能部署

---

## 依赖关系图

```
all
├── database
│   ├── sqlalchemy[asyncio]
│   ├── asyncpg
│   ├── alembic
│   └── greenlet
├── vespa
│   └── pyvespa
├── ingestion
│   ├── database (递归)
│   ├── parsing
│   │   ├── pypdf
│   │   ├── pdfplumber
│   │   ├── python-docx
│   │   ├── openpyxl
│   │   ├── python-pptx
│   │   └── unstructured-client
│   ├── chunking
│   │   └── chonkie
│   ├── api
│   │   ├── fastapi
│   │   ├── uvicorn
│   │   └── python-multipart
│   ├── minio
│   └── hatchet-sdk
└── dev
    ├── pytest
    ├── pytest-asyncio
    ├── pytest-cov
    ├── black
    ├── ruff
    └── mypy
```

---

## 可选外部服务

### PostgreSQL

**安装：**
```bash
# macOS (Homebrew)
brew install postgresql@16
brew services start postgresql@16

# Docker
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  postgres:16
```

**创建数据库：**
```bash
createdb agent_rag
```

### Vespa

**Docker 部署：**
```bash
docker run -d \
  --name vespa \
  -p 8080:8080 \
  -p 19071:19071 \
  vespaengine/vespa:latest
```

**部署 Schema：**
```bash
agent-rag-vespa
```

### MinIO

**Docker 部署：**
```bash
docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

**创建 Bucket：**
```bash
# 使用 mc (MinIO Client)
mc alias set local http://localhost:9000 minioadmin minioadmin
mc mb local/agent-rag
```

### Hatchet

**Cloud 版本：**
```bash
# 注册并获取 API Key
# https://cloud.onhatchet.run
export HATCHET_API_KEY="your-key"
```

**Self-hosted：**
```bash
# 参考 Hatchet 官方文档
# https://docs.hatchet.run/self-hosting
```

---

## 验证安装

### 1. 检查已安装的包

```bash
pip list | grep -E "litellm|pydantic|sqlalchemy|pyvespa|minio|hatchet|pypdf|chonkie|fastapi"
```

### 2. 运行测试

```bash
pip install -e ".[dev]"
pytest tests/test_core.py -v
```

### 3. 验证命令行工具

```bash
agent-rag-init --help
agent-rag-migrate --help
agent-rag-vespa --help
```

---

## 常见问题

### Q1: 安装 chonkie 失败

**问题：** `chonkie` 包未找到

**解决：**
```bash
# 确认 PyPI 中有 chonkie 包，或使用替代实现
pip install chonkie  # 如果包存在

# 或修改代码使用其他分块器
```

### Q2: Unstructured Client 安装问题

**问题：** `unstructured-client` 依赖冲突

**解决：**
```bash
# 仅在需要 Unstructured API 回退时安装
pip install unstructured-client

# 或设置环境变量禁用回退
export UNSTRUCTURED_API_KEY=""
```

### Q3: PostgreSQL 连接失败

**问题：** `asyncpg.exceptions.InvalidCatalogNameError`

**解决：**
```bash
# 创建数据库
createdb agent_rag

# 或修改连接字符串
export DATABASE_URL="postgresql+asyncpg://localhost/postgres"
```

### Q4: Vespa 部署失败

**问题：** `pyvespa.exceptions.VespaError`

**解决：**
```bash
# 检查 Vespa 是否运行
curl http://localhost:8080/state/v1/health

# 重启 Vespa
docker restart vespa

# 等待 Vespa 完全启动（约 30 秒）
sleep 30
agent-rag-vespa
```

---

## 升级依赖

```bash
# 升级到最新版本
pip install -e ".[all]" --upgrade

# 升级特定包
pip install --upgrade litellm pydantic sqlalchemy

# 查看过时的包
pip list --outdated
```

---

## 卸载

```bash
pip uninstall agent-rag -y
```

---

## 下一步

安装完成后，参考以下文档继续：

- [getting-started.md](./docs/getting-started.md) - 快速开始
- [configuration.md](./docs/configuration.md) - 配置说明
- [architecture.md](./docs/architecture.md) - 架构设计
- [GUIDELINE.md](./docs/GUIDELINE.md) - 文档导航
