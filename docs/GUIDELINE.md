# Agent RAG 文档指南 (Documentation Guideline)

本指南帮助您快速定位所需文档，了解各文档的内容范围和适用场景。

## 文档概览

```
docs/
├── README.md                    # 入口文档 - 项目概述和快速开始
├── GUIDELINE.md                 # 本文档 - 文档导航指南
│
├── 🚀 入门指南
│   ├── getting-started.md       # 安装、配置、快速上手
│   └── configuration.md         # 环境变量和配置详解
│
├── 🏗️ 架构设计
│   ├── architecture.md          # 系统架构和组件设计
│   └── ingestion-indexing-design.md  # 摄取索引设计文档
│
├── 📚 功能模块
│   ├── deep-research.md         # Deep Research 深度研究功能
│   ├── vespa-integration.md     # Vespa 向量数据库集成
│   ├── parsing.md               # 文档解析系统
│   ├── chunking.md              # 文档分块系统
│   └── indexing.md              # 文档索引系统
│
└── 📖 参考手册
    └── api-reference.md         # API 完整参考
```

## 按角色选择文档

### 👨‍💻 开发者 - 快速集成

**推荐阅读顺序：**
1. [README.md](./README.md) - 了解项目概况
2. [getting-started.md](./getting-started.md) - 安装和基础使用
3. [api-reference.md](./api-reference.md) - API 使用详情

### 🏗️ 架构师 - 系统设计

**推荐阅读顺序：**
1. [architecture.md](./architecture.md) - 整体架构
2. [ingestion-indexing-design.md](./ingestion-indexing-design.md) - 摄取索引设计
3. [vespa-integration.md](./vespa-integration.md) - 向量数据库集成

### ⚙️ 运维工程师 - 部署配置

**推荐阅读顺序：**
1. [configuration.md](./configuration.md) - 环境配置
2. [vespa-integration.md](./vespa-integration.md) - Vespa 部署
3. [indexing.md](./indexing.md) - 索引系统运维

### 🔬 功能扩展 - 定制开发

**推荐阅读顺序：**
1. [parsing.md](./parsing.md) - 自定义解析器
2. [chunking.md](./chunking.md) - 分块策略配置
3. [deep-research.md](./deep-research.md) - Deep Research 定制

## 文档详细说明

### 入门类文档

| 文档 | 内容 | 适用场景 |
|------|------|----------|
| [README.md](./README.md) | 项目介绍、快速示例、目录结构 | 首次接触项目 |
| [getting-started.md](./getting-started.md) | 安装步骤、依赖说明、基础示例 | 开始使用项目 |
| [configuration.md](./configuration.md) | 环境变量、配置选项、参数说明 | 配置和调优 |

### 架构类文档

| 文档 | 内容 | 适用场景 |
|------|------|----------|
| [architecture.md](./architecture.md) | 系统架构图、组件说明、数据流、Ingestion Pipeline | 理解系统设计 |
| [ingestion-indexing-design.md](./ingestion-indexing-design.md) | 摄取模块设计、工作流、扩展点 | 摄取模块开发 |

### 功能模块文档

| 文档 | 内容 | 适用场景 |
|------|------|----------|
| [deep-research.md](./deep-research.md) | Deep Research 架构、阶段、使用示例 | 使用深度研究功能 |
| [vespa-integration.md](./vespa-integration.md) | Vespa 配置、Schema、搜索示例 | Vespa 集成和调优 |
| [parsing.md](./parsing.md) | 解析器接口、内置解析器、自定义扩展 | 文档解析定制 |
| [chunking.md](./chunking.md) | 分块策略、配置参数、mini/large chunks | 分块策略调优 |
| [indexing.md](./indexing.md) | 索引流程、Embedding、搜索过滤 | 索引系统使用 |

### 参考手册

| 文档 | 内容 | 适用场景 |
|------|------|----------|
| [api-reference.md](./api-reference.md) | 模型定义、方法签名、使用示例 | API 开发参考 |

## 核心概念速查

### Agent 类型

| Agent | 用途 | 文档 |
|-------|------|------|
| ChatAgent | 对话式 RAG，单轮/多轮对话 | [architecture.md](./architecture.md) |
| DeepResearchOrchestrator | 多步骤深度研究 | [deep-research.md](./deep-research.md) |

### 数据流程

```
文档输入 → 解析 → 分块 → 向量化 → 索引 → 搜索 → Agent 响应
           │       │       │        │       │
           ▼       ▼       ▼        ▼       ▼
        parsing  chunking embedding indexing retrieval
          .md      .md      (配置)    .md    architecture
```

### 存储层

| 组件 | 用途 | 相关文档 |
|------|------|----------|
| PostgreSQL | 元数据、摄取状态 | [ingestion-indexing-design.md](./ingestion-indexing-design.md) |
| MinIO | 原始文件、解析产物 | [ingestion-indexing-design.md](./ingestion-indexing-design.md) |
| Vespa | 向量索引、混合搜索 | [vespa-integration.md](./vespa-integration.md), [indexing.md](./indexing.md) |

## 常见任务导航

### "我想..."

| 任务 | 参考文档 |
|------|----------|
| 快速体验 Agent RAG | [README.md](./README.md) |
| 安装和配置项目 | [getting-started.md](./getting-started.md) |
| 理解系统架构 | [architecture.md](./architecture.md) |
| 配置 LLM/Embedding | [configuration.md](./configuration.md) |
| 使用 Deep Research | [deep-research.md](./deep-research.md) |
| 部署 Vespa 索引 | [vespa-integration.md](./vespa-integration.md) |
| 添加新的文件解析器 | [parsing.md](./parsing.md) |
| 调整分块策略 | [chunking.md](./chunking.md) |
| 理解索引流程 | [indexing.md](./indexing.md) |
| 查看 API 定义 | [api-reference.md](./api-reference.md) |

### "出了问题..."

| 问题类型 | 参考文档 |
|----------|----------|
| 解析失败 | [parsing.md](./parsing.md) - Unstructured API 回退 |
| 分块过大/过小 | [chunking.md](./chunking.md) - 配置参数 |
| 搜索结果不准 | [vespa-integration.md](./vespa-integration.md) - hybrid_alpha 调优 |
| 索引失败 | [indexing.md](./indexing.md) - 错误处理和重试 |
| 配置不生效 | [configuration.md](./configuration.md) - 环境变量 |

## 文档版本

- 最后更新: 2024-12-28
- Agent RAG 版本: 0.1.0
- 文档维护: 持续更新中

## 贡献文档

如发现文档问题或有改进建议，请：
1. 提交 Issue 描述问题
2. 或直接提交 PR 修改文档

文档规范：
- 使用中文编写（技术术语可保留英文）
- 包含代码示例和配置说明
- 提供清晰的架构图和流程图
