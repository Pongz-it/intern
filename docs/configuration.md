# Configuration Guide

Complete configuration reference for Agent RAG.

## Environment Variables

### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_RAG_LLM_MODEL` | `gpt-4o` | LLM model name |
| `AGENT_RAG_LLM_PROVIDER` | `litellm` | LLM provider |
| `AGENT_RAG_LLM_API_KEY` | - | API key |
| `AGENT_RAG_LLM_API_BASE` | - | Custom API endpoint |
| `AGENT_RAG_LLM_MAX_TOKENS` | `4096` | Max output tokens |
| `AGENT_RAG_LLM_MAX_INPUT_TOKENS` | `128000` | Max input tokens |
| `AGENT_RAG_LLM_TEMPERATURE` | `0.0` | Temperature |
| `AGENT_RAG_LLM_TIMEOUT` | `120` | Request timeout (seconds) |
| `AGENT_RAG_LLM_IS_REASONING_MODEL` | `false` | Enable reasoning model mode |
| `AGENT_RAG_LLM_REASONING_EFFORT` | `medium` | Reasoning effort (low/medium/high) |

### Embedding Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_RAG_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `AGENT_RAG_EMBEDDING_PROVIDER` | `litellm` | Embedding provider |
| `AGENT_RAG_EMBEDDING_API_KEY` | - | API key (uses LLM key if not set) |
| `AGENT_RAG_EMBEDDING_API_BASE` | - | Custom API endpoint |
| `AGENT_RAG_EMBEDDING_DIMENSION` | `1536` | Embedding dimension |
| `AGENT_RAG_EMBEDDING_BATCH_SIZE` | `32` | Batch size for embedding |

### Document Index Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_RAG_INDEX_TYPE` | `memory` | Index type: `memory` or `vespa` |
| `AGENT_RAG_VESPA_HOST` | `localhost` | Vespa host |
| `AGENT_RAG_VESPA_PORT` | `8080` | Vespa port |
| `AGENT_RAG_VESPA_APP_NAME` | `agent_rag` | Vespa application name |
| `AGENT_RAG_VESPA_TIMEOUT` | `30` | Vespa request timeout |
| `AGENT_RAG_VESPA_SCHEMA_NAME` | `agent_rag_chunk` | Vespa schema name |
| `AGENT_RAG_VESPA_TITLE_CONTENT_RATIO` | `0.2` | Title/content ratio for ranking |
| `AGENT_RAG_VESPA_DECAY_FACTOR` | `0.5` | Time decay factor for recency bias |
| `AGENT_RAG_MEMORY_PERSIST_PATH` | - | Path to persist memory index |

### Vespa Schema Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_RAG_VESPA_SCHEMA_NAME` | `agent_rag_chunk` | Schema name |
| `AGENT_RAG_VESPA_SCHEMA_DIM` | `1536` | Embedding dimension |
| `AGENT_RAG_VESPA_EMBEDDING_PRECISION` | `float` | Precision: float/bfloat16/int8 |
| `AGENT_RAG_VESPA_MULTI_TENANT` | `false` | Enable multi-tenant |
| `AGENT_RAG_VESPA_ENABLE_TITLE_EMBEDDING` | `true` | Enable title embedding |
| `AGENT_RAG_VESPA_ENABLE_KNOWLEDGE_GRAPH` | `true` | Enable KG fields |
| `AGENT_RAG_VESPA_ENABLE_LARGE_CHUNKS` | `false` | Enable large chunk references |
| `AGENT_RAG_VESPA_ENABLE_ACCESS_CONTROL` | `false` | Enable ACL |
| `AGENT_RAG_VESPA_DEFAULT_DECAY_FACTOR` | `0.5` | Default time decay factor |
| `AGENT_RAG_VESPA_RERANK_COUNT` | `1000` | Rerank count |
| `AGENT_RAG_VESPA_REDUNDANCY` | `1` | Content redundancy |
| `AGENT_RAG_VESPA_SEARCHABLE_COPIES` | `1` | Searchable copies |
| `AGENT_RAG_VESPA_SEARCH_THREADS` | `4` | Search threads per node |
| `AGENT_RAG_VESPA_SUMMARY_THREADS` | `2` | Summary threads per node |

### Search Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_RAG_SEARCH_DEFAULT_HYBRID_ALPHA` | `0.5` | Default hybrid alpha (0=keyword, 1=semantic) |
| `AGENT_RAG_SEARCH_KEYWORD_HYBRID_ALPHA` | `0.2` | Alpha for keyword queries |
| `AGENT_RAG_SEARCH_NUM_RESULTS` | `10` | Number of search results |
| `AGENT_RAG_SEARCH_MAX_CHUNKS_PER_RESPONSE` | `15` | Max chunks in response |
| `AGENT_RAG_SEARCH_ENABLE_QUERY_EXPANSION` | `true` | Enable query expansion |
| `AGENT_RAG_SEARCH_MAX_EXPANDED_QUERIES` | `3` | Max expanded queries |
| `AGENT_RAG_SEARCH_ENABLE_DOCUMENT_SELECTION` | `true` | Enable LLM selection |
| `AGENT_RAG_SEARCH_MAX_DOCUMENTS_TO_SELECT` | `10` | Max documents to select |
| `AGENT_RAG_SEARCH_MAX_CHUNKS_FOR_RELEVANCE` | `3` | Chunks for relevance check |
| `AGENT_RAG_SEARCH_ENABLE_CONTEXT_EXPANSION` | `true` | Enable context expansion |
| `AGENT_RAG_SEARCH_CONTEXT_EXPANSION_CHUNKS` | `2` | Adjacent chunks to include |
| `AGENT_RAG_SEARCH_MAX_CONTEXT_TOKENS` | `4000` | Max context tokens |
| `AGENT_RAG_SEARCH_MAX_FULL_DOCUMENT_CHUNKS` | `5` | Chunks around for full doc |
| `AGENT_RAG_SEARCH_ENABLE_RERANKING` | `false` | Enable reranking |
| `AGENT_RAG_SEARCH_RERANK_MODEL` | - | Rerank model name |
| `AGENT_RAG_SEARCH_ORIGINAL_QUERY_WEIGHT` | `0.5` | Original query weight |
| `AGENT_RAG_SEARCH_LLM_SEMANTIC_QUERY_WEIGHT` | `1.3` | LLM semantic weight |
| `AGENT_RAG_SEARCH_LLM_KEYWORD_QUERY_WEIGHT` | `1.0` | LLM keyword weight |
| `AGENT_RAG_SEARCH_LLM_NON_CUSTOM_QUERY_WEIGHT` | `0.7` | LLM non-custom weight |
| `AGENT_RAG_SEARCH_RRF_K_VALUE` | `50` | RRF k parameter |

### Agent Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_RAG_AGENT_MODE` | `chat` | Mode: `chat` or `deep_research` |
| `AGENT_RAG_AGENT_MAX_CYCLES` | `6` | Max agent cycles |
| `AGENT_RAG_AGENT_MAX_STEPS` | - | Override for max cycles |
| `AGENT_RAG_AGENT_MAX_TOKENS` | - | Max response tokens |
| `AGENT_RAG_AGENT_ENABLED_TOOLS` | `internal_search,web_search,open_url` | Enabled tools (comma-separated) |
| `AGENT_RAG_AGENT_ENABLE_CITATIONS` | `true` | Enable citation processing |

### Deep Research Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_RAG_DR_MAX_ORCHESTRATOR_CYCLES` | `8` | Max orchestrator cycles |
| `AGENT_RAG_DR_MAX_RESEARCH_CYCLES` | `3` | Max research iterations |
| `AGENT_RAG_DR_MAX_RESEARCH_AGENTS` | `5` | Max parallel agents |
| `AGENT_RAG_DR_NUM_RESEARCH_AGENTS` | - | Override number of agents |
| `AGENT_RAG_DR_MAX_AGENT_CYCLES` | - | Override agent cycles |
| `AGENT_RAG_DR_SKIP_CLARIFICATION` | `false` | Skip clarification phase |
| `AGENT_RAG_DR_ENABLE_THINK_TOOL` | `true` | Enable think tool |
| `AGENT_RAG_DR_MAX_INTERMEDIATE_REPORT_TOKENS` | `10000` | Intermediate report tokens |
| `AGENT_RAG_DR_MAX_FINAL_REPORT_TOKENS` | `20000` | Final report tokens |

### Logging Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_RAG_LOG_LEVEL` | `INFO` | Log level |
| `AGENT_RAG_LOG_FORMAT` | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` | Log format |

---

## Loading Configuration

### From Environment

```python
from agent_rag.core.env_config import (
    load_dotenv,
    get_config_from_env,
)

# Load .env file (optional)
load_dotenv()

# Get complete configuration
config = get_config_from_env()
```

### From Code

```python
from agent_rag.core.config import (
    AgentRAGConfig,
    LLMConfig,
    EmbeddingConfig,
    SearchConfig,
    AgentConfig,
)

config = AgentRAGConfig(
    llm=LLMConfig(
        model="gpt-4o",
        api_key="sk-...",
    ),
    embedding=EmbeddingConfig(
        model="text-embedding-3-small",
    ),
    agent=AgentConfig(
        max_cycles=6,
        search=SearchConfig(
            default_hybrid_alpha=0.5,
        ),
    ),
)
```

### From Dictionary

```python
config = AgentRAGConfig.from_dict({
    "llm": {
        "model": "gpt-4o",
        "api_key": "sk-...",
    },
    "agent": {
        "max_cycles": 6,
    },
})
```

---

## Configuration Presets

### Minimal (Development)

```bash
# .env
AGENT_RAG_LLM_MODEL=gpt-4o-mini
AGENT_RAG_LLM_API_KEY=sk-...
AGENT_RAG_INDEX_TYPE=memory
```

### Standard (Production)

```bash
# .env
AGENT_RAG_LLM_MODEL=gpt-4o
AGENT_RAG_LLM_API_KEY=sk-...

AGENT_RAG_INDEX_TYPE=vespa
AGENT_RAG_VESPA_HOST=vespa.example.com
AGENT_RAG_VESPA_PORT=8080

AGENT_RAG_SEARCH_ENABLE_QUERY_EXPANSION=true
AGENT_RAG_SEARCH_ENABLE_DOCUMENT_SELECTION=true
```

### Enterprise (Full Features)

```bash
# .env
AGENT_RAG_LLM_MODEL=gpt-4o
AGENT_RAG_LLM_API_KEY=sk-...

AGENT_RAG_INDEX_TYPE=vespa
AGENT_RAG_VESPA_HOST=vespa.example.com
AGENT_RAG_VESPA_MULTI_TENANT=true
AGENT_RAG_VESPA_ENABLE_ACCESS_CONTROL=true
AGENT_RAG_VESPA_ENABLE_KNOWLEDGE_GRAPH=true
AGENT_RAG_VESPA_RERANK_COUNT=2000

AGENT_RAG_SEARCH_ENABLE_RERANKING=true
AGENT_RAG_SEARCH_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Deep Research

```bash
# .env
AGENT_RAG_AGENT_MODE=deep_research
AGENT_RAG_DR_MAX_ORCHESTRATOR_CYCLES=10
AGENT_RAG_DR_MAX_RESEARCH_AGENTS=5
AGENT_RAG_DR_SKIP_CLARIFICATION=false
AGENT_RAG_DR_ENABLE_THINK_TOOL=true
```

---

## Vespa Schema Presets

### Minimal
```python
config = get_schema_preset("minimal")
# - Basic fields only
# - No title embedding
# - No knowledge graph
# - Single tenant
```

### Standard
```python
config = get_schema_preset("standard")
# - Title embedding enabled
# - Large chunks enabled
# - Optimized for general use
```

### Enterprise
```python
config = get_schema_preset("enterprise")
# - Full features enabled
# - Multi-tenant
# - Access control
# - Knowledge graph
# - High rerank count
```

---

## Best Practices

### LLM Selection

| Use Case | Recommended Model |
|----------|-------------------|
| Fast responses | `gpt-4o-mini`, `claude-3-haiku` |
| Quality responses | `gpt-4o`, `claude-3-5-sonnet` |
| Reasoning tasks | `o1`, `claude-3-5-sonnet` |
| Cost-sensitive | `gpt-4o-mini` |

### Hybrid Alpha Tuning

| Content Type | Recommended Alpha |
|--------------|-------------------|
| Technical docs | 0.4-0.5 |
| Natural language | 0.6-0.7 |
| Code search | 0.3-0.4 |
| Mixed content | 0.5 |

### Search Performance

```bash
# For faster search
AGENT_RAG_SEARCH_NUM_RESULTS=5
AGENT_RAG_SEARCH_ENABLE_QUERY_EXPANSION=false
AGENT_RAG_SEARCH_ENABLE_DOCUMENT_SELECTION=false

# For better quality
AGENT_RAG_SEARCH_NUM_RESULTS=15
AGENT_RAG_SEARCH_ENABLE_QUERY_EXPANSION=true
AGENT_RAG_SEARCH_ENABLE_DOCUMENT_SELECTION=true
AGENT_RAG_SEARCH_MAX_DOCUMENTS_TO_SELECT=10
```

### Memory Management

```bash
# For large document sets
AGENT_RAG_SEARCH_MAX_CHUNKS_PER_RESPONSE=10
AGENT_RAG_SEARCH_MAX_CONTEXT_TOKENS=3000

# For detailed responses
AGENT_RAG_SEARCH_MAX_CHUNKS_PER_RESPONSE=20
AGENT_RAG_SEARCH_MAX_CONTEXT_TOKENS=6000
```
