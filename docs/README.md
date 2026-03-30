# Agent RAG Documentation

A standalone Agent RAG component with Chat and Deep Research capabilities, extracted and optimized from the Onyx project.

## Overview

Agent RAG provides two main agent types:
- **ChatAgent**: Conversational RAG with tool calling and streaming
- **DeepResearchAgent**: Multi-step research with parallel execution and comprehensive report generation

## Quick Start

```python
from agent_rag import ChatAgent, LLMConfig, AgentConfig
from agent_rag.llm import LiteLLMProvider
from agent_rag.document_index.vespa import VespaIndex
from agent_rag.tools.builtin.search import SearchTool

# Configure LLM
llm_config = LLMConfig(model="gpt-4o", api_key="your-api-key")
llm = LiteLLMProvider(llm_config)

# Create document index
index = VespaIndex(host="localhost", port=8080)

# Create search tool
search_tool = SearchTool(document_index=index, llm=llm)

# Create and run agent
agent = ChatAgent(llm=llm, config=AgentConfig())
agent.tool_registry.register(search_tool)

response = agent.run("What is Agent RAG?")
print(response.content)
```

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](./architecture.md) | System architecture and component design |
| [API Reference](./api-reference.md) | Complete API documentation |
| [Configuration](./configuration.md) | Environment and config options |
| [Getting Started](./getting-started.md) | Installation and setup guide |
| [Deep Research](./deep-research.md) | Deep Research agent usage |
| [Vespa Integration](./vespa-integration.md) | Vespa document index setup |

## Project Structure

```
agent_rag/
├── agent/                    # Agent implementations
│   ├── chat_agent.py         # ChatAgent for conversational RAG
│   ├── base.py               # BaseAgent abstract class
│   ├── step.py               # Agent step execution
│   └── deep_research/        # Deep Research components
│       ├── orchestrator.py   # DeepResearchOrchestrator
│       ├── research_agent.py # Individual research agents
│       ├── report_generator.py
│       └── packets.py        # Streaming packet types
├── citation/                 # Citation processing
│   ├── processor.py          # DynamicCitationProcessor
│   ├── accumulator.py        # GlobalCitationAccumulator
│   └── utils.py              # Citation utilities
├── core/                     # Core types and config
│   ├── models.py             # Chunk, Message, Section, etc.
│   ├── config.py             # Configuration dataclasses
│   ├── env_config.py         # Environment variable loading
│   ├── callbacks.py          # Callback protocols
│   └── exceptions.py         # Custom exceptions
├── document_index/           # Document storage backends
│   ├── interface.py          # DocumentIndex protocol
│   ├── memory/               # In-memory implementation
│   └── vespa/                # Vespa implementation
│       ├── vespa_index.py    # VespaIndex
│       ├── enhanced_vespa_index.py  # Enhanced features
│       └── schema_config.py  # Schema configuration
├── embedding/                # Embedding providers
│   ├── interface.py          # Embedder protocol
│   └── providers/            # LiteLLM embedder
├── llm/                      # LLM providers
│   ├── interface.py          # LLM protocol
│   └── providers/            # LiteLLM provider
├── retrieval/                # Retrieval algorithms
│   ├── pipeline.py           # RetrievalPipeline
│   └── ranking.py            # RRF, linear combination
├── tools/                    # Tool system
│   ├── interface.py          # Tool protocol
│   ├── registry.py           # ToolRegistry
│   ├── runner.py             # ToolRunner
│   ├── builtin/              # Built-in tools
│   │   ├── search/           # SearchTool
│   │   ├── web_search/       # WebSearchTool
│   │   └── open_url/         # OpenURLTool
│   └── mcp/                  # MCP integration
└── utils/                    # Utilities
    ├── logger.py             # Logging
    ├── timing.py             # Performance timing
    └── concurrency.py        # Async utilities
```

## Key Features

### Chat Agent
- Conversational RAG with streaming support
- Tool calling with automatic citation
- Configurable system prompts
- Multi-turn conversation history

### Deep Research
- Multi-step research orchestration
- Parallel research agent execution
- Iterative refinement cycles
- Comprehensive report generation

### Document Index
- Vespa integration with hybrid search
- Knowledge graph fields support
- Multi-tenant capabilities
- Chunk cleanup for clean responses

### Retrieval
- Reciprocal Rank Fusion (RRF)
- Weighted RRF with tie-breaking
- Query expansion (semantic + keyword)
- Context-aware section expansion

## Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
# LLM Configuration
AGENT_RAG_LLM_MODEL=gpt-4o
AGENT_RAG_LLM_API_KEY=your-api-key

# Vespa Configuration
AGENT_RAG_VESPA_HOST=localhost
AGENT_RAG_VESPA_PORT=8080

# Search Configuration
AGENT_RAG_SEARCH_DEFAULT_HYBRID_ALPHA=0.5
AGENT_RAG_SEARCH_NUM_RESULTS=10
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_vespa_enhanced.py -v

# Run with coverage
python -m pytest tests/ --cov=agent_rag
```

## License

MIT License - See LICENSE file for details.
