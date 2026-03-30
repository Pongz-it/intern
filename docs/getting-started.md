# Getting Started

Quick start guide for Agent RAG.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/onyx-dot-app/onyx.git
cd onyx/agent_rag

# Install with pip
pip install -e .

# Install with optional dependencies
pip install -e ".[vespa]"  # Vespa support
pip install -e ".[dev]"    # Development tools
pip install -e ".[all]"    # Everything
```

### Dependencies

**Core:**
- Python >= 3.10
- litellm >= 1.40.0
- httpx >= 0.25.0
- pydantic >= 2.0.0
- tiktoken >= 0.5.0
- numpy >= 1.24.0
- jinja2 >= 3.0.0

**Optional:**
- pyvespa >= 0.45.0 (Vespa support)
- pytest >= 7.0.0 (Testing)

---

## Quick Start

### 1. Configure Environment

Create `.env` file:

```bash
# Required
AGENT_RAG_LLM_MODEL=gpt-4o
AGENT_RAG_LLM_API_KEY=sk-your-api-key

# Optional: Vespa (defaults to memory index)
AGENT_RAG_INDEX_TYPE=vespa
AGENT_RAG_VESPA_HOST=localhost
AGENT_RAG_VESPA_PORT=8080
```

### 2. Basic Chat Agent

```python
from agent_rag import ChatAgent, LLMConfig, AgentConfig
from agent_rag.llm.providers import LiteLLMProvider
from agent_rag.document_index.memory import MemoryIndex
from agent_rag.tools.builtin.search import SearchTool

# Create LLM
llm = LiteLLMProvider(LLMConfig(
    model="gpt-4o",
    api_key="sk-your-api-key",
))

# Create document index
index = MemoryIndex()

# Index some documents
from agent_rag.core.models import Chunk

chunks = [
    Chunk(
        document_id="doc1",
        chunk_id=0,
        content="Agent RAG is a retrieval augmented generation system.",
        title="Introduction to Agent RAG",
    ),
    Chunk(
        document_id="doc1",
        chunk_id=1,
        content="It supports both chat and deep research modes.",
        title="Introduction to Agent RAG",
    ),
]
index.index(chunks)

# Create search tool
search_tool = SearchTool(
    document_index=index,
    llm=llm,
)

# Create agent
agent = ChatAgent(llm=llm, config=AgentConfig())
agent.tool_registry.register(search_tool)

# Run query
response = agent.run("What is Agent RAG?")
print(response.content)
```

### 3. Streaming Response

```python
# Stream tokens
for token in agent.run_stream("Tell me about Agent RAG features"):
    print(token, end="", flush=True)
print()  # New line at end
```

### 4. Multi-turn Conversation

```python
from agent_rag.core.models import Message

# First turn
response = agent.run("What is Agent RAG?")
print(f"Assistant: {response.content}")

# Follow-up
agent.add_messages([Message(role="user", content="What modes does it support?")])
response = agent.run()
print(f"Assistant: {response.content}")

# Reset for new conversation
agent.reset()
```

---

## Using Vespa

### 1. Deploy Vespa

```bash
# Using Docker
docker run --rm -d \
  -p 8080:8080 \
  --name vespa \
  vespaengine/vespa

# Wait for startup
sleep 30
```

### 2. Generate Schema

```python
from agent_rag.document_index.vespa.schema_config import (
    VespaSchemaConfig,
    VespaSchemaRenderer,
    get_schema_preset,
)
from pathlib import Path

# Use preset
config = get_schema_preset("standard")

# Or custom
config = VespaSchemaConfig(
    schema_name="my_chunks",
    dim=1536,
    enable_title_embedding=True,
)

# Generate application package
renderer = VespaSchemaRenderer()
files = renderer.generate_application_package(config, Path("./vespa_app"))

print(f"Generated: {files}")
```

### 3. Deploy Application

```bash
# Deploy to Vespa
vespa deploy ./vespa_app
```

### 4. Use VespaIndex

```python
from agent_rag.document_index.vespa import VespaIndex

index = VespaIndex(
    host="localhost",
    port=8080,
    schema_name="my_chunks",
)

# Index documents
result = index.index(chunks)
print(f"Indexed {result.num_indexed} chunks")

# Search
results = index.search(
    query="machine learning",
    hybrid_alpha=0.5,
    limit=10,
)
```

---

## Deep Research

### Basic Usage

```python
from agent_rag import DeepResearchAgent
from agent_rag.core.config import DeepResearchConfig

agent = DeepResearchAgent(
    llm=llm,
    config=DeepResearchConfig(
        max_orchestrator_cycles=8,
        max_research_agents=5,
    ),
)

# Register search tools
agent.tool_registry.register(search_tool)

# Run research
for packet in agent.run_stream("Comprehensive analysis of RAG systems"):
    if packet.type == "report_token":
        print(packet.token, end="")
    elif packet.type == "phase_start":
        print(f"\n[Phase: {packet.phase}]")
    elif packet.type == "sub_questions":
        print(f"Sub-questions: {packet.questions}")
```

### Packet Types

```python
# Handle different packet types
for packet in agent.run_stream(question):
    match packet.type:
        case "research_start":
            print("Research started")
        case "phase_start":
            print(f"Phase: {packet.phase}")
        case "sub_questions":
            for q in packet.questions:
                print(f"  - {q}")
        case "agent_start":
            print(f"Agent {packet.agent_id}: {packet.question}")
        case "think_content":
            print(f"Thinking: {packet.content[:100]}...")
        case "finding_summary":
            print(f"Finding: {packet.summary[:100]}...")
        case "report_token":
            print(packet.token, end="")
        case "research_end":
            print("\nResearch complete")
```

---

## Custom Tools

### Creating a Tool

```python
from agent_rag.tools.interface import Tool, ToolResponse

class CalculatorTool(Tool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform mathematical calculations"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        }

    def execute(self, expression: str) -> ToolResponse:
        try:
            # WARNING: eval is dangerous - use proper parser in production
            result = eval(expression)
            return ToolResponse(
                success=True,
                data={"result": result},
            )
        except Exception as e:
            return ToolResponse(
                success=False,
                error=str(e),
            )

# Register and use
agent.tool_registry.register(CalculatorTool())
```

---

## Callbacks

### Streaming Callback

```python
from agent_rag.core.callbacks import StreamCallback

def on_token(token: str) -> None:
    print(token, end="", flush=True)

callback = StreamCallback(on_token=on_token)
agent = ChatAgent(llm=llm, stream_callback=callback)
```

### Tool Callback

```python
from agent_rag.core.callbacks import ToolCallback

def on_tool_start(name: str, args: dict) -> None:
    print(f"Calling tool: {name}")

def on_tool_end(name: str, result) -> None:
    print(f"Tool {name} returned")

callback = ToolCallback(
    on_tool_start=on_tool_start,
    on_tool_end=on_tool_end,
)
agent = ChatAgent(llm=llm, tool_callback=callback)
```

---

## Testing

### Run Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_vespa_enhanced.py -v

# With coverage
python -m pytest tests/ --cov=agent_rag --cov-report=html
```

### Test Example

```python
import pytest
from agent_rag.core.models import Chunk
from agent_rag.document_index.memory import MemoryIndex

def test_memory_index_search():
    index = MemoryIndex()

    chunks = [
        Chunk(document_id="doc1", chunk_id=0, content="Hello world"),
    ]
    index.index(chunks)

    results = index.search("hello", limit=5)
    assert len(results) == 1
    assert results[0].document_id == "doc1"
```

---

## Next Steps

1. [Architecture](./architecture.md) - Understand the system design
2. [API Reference](./api-reference.md) - Complete API documentation
3. [Configuration](./configuration.md) - All configuration options
4. [Deep Research](./deep-research.md) - Advanced research features
5. [Vespa Integration](./vespa-integration.md) - Production Vespa setup
