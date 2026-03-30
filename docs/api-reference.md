# API Reference

Complete API documentation for Agent RAG.

## Core Models

### Chunk

Represents a document chunk with metadata and embeddings.

```python
from agent_rag.core.models import Chunk

chunk = Chunk(
    document_id="doc_123",
    chunk_id=0,
    content="Document content here...",
    title="Document Title",
    source_type="confluence",
    embedding=[0.1, 0.2, ...],  # Optional embedding vector
    metadata={"author": "John"},
    # Knowledge graph fields
    kg_entities=["Entity1", "Entity2"],
    kg_relationships=[KGRelationship(source="A", rel_type="relates_to", target="B")],
    # Boost and scoring
    boost=1.0,
    score=0.95,
)
```

**Key Properties:**
- `unique_id`: Returns `"{document_id}_{chunk_id}"`
- `to_vespa_fields()`: Convert to Vespa document format

### Message

Represents a conversation message.

```python
from agent_rag.core.models import Message

# User message
user_msg = Message(role="user", content="What is Agent RAG?")

# Assistant message with tool calls
assistant_msg = Message(
    role="assistant",
    content="Let me search for that...",
    tool_calls=[
        ToolCall(id="call_1", name="search", arguments='{"query": "Agent RAG"}')
    ]
)

# Tool result message
tool_msg = Message(role="tool", content="Search results...", tool_call_id="call_1")
```

### Section

Represents a group of related chunks.

```python
from agent_rag.core.models import Section

section = Section(
    center_chunk=chunk,
    chunks=[chunk1, chunk2, chunk3],
    combined_content="All chunks content combined..."
)
```

### AgentResponse

Response from an agent run.

```python
from agent_rag.core.models import AgentResponse

response = AgentResponse(
    content="The answer is...",
    citations=[Citation(number=1, document_id="doc1", ...)],
    tool_calls=[...],
    usage={"prompt_tokens": 100, "completion_tokens": 50},
)
```

---

## Configuration

### LLMConfig

LLM provider configuration.

```python
from agent_rag.core.config import LLMConfig

config = LLMConfig(
    model="gpt-4o",
    provider="litellm",
    api_key="sk-...",
    api_base=None,  # Optional custom endpoint
    max_tokens=4096,
    max_input_tokens=128000,
    temperature=0.0,
    timeout=120,
    # Reasoning model options
    is_reasoning_model=False,
    reasoning_effort="medium",  # low, medium, high
)
```

### EmbeddingConfig

Embedding model configuration.

```python
from agent_rag.core.config import EmbeddingConfig

config = EmbeddingConfig(
    model="text-embedding-3-small",
    provider="litellm",
    dimension=1536,
    batch_size=32,
)
```

### SearchConfig

Search behavior configuration.

```python
from agent_rag.core.config import SearchConfig

config = SearchConfig(
    # Hybrid search
    default_hybrid_alpha=0.5,  # 0=keyword, 1=semantic
    num_results=10,
    max_chunks_per_response=15,
    # Query expansion
    enable_query_expansion=True,
    max_expanded_queries=3,
    # Document selection (LLM-based)
    enable_document_selection=True,
    max_documents_to_select=10,
    # Context expansion
    enable_context_expansion=True,
    context_expansion_chunks=2,
    max_full_document_chunks=5,
    # RRF weights
    original_query_weight=0.5,
    llm_semantic_query_weight=1.3,
    rrf_k_value=50,
)
```

### AgentConfig

Agent behavior configuration.

```python
from agent_rag.core.config import AgentConfig, AgentMode

config = AgentConfig(
    mode=AgentMode.CHAT,  # or AgentMode.DEEP_RESEARCH
    system_prompt=None,  # Optional custom prompt
    max_cycles=6,
    enabled_tools=["internal_search", "web_search", "open_url"],
    enable_citations=True,
    search=SearchConfig(),
    deep_research=DeepResearchConfig(),
)
```

### DeepResearchConfig

Deep Research specific configuration.

```python
from agent_rag.core.config import DeepResearchConfig

config = DeepResearchConfig(
    max_orchestrator_cycles=8,
    max_research_cycles=3,
    max_research_agents=5,
    skip_clarification=False,
    enable_think_tool=True,  # For non-reasoning models
    max_intermediate_report_tokens=10000,
    max_final_report_tokens=20000,
)
```

---

## Agents

### ChatAgent

Conversational RAG agent with tool calling.

```python
from agent_rag.agent.chat_agent import ChatAgent

agent = ChatAgent(
    llm=llm,
    config=AgentConfig(),
    tool_registry=registry,
    system_prompt="Custom prompt...",
    stream_callback=lambda token: print(token, end=""),
    tool_callback=ToolCallback(
        on_tool_start=lambda name, args: print(f"Calling {name}"),
        on_tool_end=lambda name, result: print(f"Result: {result}"),
    ),
)

# Single turn
response = agent.run("What is Agent RAG?")

# Multi-turn
agent.add_messages([Message(role="user", content="Follow up question")])
response = agent.run()

# Streaming
for token in agent.run_stream("Tell me more"):
    print(token, end="")

# Reset conversation
agent.reset()
```

### DeepResearchAgent (DeepResearchOrchestrator)

Multi-step research agent with parallel execution.

```python
from agent_rag import DeepResearchAgent
from agent_rag.core.config import DeepResearchConfig

agent = DeepResearchAgent(
    llm=llm,
    config=DeepResearchConfig(),
    tool_registry=registry,
)

# Run research
for packet in agent.run_stream("Comprehensive analysis of topic X"):
    if packet.type == "report_token":
        print(packet.token, end="")
    elif packet.type == "phase_start":
        print(f"\n[Phase: {packet.phase}]")
```

**Streaming Packet Types:**
- `ResearchStartPacket`, `ResearchEndPacket`
- `PhaseStartPacket`, `PhaseEndPacket`
- `SubQuestionsPacket`
- `AgentStartPacket`, `AgentEndPacket`
- `ThinkStartPacket`, `ThinkContentPacket`, `ThinkEndPacket`
- `ReportTokenPacket`
- `MetricsPacket`

---

## Document Index

### DocumentIndex Protocol

```python
from agent_rag.document_index.interface import DocumentIndex, ChunkRequest

class DocumentIndex(Protocol):
    def index(self, chunks: list[Chunk]) -> IndexingResult: ...
    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        embedding: Optional[list[float]] = None,
        hybrid_alpha: float = 0.5,
        limit: int = 10,
    ) -> list[Chunk]: ...
    def get_chunk(self, document_id: str, chunk_id: int) -> Optional[Chunk]: ...
    def id_based_retrieval(
        self,
        chunk_requests: list[ChunkRequest],
        batch_retrieval: bool = True,
    ) -> list[Chunk]: ...
    def delete_document(self, document_id: str) -> bool: ...
```

### VespaIndex

```python
from agent_rag.document_index.vespa import VespaIndex

index = VespaIndex(
    host="localhost",
    port=8080,
    app_name="agent_rag",
    schema_name="agent_rag_chunk",
)

# Index chunks
result = index.index(chunks)
print(f"Indexed {result.num_indexed} chunks")

# Search
results = index.search(
    query="machine learning",
    embedding=embedding_vector,
    hybrid_alpha=0.5,
    limit=10,
)

# Cleanup chunks (remove indexing augmentations)
cleaned = index.cleanup_chunks(results)
```

### EnhancedVespaIndex

Extended Vespa index with additional features.

```python
from agent_rag.document_index.vespa.enhanced_vespa_index import EnhancedVespaIndex

index = EnhancedVespaIndex(
    host="localhost",
    port=8080,
    schema_config=VespaSchemaConfig(
        enable_knowledge_graph=True,
        enable_title_embedding=True,
        multi_tenant=True,
    ),
)

# Parallel indexing
result = index.index_parallel(chunks, batch_size=128, num_threads=32)

# Visit API for large-scale operations
for chunk in index.visit_documents(chunk_count=1000):
    process(chunk)

# Chunk cleanup
cleaned = index.cleanup_chunks(results)
```

### VespaSchemaConfig

Schema configuration for Vespa.

```python
from agent_rag.document_index.vespa.schema_config import (
    VespaSchemaConfig,
    VespaSchemaRenderer,
    get_schema_preset,
)

# Use preset
config = get_schema_preset("enterprise")  # minimal, standard, enterprise

# Custom config
config = VespaSchemaConfig(
    schema_name="my_chunk",
    dim=1536,
    embedding_precision="float",  # float, bfloat16, int8
    enable_title_embedding=True,
    enable_knowledge_graph=True,
    multi_tenant=True,
    enable_access_control=True,
    rerank_count=2000,
)

# Generate schema files
renderer = VespaSchemaRenderer()
files = renderer.generate_application_package(config, output_dir)
```

---

## Tools

### Tool Protocol

```python
from agent_rag.tools.interface import Tool, ToolResponse

class Tool(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def parameters(self) -> dict: ...
    def execute(self, **kwargs) -> ToolResponse: ...
```

### SearchTool

```python
from agent_rag.tools.builtin.search import SearchTool

tool = SearchTool(
    document_index=index,
    llm=llm,
    embedding_provider=embedder,
    config=SearchConfig(),
)

result = tool.execute(query="machine learning concepts")
```

### WebSearchTool

```python
from agent_rag.tools.builtin.web_search import WebSearchTool

tool = WebSearchTool(api_key="tavily-api-key")
result = tool.execute(query="latest AI news")
```

### ToolRegistry

```python
from agent_rag.tools.registry import ToolRegistry

registry = ToolRegistry()
registry.register(search_tool)
registry.register(web_search_tool)

# Get tool definitions for LLM
definitions = registry.get_all_definitions()

# Execute tool
result = registry.execute("search", query="test")
```

---

## Retrieval

### Reciprocal Rank Fusion

```python
from agent_rag.retrieval.ranking import (
    reciprocal_rank_fusion,
    weighted_reciprocal_rank_fusion,
    linear_combination,
)

# Basic RRF
merged = reciprocal_rank_fusion(
    chunk_lists=[semantic_results, keyword_results],
    k=60,
    weights=[1.0, 0.5],
)

# Weighted RRF with tie-breaking
merged = weighted_reciprocal_rank_fusion(
    ranked_results=[list1, list2],
    weights=[1.0, 0.5],
    id_extractor=lambda x: x.unique_id,
    k=50,
)

# Linear combination
merged = linear_combination(
    chunk_lists=[list1, list2],
    weights=[0.7, 0.3],
    normalize=True,
)
```

---

## Environment Configuration

### Loading from .env

```python
from agent_rag.core.env_config import (
    load_dotenv,
    get_config_from_env,
    get_llm_config_from_env,
    get_vespa_schema_config_from_env,
)

# Load .env file
load_dotenv()

# Get full configuration
config = get_config_from_env()

# Get specific config
llm_config = get_llm_config_from_env()
vespa_config = get_vespa_schema_config_from_env()
```

### Environment Variables

See [Configuration](./configuration.md) for complete list of environment variables.

---

## Citation Processing

### DynamicCitationProcessor

```python
from agent_rag.citation.processor import DynamicCitationProcessor
from agent_rag.citation.utils import chunks_to_citations

# Create citations from chunks
citations = chunks_to_citations(chunks)

# Process streaming response
processor = DynamicCitationProcessor(citations)
for token in stream:
    processed = processor.process_token(token)
    print(processed, end="")

# Get final text with renumbered citations
final_text = processor.finalize()
used_citations = processor.get_used_citations()
```

---

## Callbacks

### StreamCallback

```python
from agent_rag.core.callbacks import StreamCallback

callback = StreamCallback(on_token=lambda token: print(token, end=""))
```

### ToolCallback

```python
from agent_rag.core.callbacks import ToolCallback

callback = ToolCallback(
    on_tool_start=lambda name, args: print(f"Starting {name}"),
    on_tool_end=lambda name, result: print(f"Completed {name}"),
)
```

### AgentCallback

```python
from agent_rag.core.callbacks import AgentCallback

callback = AgentCallback(
    on_step_start=lambda step: print(f"Step {step}"),
    on_step_end=lambda step, result: print(f"Step {step} done"),
    on_agent_end=lambda response: print("Agent finished"),
)
```
