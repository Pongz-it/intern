# Deep Research Guide

Comprehensive guide for using the Deep Research agent.

## Overview

Deep Research provides multi-step, parallel research capabilities for complex questions that require:
- Breaking down into sub-questions
- Parallel investigation by multiple agents
- Iterative refinement
- Comprehensive report generation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  DeepResearchOrchestrator                       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                      Phases                                │ │
│  │  1. Clarification → 2. Planning → 3. Execution → 4. Report│ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Research Agent Pool                         │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │  │ Agent 1 │ │ Agent 2 │ │ Agent 3 │ │ Agent N │ ...    │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘        │   │
│  │       │           │           │           │              │   │
│  │       └───────────┴───────────┴───────────┘              │   │
│  │                       │                                   │   │
│  │                       ▼                                   │   │
│  │              [Parallel Execution]                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Report Generator                        │   │
│  │  Findings → Synthesis → Sections → Final Report          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Phases

### Phase 1: Clarification (Optional)

The orchestrator may ask clarifying questions:

```python
for packet in agent.run_stream("Analyze market trends"):
    if packet.type == "clarification_question":
        print(f"Question: {packet.question}")
        # User provides clarification via follow-up message
```

Skip clarification:
```python
config = DeepResearchConfig(skip_clarification=True)
```

### Phase 2: Planning

Generates research plan and sub-questions:

```python
for packet in agent.run_stream(question):
    if packet.type == "research_plan":
        print(f"Plan: {packet.plan}")
    elif packet.type == "sub_questions":
        for q in packet.questions:
            print(f"  - {q}")
```

### Phase 3: Execution

Parallel research agents investigate sub-questions:

```python
for packet in agent.run_stream(question):
    if packet.type == "agent_start":
        print(f"Agent {packet.agent_id} starting: {packet.question}")
    elif packet.type == "agent_end":
        print(f"Agent {packet.agent_id} finished")
    elif packet.type == "finding_summary":
        print(f"Finding: {packet.summary}")
```

### Phase 4: Report Generation

Synthesizes findings into comprehensive report:

```python
report_content = ""
for packet in agent.run_stream(question):
    if packet.type == "report_start":
        print("Generating report...")
    elif packet.type == "report_token":
        report_content += packet.token
        print(packet.token, end="")
    elif packet.type == "report_end":
        print("\nReport complete!")
```

---

## Configuration

### Basic Configuration

```python
from agent_rag.core.config import DeepResearchConfig

config = DeepResearchConfig(
    max_orchestrator_cycles=8,    # Max cycles for orchestrator
    max_research_cycles=3,        # Max research iterations
    max_research_agents=5,        # Max parallel agents
    skip_clarification=False,     # Ask clarifying questions
    enable_think_tool=True,       # Enable think tool for non-reasoning models
    max_intermediate_report_tokens=10000,
    max_final_report_tokens=20000,
)
```

### Advanced Configuration

```python
config = DeepResearchConfig(
    max_orchestrator_cycles=10,
    max_research_cycles=5,
    max_research_agents=8,
    num_research_agents=5,        # Override auto-selection
    max_agent_cycles=6,           # Override per-agent cycles
    skip_clarification=True,
    enable_think_tool=True,
)
```

---

## Streaming Packets

### Packet Types

| Type | Description | Key Fields |
|------|-------------|------------|
| `research_start` | Research begins | `question`, `timestamp` |
| `research_end` | Research complete | `success`, `duration` |
| `research_error` | Error occurred | `error`, `phase` |
| `phase_start` | Phase begins | `phase` |
| `phase_end` | Phase complete | `phase` |
| `clarification_question` | Needs input | `question` |
| `research_plan` | Plan generated | `plan` |
| `sub_questions` | Questions generated | `questions` |
| `cycle_start` | Cycle begins | `cycle_number` |
| `cycle_end` | Cycle complete | `cycle_number` |
| `agent_start` | Agent begins | `agent_id`, `question` |
| `agent_end` | Agent complete | `agent_id`, `success` |
| `think_start` | Thinking begins | `agent_id` |
| `think_content` | Thinking content | `content` |
| `think_end` | Thinking complete | `agent_id` |
| `finding_summary` | Finding summary | `summary`, `agent_id` |
| `finding_key_facts` | Key facts | `facts`, `agent_id` |
| `finding_sources` | Sources used | `sources` |
| `intermediate_report_start` | Interim report | `cycle` |
| `intermediate_report_content` | Report content | `content` |
| `intermediate_report_end` | Report complete | `cycle` |
| `report_start` | Final report begins | - |
| `report_token` | Report token | `token` |
| `report_end` | Report complete | - |
| `metrics` | Performance metrics | `metrics` |

### Handling Packets

```python
from agent_rag.agent.deep_research.packets import ResearchPhase

for packet in agent.run_stream(question):
    match packet.type:
        # Lifecycle
        case "research_start":
            print(f"Starting: {packet.question}")

        case "research_error":
            print(f"Error in {packet.phase}: {packet.error}")

        # Phases
        case "phase_start":
            if packet.phase == ResearchPhase.CLARIFICATION:
                print("Clarification phase")
            elif packet.phase == ResearchPhase.PLANNING:
                print("Planning phase")
            elif packet.phase == ResearchPhase.EXECUTION:
                print("Execution phase")
            elif packet.phase == ResearchPhase.SYNTHESIS:
                print("Synthesis phase")

        # Planning
        case "sub_questions":
            print("Sub-questions:")
            for i, q in enumerate(packet.questions, 1):
                print(f"  {i}. {q}")

        # Execution
        case "agent_start":
            print(f"Agent {packet.agent_id}: {packet.question}")

        case "think_content":
            # Reasoning content (can be verbose)
            pass

        case "finding_summary":
            print(f"Finding: {packet.summary[:100]}...")

        # Report
        case "report_token":
            print(packet.token, end="", flush=True)

        # Metrics
        case "metrics":
            print(f"\nMetrics: {packet.metrics}")
```

---

## Think Tool

For non-reasoning models (e.g., GPT-4), the think tool enables structured reasoning:

```python
config = DeepResearchConfig(
    enable_think_tool=True,  # Enable for non-reasoning models
)
```

Think tool output:

```python
for packet in agent.run_stream(question):
    if packet.type == "think_start":
        print("Thinking...")
    elif packet.type == "think_content":
        print(f"  {packet.content}")
    elif packet.type == "think_end":
        print("Thought complete")
```

---

## Report Generation

### Report Structure

```python
from agent_rag.agent.deep_research.report_generator import (
    ReportConfig,
    ReportGenerator,
    format_report_markdown,
)

# Configure report
report_config = ReportConfig(
    max_tokens=20000,
    include_sources=True,
    include_methodology=True,
)

# Generate report
generator = ReportGenerator(llm, report_config)
report = generator.generate(findings, question)

# Format as markdown
markdown = format_report_markdown(report)
```

### Report Sections

```
# [Research Question]

## Executive Summary
Brief overview of findings

## Methodology
How research was conducted

## Findings

### [Sub-question 1]
Detailed findings...

### [Sub-question 2]
Detailed findings...

## Synthesis
Cross-cutting analysis

## Conclusions
Key takeaways

## Sources
[1] Source 1
[2] Source 2
...
```

---

## Citation Accumulator

Track citations across all research agents:

```python
from agent_rag.citation.accumulator import (
    GlobalCitationAccumulator,
    create_global_accumulator,
)

# Create accumulator
accumulator = create_global_accumulator()

# Citations are automatically tracked across agents
# Access final citations
final_citations = accumulator.get_all_citations()
```

---

## Error Handling

```python
for packet in agent.run_stream(question):
    if packet.type == "research_error":
        print(f"Error: {packet.error}")
        print(f"Phase: {packet.phase}")

        # Decide how to handle
        if "rate limit" in packet.error.lower():
            print("Retrying after delay...")
        else:
            print("Research failed")
            break
```

---

## Performance Metrics

```python
for packet in agent.run_stream(question):
    if packet.type == "metrics":
        metrics = packet.metrics
        print(f"Total time: {metrics['total_time_seconds']}s")
        print(f"Agents run: {metrics['agents_run']}")
        print(f"Findings: {metrics['total_findings']}")
        print(f"Sources used: {metrics['sources_used']}")
```

---

## Best Practices

### 1. Question Formulation

**Good:**
- "Comprehensive analysis of RAG systems including architecture, performance, and limitations"
- "Compare and contrast different vector database options for production RAG"

**Less Effective:**
- "Tell me about RAG" (too vague)
- "What is a vector?" (too simple, use ChatAgent)

### 2. Configuration Tuning

```python
# Quick research (1-2 minutes)
config = DeepResearchConfig(
    max_orchestrator_cycles=4,
    max_research_agents=3,
    max_research_cycles=2,
)

# Thorough research (5-10 minutes)
config = DeepResearchConfig(
    max_orchestrator_cycles=10,
    max_research_agents=8,
    max_research_cycles=5,
)
```

### 3. Tool Selection

Ensure appropriate tools are registered:

```python
# For document-based research
agent.tool_registry.register(search_tool)

# For web research
agent.tool_registry.register(web_search_tool)

# For URL content
agent.tool_registry.register(open_url_tool)
```

### 4. Memory Management

For long research sessions:

```python
# Limit intermediate report tokens
config = DeepResearchConfig(
    max_intermediate_report_tokens=5000,
    max_final_report_tokens=15000,
)
```

---

## Example: Complete Research Session

```python
from agent_rag import DeepResearchAgent
from agent_rag.core.config import DeepResearchConfig

# Setup
agent = DeepResearchAgent(
    llm=llm,
    config=DeepResearchConfig(
        max_orchestrator_cycles=8,
        max_research_agents=5,
    ),
)
agent.tool_registry.register(search_tool)
agent.tool_registry.register(web_search_tool)

# Run research
question = """
Analyze the current state of RAG systems:
1. Common architectures and their trade-offs
2. Performance optimization techniques
3. Evaluation methodologies
4. Future directions
"""

report = ""
for packet in agent.run_stream(question):
    match packet.type:
        case "phase_start":
            print(f"\n=== {packet.phase.value.upper()} ===")

        case "sub_questions":
            print("\nResearch Questions:")
            for i, q in enumerate(packet.questions, 1):
                print(f"  {i}. {q}")

        case "agent_start":
            print(f"\n[Agent {packet.agent_id}] {packet.question[:50]}...")

        case "finding_summary":
            print(f"  ✓ {packet.summary[:80]}...")

        case "report_token":
            report += packet.token
            print(packet.token, end="", flush=True)

        case "metrics":
            print(f"\n\n=== METRICS ===")
            for k, v in packet.metrics.items():
                print(f"  {k}: {v}")

# Save report
with open("research_report.md", "w") as f:
    f.write(report)
```
