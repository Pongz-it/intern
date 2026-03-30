"""Agent module for Agent RAG."""

from agent_rag.agent.base import AgentState, BaseAgent
from agent_rag.agent.chat_agent import ChatAgent
from agent_rag.agent.step import AgentStep, StepResult

__all__ = [
    # Base
    "BaseAgent",
    "AgentState",
    # Chat Agent
    "ChatAgent",
    "AgentStep",
    "StepResult",
]

try:
    from agent_rag.agent.session_memory_agent import SessionMemoryChatAgent, MemoryContextBuilder

    __all__.extend(
        [
            "SessionMemoryChatAgent",
            "MemoryContextBuilder",
        ]
    )
except ModuleNotFoundError:
    pass

try:
    from agent_rag.agent.deep_research import (
        DeepResearchOrchestrator,
        OrchestratorProgress,
        OrchestratorState,
        ReportGenerator,
        ResearchAgent,
        ResearchFindings,
        ResearchReport,
        ThinkTool,
        create_deep_research_agent,
    )

    __all__.extend(
        [
            "DeepResearchOrchestrator",
            "OrchestratorProgress",
            "OrchestratorState",
            "ReportGenerator",
            "ResearchAgent",
            "ResearchFindings",
            "ResearchReport",
            "ThinkTool",
            "create_deep_research_agent",
        ]
    )
except ModuleNotFoundError:
    pass
