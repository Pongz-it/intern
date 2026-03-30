"""Tool wrappers for Deep Research orchestration.

Supports reasoning models (o1, Claude-3.5-sonnet, etc.) with built-in
chain-of-thought and non-reasoning models requiring explicit think tool.
"""

from dataclasses import dataclass
from typing import Any, Optional

from agent_rag.agent.deep_research.report_generator import ReportGenerator
from agent_rag.agent.deep_research.research_agent import ResearchAgent, ResearchAgentConfig
from agent_rag.llm.interface import LLM
from agent_rag.tools.interface import Tool, ToolResponse
from agent_rag.tools.registry import ToolRegistry


@dataclass
class ResearchAgentToolConfig:
    """Config for research agent tool."""
    llm: LLM
    tool_registry: ToolRegistry
    agent_config: ResearchAgentConfig
    is_reasoning_model: bool = False


class ResearchAgentTool(Tool[ResearchAgentToolConfig]):
    """Tool wrapper to run a research agent for a sub-question.

    For reasoning models:
    - Uses simpler prompts without explicit think instructions
    - Relies on model's built-in chain-of-thought

    For non-reasoning models:
    - Uses detailed prompts with explicit think tool guidance
    - Requires think tool usage between searches
    """

    NAME = "research_agent"
    DESCRIPTION = "Run a focused research agent for a specific sub-question."

    def __init__(
        self,
        llm: LLM,
        tool_registry: ToolRegistry,
        agent_config: Optional[ResearchAgentConfig] = None,
        is_reasoning_model: bool = False,
        id: Optional[int] = None,
    ) -> None:
        super().__init__(id)
        self.llm = llm
        self.tool_registry = tool_registry
        self.agent_config = agent_config or ResearchAgentConfig()
        self.is_reasoning_model = is_reasoning_model

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def description(self) -> str:
        return self.DESCRIPTION

    def tool_definition(self) -> dict[str, Any]:
        return self.build_tool_definition(
            parameters={
                "sub_question": {
                    "type": "string",
                    "description": "The focused sub-question to research",
                },
                "main_question": {
                    "type": "string",
                    "description": "The original research question",
                },
            },
            required=["sub_question"],
        )

    def run(
        self,
        override_kwargs: Optional[ResearchAgentToolConfig] = None,
        **llm_kwargs: Any,
    ) -> ToolResponse:
        config = override_kwargs
        llm = config.llm if config else self.llm
        tool_registry = config.tool_registry if config else self.tool_registry
        agent_config = config.agent_config if config else self.agent_config
        is_reasoning_model = config.is_reasoning_model if config else self.is_reasoning_model

        sub_question = llm_kwargs.get("sub_question", "")
        main_question = llm_kwargs.get("main_question") or ""
        context = llm_kwargs.get("context")

        if not sub_question:
            return ToolResponse(llm_response="Error: sub_question is required")

        agent = ResearchAgent(
            llm=llm,
            tool_registry=tool_registry,
            config=agent_config,
            is_reasoning_model=is_reasoning_model,
        )
        findings = agent.research(
            sub_question=sub_question,
            main_question=main_question,
            context=context,
        )

        return ToolResponse(
            llm_response=findings.summary,
            rich_response={"findings": findings},
        )


@dataclass
class GenerateReportToolConfig:
    """Config for generate report tool."""
    report_generator: ReportGenerator


class GenerateReportTool(Tool[GenerateReportToolConfig]):
    """Tool wrapper to generate the final report."""

    NAME = "generate_report"
    DESCRIPTION = "Generate the final research report from accumulated findings."

    def __init__(
        self,
        report_generator: ReportGenerator,
        id: Optional[int] = None,
    ) -> None:
        super().__init__(id)
        self.report_generator = report_generator

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def description(self) -> str:
        return self.DESCRIPTION

    def tool_definition(self) -> dict[str, Any]:
        return self.build_tool_definition(
            parameters={
                "question": {
                    "type": "string",
                    "description": "The original research question",
                },
            },
            required=["question"],
        )

    def run(
        self,
        override_kwargs: Optional[GenerateReportToolConfig] = None,
        **llm_kwargs: Any,
    ) -> ToolResponse:
        config = override_kwargs
        report_generator = config.report_generator if config else self.report_generator

        question = llm_kwargs.get("question", "")
        context = llm_kwargs.get("context")
        findings = llm_kwargs.get("findings", [])

        if not question:
            return ToolResponse(llm_response="Error: question is required")

        report = report_generator.generate(
            question=question,
            findings=findings,
            context=context,
        )

        return ToolResponse(
            llm_response=report.full_report,
            rich_response={"report": report},
        )
