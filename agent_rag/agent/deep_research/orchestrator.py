"""Deep Research Orchestrator.

Coordinates the deep research process:
1. Analyze question and generate sub-questions
2. Spawn parallel research agents
3. Collect and synthesize findings
4. Generate comprehensive report

Supports both reasoning models (o1, Claude-3.5-sonnet, etc.) with built-in
chain-of-thought and non-reasoning models requiring explicit think tool usage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time
from typing import Any, Callable, Iterator, Optional

from agent_rag.agent.deep_research.packets import (
    StreamPacket,
    PacketEmitter,
    ResearchPhase,
    ResearchStartPacket,
    ResearchEndPacket,
    ResearchErrorPacket,
    PhaseStartPacket,
    PhaseEndPacket,
    ClarificationQuestionPacket,
    ResearchPlanPacket,
    SubQuestionsPacket,
    CycleStartPacket,
    CycleEndPacket,
    AgentStartPacket,
    AgentEndPacket,
    ThinkStartPacket,
    ThinkContentPacket,
    ThinkEndPacket,
    FindingSummaryPacket,
    FindingKeyFactsPacket,
    FindingSourcesPacket,
    ReportStartPacket,
    ReportTokenPacket,
    ReportEndPacket,
    IntermediateReportStartPacket,
    IntermediateReportContentPacket,
    IntermediateReportEndPacket,
    MetricsPacket,
)
from agent_rag.agent.deep_research.report_generator import (
    ReportConfig,
    ReportGenerator,
    ResearchReport,
    format_report_markdown,
)
from agent_rag.agent.deep_research.research_agent import (
    ResearchAgent,
    ResearchAgentConfig,
    ResearchFindings,
    run_research_agents_parallel,
)
from agent_rag.agent.deep_research.orchestrator_tools import (
    GenerateReportTool,
    ResearchAgentTool,
)
from agent_rag.agent.deep_research.think_tool import ThinkTool
from agent_rag.agent.deep_research.prompts import (
    format_orchestrator_prompt,
    format_clarification_prompt,
    format_research_plan_prompt,
    get_max_orchestrator_cycles,
    QUESTION_ANALYSIS_PROMPT,
)
from agent_rag.citation.accumulator import GlobalCitationAccumulator, create_global_accumulator
from agent_rag.core.config import DeepResearchConfig, LLMConfig, ReasoningEffort
from agent_rag.core.models import ToolCall
from agent_rag.llm.interface import LLM, LLMMessage
from agent_rag.tools.runner import ToolRunner
from agent_rag.tools.registry import ToolRegistry
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class OrchestratorState(Enum):
    """States of the orchestrator."""
    IDLE = "idle"
    ANALYZING = "analyzing"
    RESEARCHING = "researching"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class OrchestratorProgress:
    """Progress update from orchestrator."""
    state: OrchestratorState
    cycle: int
    total_cycles: int
    message: str
    sub_questions: list[str] = field(default_factory=list)
    completed_agents: int = 0
    total_agents: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# Legacy prompts removed - now using prompts.py module
# QUESTION_ANALYSIS_PROMPT is imported from prompts.py


class DeepResearchOrchestrator:
    """
    Orchestrates the deep research process.

    The orchestrator:
    1. Analyzes the research question
    2. Generates focused sub-questions
    3. Spawns parallel research agents
    4. Collects findings using Think tool (non-reasoning models only)
    5. Decides if more research is needed
    6. Generates final report

    Reasoning models (o1, Claude-3.5-sonnet, etc.):
    - Max cycles: 4 (reasoning models have built-in chain-of-thought)
    - Think tool disabled (reasoning happens internally)

    Non-reasoning models:
    - Max cycles: 8 (need more cycles due to explicit thinking)
    - Think tool enabled (mandatory between research cycles)

    Research agents per cycle: 3 (configurable)
    Agent cycles: 3 each
    """

    def __init__(
        self,
        llm: LLM,
        tool_registry: ToolRegistry,
        config: Optional[DeepResearchConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        progress_callback: Optional[Callable[[OrchestratorProgress], None]] = None,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            llm: LLM provider
            tool_registry: Registry of available tools
            config: Deep research configuration
            llm_config: LLM configuration (for reasoning model detection)
            progress_callback: Callback for progress updates
        """
        self.llm = llm
        self.tool_registry = tool_registry
        self.config = config or DeepResearchConfig()
        self.llm_config = llm_config
        self.progress_callback = progress_callback

        # Detect if using reasoning model
        self._is_reasoning_model = self._detect_reasoning_model()

        # Adjust max cycles based on model type
        self._effective_max_cycles = self._compute_max_cycles()

        # Components
        self.think_tool = ThinkTool(llm=llm)
        self.report_generator = ReportGenerator(llm=llm)
        self.orchestrator_registry = ToolRegistry()

        # Only register think tool for non-reasoning models
        if not self._is_reasoning_model:
            self.orchestrator_registry.register(self.think_tool)

        self.orchestrator_registry.register(ResearchAgentTool(
            llm=llm,
            tool_registry=self.tool_registry,
            agent_config=ResearchAgentConfig(
                max_cycles=self._max_agent_cycles(),
            ),
            is_reasoning_model=self._is_reasoning_model,
        ))
        self.orchestrator_registry.register(GenerateReportTool(
            report_generator=self.report_generator,
        ))
        self.orchestrator_runner = ToolRunner(self.orchestrator_registry)

        # State
        self.state = OrchestratorState.IDLE
        self.current_cycle = 0
        self.all_findings: list[ResearchFindings] = []
        self.sub_questions_history: list[str] = []

        # Global citation accumulator for cross-agent citation merging
        self.citation_accumulator = create_global_accumulator(
            fold_by_document=True,
            fold_by_chunk=True,
        )

        logger.info(
            f"Orchestrator initialized: reasoning_model={self._is_reasoning_model}, "
            f"max_cycles={self._effective_max_cycles}"
        )

    def _detect_reasoning_model(self) -> bool:
        """Detect if the LLM is a reasoning model with built-in chain-of-thought."""
        if self.llm_config is not None and self.llm_config.is_reasoning_model:
            return True

        # Check model name patterns for known reasoning models
        model_name = getattr(self.llm, 'model', '') or ''
        if isinstance(model_name, str):
            model_lower = model_name.lower()
            reasoning_patterns = [
                'o1', 'o1-preview', 'o1-mini',
                'claude-3-5-sonnet', 'claude-3.5-sonnet',
                'claude-4', 'gpt-4o-reasoning',
                'deepseek-r1', 'qwen-reasoning',
            ]
            for pattern in reasoning_patterns:
                if pattern in model_lower:
                    logger.info(f"Detected reasoning model: {model_name}")
                    return True

        return False

    def _compute_max_cycles(self) -> int:
        """Compute effective max cycles based on model type."""
        # Use helper function from prompts module
        base_cycles = get_max_orchestrator_cycles(self._is_reasoning_model)

        # Allow config override if explicitly set
        if self.config.max_orchestrator_cycles != 8:  # 8 is default
            return self.config.max_orchestrator_cycles

        return base_cycles

    @property
    def is_reasoning_model(self) -> bool:
        """Whether the orchestrator is using a reasoning model."""
        return self._is_reasoning_model

    @property
    def max_cycles(self) -> int:
        """Effective maximum orchestrator cycles."""
        return self._effective_max_cycles

    def _num_research_agents(self) -> int:
        """Resolve configured number of research agents."""
        return self.config.num_research_agents or self.config.max_research_agents

    def _max_agent_cycles(self) -> int:
        """Resolve configured max cycles per research agent."""
        return self.config.max_agent_cycles or self.config.max_research_cycles

    def reset(self) -> None:
        """Reset orchestrator state."""
        self.state = OrchestratorState.IDLE
        self.current_cycle = 0
        self.all_findings = []
        self.sub_questions_history = []
        self.citation_accumulator.reset()

    def research(
        self,
        question: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ResearchReport:
        """
        Conduct deep research on a question.

        Args:
            question: Research question
            context: Optional context for tools

        Returns:
            Comprehensive research report
        """
        self.reset()
        self._notify_progress(
            OrchestratorState.ANALYZING,
            "Analyzing research question...",
        )

        clarification_question = None
        research_plan = None

        try:
            if not self.config.skip_clarification:
                clarification_question = self._maybe_clarify(question)

            research_plan, initial_sub_questions = self._generate_research_plan(question)

            # Main research loop
            pending_sub_questions = initial_sub_questions or []
            while self.current_cycle < self._effective_max_cycles:
                self.current_cycle += 1

                if not pending_sub_questions:
                    break

                sub_questions = pending_sub_questions[: self._num_research_agents()]
                pending_sub_questions = pending_sub_questions[self._num_research_agents():]

                if not sub_questions:
                    logger.warning("No sub-questions generated, ending research")
                    break

                self.sub_questions_history.extend(sub_questions)

                # Run research agents
                self._notify_progress(
                    OrchestratorState.RESEARCHING,
                    f"Cycle {self.current_cycle}: Researching {len(sub_questions)} sub-questions...",
                    sub_questions=sub_questions,
                )

                findings = self._run_research_cycle_with_tools(
                    sub_questions=sub_questions,
                    main_question=question,
                    context=context,
                )

                # Register findings with global citation accumulator
                for finding in findings:
                    agent_id = finding.metadata.get("agent_id", f"agent_{self.current_cycle}")
                    self.citation_accumulator.register_agent_citations(
                        agent_id=agent_id,
                        chunks=finding.sources,
                    )

                self.all_findings.extend(findings)

                # Think about findings
                think_result = self._think_about_findings(question)

                if think_result.has_sufficient_info:
                    logger.info("Sufficient information gathered, generating report")
                    break

                if not think_result.refined_queries:
                    logger.info("No more refined queries, ending research")
                    break
                pending_sub_questions.extend(think_result.refined_queries)

            # Generate report
            self._notify_progress(
                OrchestratorState.SYNTHESIZING,
                "Synthesizing findings into report...",
            )

            report = self._generate_report_with_tool(
                question=question,
                findings=self.all_findings,
                context=context,
            )
            report.research_plan = research_plan
            report.clarification_question = clarification_question

            self._notify_progress(
                OrchestratorState.COMPLETE,
                "Research complete!",
            )

            return report

        except Exception as e:
            self.state = OrchestratorState.FAILED
            self._notify_progress(
                OrchestratorState.FAILED,
                f"Research failed: {str(e)}",
            )
            raise

    def research_stream(
        self,
        question: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Iterator[tuple[str, Optional[ResearchReport]]]:
        """
        Conduct deep research with streaming output.

        Args:
            question: Research question
            context: Optional context

        Yields:
            Tuples of (update_text, final_report) where report is None until complete
        """
        self.reset()

        yield ("Analyzing research question...\n", None)

        try:
            clarification_question = None
            if not self.config.skip_clarification:
                clarification_question = self._maybe_clarify(question)
                if clarification_question:
                    yield (f"\nClarification: {clarification_question}\n", None)

            research_plan, initial_sub_questions = self._generate_research_plan(question)
            if research_plan:
                yield (f"\n## Research Plan\n{research_plan}\n", None)

            pending_sub_questions = initial_sub_questions or []

            while self.current_cycle < self._effective_max_cycles:
                self.current_cycle += 1

                if not pending_sub_questions:
                    break

                sub_questions = pending_sub_questions[: self._num_research_agents()]
                pending_sub_questions = pending_sub_questions[self._num_research_agents():]

                if not sub_questions:
                    break

                self.sub_questions_history.extend(sub_questions)

                yield (f"\n## Cycle {self.current_cycle}\n", None)
                yield (f"Researching {len(sub_questions)} sub-questions:\n", None)
                for i, sq in enumerate(sub_questions, 1):
                    yield (f"  {i}. {sq}\n", None)

                findings = self._run_research_cycle_with_tools(
                    sub_questions=sub_questions,
                    main_question=question,
                    context=context,
                )

                # Register findings with global citation accumulator
                for finding in findings:
                    agent_id = finding.metadata.get("agent_id", f"agent_{self.current_cycle}")
                    self.citation_accumulator.register_agent_citations(
                        agent_id=agent_id,
                        chunks=finding.sources,
                    )

                self.all_findings.extend(findings)

                # Report findings
                for finding in findings:
                    yield (f"\n### {finding.sub_question}\n", None)
                    yield (f"{finding.summary[:200]}...\n", None)

                think_result = self._think_about_findings(question)

                if think_result.has_sufficient_info:
                    yield ("\nSufficient information gathered.\n", None)
                    break
                if think_result.refined_queries:
                    pending_sub_questions.extend(think_result.refined_queries)

            # Generate report with streaming
            yield ("\n---\n\n# Generating Report\n\n", None)

            report = None
            for token, final_report in self.report_generator.generate_stream(
                question=question,
                findings=self.all_findings,
                context=context,
            ):
                if token:
                    yield (token, None)
                if final_report:
                    report = final_report

            if report:
                report.research_plan = research_plan
                report.clarification_question = clarification_question

            yield ("", report)

        except Exception as e:
            yield (f"\nError: {str(e)}\n", None)
            raise

    def research_stream_packets(
        self,
        question: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Iterator[StreamPacket]:
        """
        Conduct deep research with structured packet streaming.

        This method yields structured StreamPacket objects for real-time UI updates.
        Each packet contains typed data about research progress, findings, and reports.

        Args:
            question: Research question
            context: Optional context

        Yields:
            StreamPacket objects with typed research events
        """
        self.reset()
        start_time = time.time()
        total_sources = 0

        # Emit research start
        yield ResearchStartPacket(
            question=question,
            config={
                "max_cycles": self._effective_max_cycles,
                "num_agents": self._num_research_agents(),
                "is_reasoning_model": self._is_reasoning_model,
            },
        )

        try:
            # Phase: Initialization
            yield PhaseStartPacket(
                phase=ResearchPhase.INITIALIZATION,
                message="Analyzing research question...",
            )
            phase_start = time.time()

            # Phase: Clarification
            clarification_question = None
            if not self.config.skip_clarification:
                yield PhaseStartPacket(
                    phase=ResearchPhase.CLARIFICATION,
                    message="Checking if clarification is needed...",
                )
                clarification_question = self._maybe_clarify(question)
                if clarification_question:
                    yield ClarificationQuestionPacket(
                        questions=[clarification_question],
                    )
                yield PhaseEndPacket(
                    phase=ResearchPhase.CLARIFICATION,
                    duration_seconds=time.time() - phase_start,
                )

            # Phase: Planning
            yield PhaseStartPacket(
                phase=ResearchPhase.PLANNING,
                message="Generating research plan...",
            )
            plan_start = time.time()
            research_plan, initial_sub_questions = self._generate_research_plan(question)
            yield ResearchPlanPacket(
                plan=research_plan or "",
                sub_questions=initial_sub_questions,
            )
            yield PhaseEndPacket(
                phase=ResearchPhase.PLANNING,
                duration_seconds=time.time() - plan_start,
            )

            # Phase: Research
            yield PhaseStartPacket(
                phase=ResearchPhase.RESEARCH,
                message="Beginning research cycles...",
            )
            research_start = time.time()
            pending_sub_questions = initial_sub_questions or []

            while self.current_cycle < self._effective_max_cycles:
                self.current_cycle += 1

                if not pending_sub_questions:
                    break

                sub_questions = pending_sub_questions[: self._num_research_agents()]
                pending_sub_questions = pending_sub_questions[self._num_research_agents():]

                if not sub_questions:
                    break

                self.sub_questions_history.extend(sub_questions)

                # Emit cycle start
                yield CycleStartPacket(
                    cycle=self.current_cycle,
                    max_cycles=self._effective_max_cycles,
                    sub_questions=sub_questions,
                )

                # Emit sub-questions for this cycle
                yield SubQuestionsPacket(
                    sub_questions=sub_questions,
                    cycle=self.current_cycle,
                )

                # Emit agent start packets for each sub-question
                for i, sq in enumerate(sub_questions):
                    agent_id = f"agent_{self.current_cycle}_{i}"
                    yield AgentStartPacket(
                        agent_id=agent_id,
                        sub_question=sq,
                        cycle=self.current_cycle,
                    )

                # Run research cycle
                findings = self._run_research_cycle_with_tools(
                    sub_questions=sub_questions,
                    main_question=question,
                    context=context,
                )

                # Register findings with global citation accumulator
                for i, finding in enumerate(findings):
                    agent_id = finding.metadata.get("agent_id", f"agent_{self.current_cycle}_{i}")
                    self.citation_accumulator.register_agent_citations(
                        agent_id=agent_id,
                        chunks=finding.sources,
                    )
                    total_sources += len(finding.sources)

                    # Emit agent end packet
                    yield AgentEndPacket(
                        agent_id=agent_id,
                        success=finding.confidence > 0,
                        summary=finding.summary[:200] if finding.summary else "",
                        num_sources=len(finding.sources),
                        confidence=finding.confidence,
                    )

                    # Emit finding summary
                    yield FindingSummaryPacket(
                        agent_id=agent_id,
                        sub_question=finding.sub_question,
                        summary=finding.summary,
                    )

                    # Emit key facts
                    if finding.key_facts:
                        yield FindingKeyFactsPacket(
                            agent_id=agent_id,
                            key_facts=finding.key_facts,
                        )

                    # Emit sources
                    if finding.sources:
                        yield FindingSourcesPacket(
                            agent_id=agent_id,
                            sources=[
                                {
                                    "document_id": s.document_id,
                                    "title": s.title,
                                    "source_type": s.source_type,
                                }
                                for s in finding.sources
                            ],
                        )

                self.all_findings.extend(findings)

                # Think about findings
                yield ThinkStartPacket(question=question)
                think_result = self._think_about_findings(question)
                yield ThinkEndPacket(
                    refined_queries=think_result.refined_queries,
                    has_sufficient_info=think_result.has_sufficient_info,
                )

                # Emit intermediate report after each cycle
                yield IntermediateReportStartPacket(cycle=self.current_cycle)
                intermediate_summary = self._generate_intermediate_summary()
                yield IntermediateReportContentPacket(
                    cycle=self.current_cycle,
                    content=intermediate_summary,
                    is_streaming=False,
                )
                yield IntermediateReportEndPacket(
                    cycle=self.current_cycle,
                    sources_cited=total_sources,
                )

                # Emit cycle end
                yield CycleEndPacket(
                    cycle=self.current_cycle,
                    agents_completed=len(findings),
                    sources_found=sum(len(f.sources) for f in findings),
                    has_more_queries=bool(think_result.refined_queries),
                )

                # Check if we have sufficient info
                if think_result.has_sufficient_info:
                    break
                if think_result.refined_queries:
                    pending_sub_questions.extend(think_result.refined_queries)

            yield PhaseEndPacket(
                phase=ResearchPhase.RESEARCH,
                duration_seconds=time.time() - research_start,
            )

            # Phase: Synthesis
            yield PhaseStartPacket(
                phase=ResearchPhase.SYNTHESIS,
                message="Synthesizing findings into report...",
            )
            synthesis_start = time.time()

            yield ReportStartPacket(
                total_findings=len(self.all_findings),
                total_sources=total_sources,
            )

            # Stream report tokens
            report = None
            word_count = 0
            for token, final_report in self.report_generator.generate_stream(
                question=question,
                findings=self.all_findings,
                context=context,
            ):
                if token:
                    yield ReportTokenPacket(token=token)
                    word_count += len(token.split())
                if final_report:
                    report = final_report

            if report:
                report.research_plan = research_plan
                report.clarification_question = clarification_question

            yield ReportEndPacket(
                total_citations=len(self.citation_accumulator.get_all_citations()),
                word_count=word_count,
                confidence=self._compute_overall_confidence(),
            )

            yield PhaseEndPacket(
                phase=ResearchPhase.SYNTHESIS,
                duration_seconds=time.time() - synthesis_start,
            )

            # Emit final metrics
            yield MetricsPacket(
                metrics={
                    "total_cycles": self.current_cycle,
                    "total_findings": len(self.all_findings),
                    "total_sources": total_sources,
                    "total_citations": len(self.citation_accumulator.get_all_citations()),
                    "is_reasoning_model": self._is_reasoning_model,
                    "duration_seconds": time.time() - start_time,
                },
            )

            # Emit research end
            yield ResearchEndPacket(
                success=True,
                duration_seconds=time.time() - start_time,
                total_cycles=self.current_cycle,
                total_sources=total_sources,
            )

        except Exception as e:
            yield ResearchErrorPacket(
                error=str(e),
                error_type=type(e).__name__,
                recoverable=False,
            )
            raise

    def _generate_intermediate_summary(self) -> str:
        """Generate an intermediate summary of current findings."""
        if not self.all_findings:
            return "No findings yet."

        summaries = []
        summaries.append(f"## Research Progress - Cycle {self.current_cycle}")
        summaries.append(f"\n**Findings so far**: {len(self.all_findings)} sub-questions investigated\n")

        for i, finding in enumerate(self.all_findings[-3:], 1):  # Last 3 findings
            summaries.append(f"### {finding.sub_question}")
            summaries.append(finding.summary[:300] + "..." if len(finding.summary) > 300 else finding.summary)
            if finding.key_facts:
                summaries.append("\n**Key Facts:**")
                for fact in finding.key_facts[:3]:
                    summaries.append(f"- {fact}")
            summaries.append("")

        return "\n".join(summaries)

    def _compute_overall_confidence(self) -> float:
        """Compute overall confidence from all findings."""
        if not self.all_findings:
            return 0.0

        confidences = [f.confidence for f in self.all_findings if f.confidence > 0]
        if not confidences:
            return 0.0

        return sum(confidences) / len(confidences)

    def _generate_initial_subquestions(self, question: str) -> list[str]:
        """Generate initial sub-questions from main question."""
        prompt = QUESTION_ANALYSIS_PROMPT.format(
            question=question,
            num_subquestions=self._num_research_agents(),
        )

        messages = [LLMMessage(role="user", content=prompt)]
        # Use ReasoningEffort.LOW for sub-question generation (auxiliary task)
        response = self.llm.generate(messages, reasoning_effort=ReasoningEffort.LOW)

        return self._parse_subquestions(response.content)

    def _maybe_clarify(self, question: str) -> Optional[str]:
        """Return a clarification question if needed."""
        prompt = format_clarification_prompt(
            question=question,
            is_reasoning_model=self._is_reasoning_model,
        )
        messages = [LLMMessage(role="user", content=prompt)]
        # Use ReasoningEffort.LOW for clarification (auxiliary task)
        response = self.llm.generate(messages, reasoning_effort=ReasoningEffort.LOW)
        text = response.content.strip()
        if text.upper() == "NONE":
            return None
        return text or None

    def _generate_research_plan(self, question: str) -> tuple[str, list[str]]:
        """Generate a research plan and initial sub-questions."""
        prompt = format_research_plan_prompt(
            question=question,
            is_reasoning_model=self._is_reasoning_model,
        )
        messages = [LLMMessage(role="user", content=prompt)]
        # Use ReasoningEffort.LOW for research planning (auxiliary task)
        response = self.llm.generate(messages, reasoning_effort=ReasoningEffort.LOW)
        plan_text = response.content.strip()
        sub_questions = self._parse_subquestions(plan_text)
        if not sub_questions:
            sub_questions = self._generate_initial_subquestions(question)
        return plan_text, sub_questions

    def _generate_refined_subquestions(self, question: str) -> list[str]:
        """Generate refined sub-questions based on current findings."""
        think_result = self.think_tool.think(
            question=question,
            current_findings=self._summarize_findings(),
            search_history=self.sub_questions_history,
            max_queries=self._num_research_agents(),
        )

        return think_result.refined_queries

    def _run_research_cycle(
        self,
        sub_questions: list[str],
        main_question: str,
        context: Optional[dict[str, Any]] = None,
    ) -> list[ResearchFindings]:
        """Run a cycle of parallel research agents."""
        # Create agents
        agents = []
        for i, sq in enumerate(sub_questions):
            agent = ResearchAgent(
                llm=self.llm,
                tool_registry=self.tool_registry,
                config=ResearchAgentConfig(
                    max_cycles=self._max_agent_cycles(),
                ),
                agent_id=f"agent_{self.current_cycle}_{i}",
            )
            agents.append(agent)

        # Run in parallel
        findings = run_research_agents_parallel(
            agents=agents,
            sub_questions=sub_questions,
            main_question=main_question,
            context=context,
            max_workers=self._num_research_agents(),
        )

        return findings

    def _run_research_cycle_with_tools(
        self,
        sub_questions: list[str],
        main_question: str,
        context: Optional[dict[str, Any]] = None,
    ) -> list[ResearchFindings]:
        """Run a cycle using research_agent tool calls."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        findings: list[ResearchFindings] = []
        tool_calls: list[ToolCall] = []
        for i, sq in enumerate(sub_questions):
            tool_calls.append(ToolCall(
                id=f"research_{self.current_cycle}_{i}",
                name="research_agent",
                arguments={
                    "sub_question": sq,
                    "main_question": main_question,
                },
            ))

        def run_tool(tc: ToolCall) -> ResearchFindings:
            result = self.orchestrator_runner.run(tc, context=context)
            tool_findings = result.rich_response.get("findings") if result.rich_response else None
            if isinstance(tool_findings, ResearchFindings):
                return tool_findings
            return ResearchFindings(
                sub_question=tc.arguments.get("sub_question", ""),
                summary=result.llm_response,
                key_facts=[],
                sources=[],
                confidence=0.0,
                search_queries_used=[],
            )

        with ThreadPoolExecutor(max_workers=self._num_research_agents()) as executor:
            futures = {executor.submit(run_tool, tc): tc for tc in tool_calls}
            for future in as_completed(futures):
                try:
                    findings.append(future.result())
                except Exception as e:
                    tc = futures[future]
                    logger.error(f"Research tool failed: {tc.name}: {e}")
                    findings.append(ResearchFindings(
                        sub_question=tc.arguments.get("sub_question", ""),
                        summary=f"Research failed: {str(e)}",
                        key_facts=[],
                        sources=[],
                        confidence=0.0,
                        search_queries_used=[],
                    ))

        return findings

    def _generate_report_with_tool(
        self,
        question: str,
        findings: list[ResearchFindings],
        context: Optional[dict[str, Any]] = None,
    ) -> ResearchReport:
        """Generate final report using tool wrapper."""
        tool_call = ToolCall(
            id=f"report_{self.current_cycle}",
            name="generate_report",
            arguments={"question": question, "findings": findings},
        )
        result = self.orchestrator_runner.run(tool_call, context=context)
        report = result.rich_response.get("report") if result.rich_response else None
        if isinstance(report, ResearchReport):
            return report
        return self.report_generator.generate(question=question, findings=findings, context=context)

    def _think_about_findings(self, question: str) -> Any:
        """Use think tool to analyze current findings."""
        return self.think_tool.think(
            question=question,
            current_findings=self._summarize_findings(),
            search_history=self.sub_questions_history,
        )

    def _summarize_findings(self) -> str:
        """Summarize all current findings."""
        if not self.all_findings:
            return "No findings yet."

        summaries = []
        for finding in self.all_findings:
            summaries.append(f"**{finding.sub_question}**: {finding.summary[:200]}...")

        return "\n\n".join(summaries)

    def _parse_subquestions(self, text: str) -> list[str]:
        """Parse sub-questions from LLM response."""
        import re

        questions = []
        for line in text.split('\n'):
            line = line.strip()
            # Match numbered questions
            match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
            if match:
                question = match.group(1).strip()
                if question:
                    questions.append(question)

        return questions[:self._num_research_agents()]

    def _notify_progress(
        self,
        state: OrchestratorState,
        message: str,
        sub_questions: Optional[list[str]] = None,
    ) -> None:
        """Notify progress callback."""
        self.state = state

        if self.progress_callback:
            progress = OrchestratorProgress(
                state=state,
                cycle=self.current_cycle,
                total_cycles=self._effective_max_cycles,
                message=message,
                sub_questions=sub_questions or [],
                completed_agents=len(self.all_findings),
                total_agents=len(self.sub_questions_history),
                metadata={
                    "is_reasoning_model": self._is_reasoning_model,
                },
            )
            self.progress_callback(progress)


def create_deep_research_agent(
    llm: LLM,
    tool_registry: ToolRegistry,
    config: Optional[DeepResearchConfig] = None,
    llm_config: Optional[LLMConfig] = None,
) -> DeepResearchOrchestrator:
    """Factory function to create a deep research orchestrator.

    Args:
        llm: LLM provider
        tool_registry: Registry of available tools
        config: Deep research configuration
        llm_config: LLM configuration (for reasoning model detection)

    Returns:
        Configured DeepResearchOrchestrator instance
    """
    return DeepResearchOrchestrator(
        llm=llm,
        tool_registry=tool_registry,
        config=config,
        llm_config=llm_config,
    )
