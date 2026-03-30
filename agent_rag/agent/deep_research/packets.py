"""Streaming Packet types for Deep Research.

Defines structured packet types for real-time streaming of research progress,
intermediate results, and final reports.

Reference: backend/onyx/agents/agent_search/deep/initial/answer_initial_stream_packets.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union


class PacketType(str, Enum):
    """Types of streaming packets."""
    # Lifecycle events
    RESEARCH_START = "research_start"
    RESEARCH_END = "research_end"
    RESEARCH_ERROR = "research_error"

    # Phase transitions
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"

    # Clarification
    CLARIFICATION_QUESTION = "clarification_question"
    CLARIFICATION_RESPONSE = "clarification_response"

    # Planning
    RESEARCH_PLAN = "research_plan"
    SUB_QUESTIONS = "sub_questions"

    # Research cycle
    CYCLE_START = "cycle_start"
    CYCLE_END = "cycle_end"

    # Agent events
    AGENT_START = "agent_start"
    AGENT_PROGRESS = "agent_progress"
    AGENT_SEARCH = "agent_search"
    AGENT_RESULT = "agent_result"
    AGENT_END = "agent_end"

    # Think tool
    THINK_START = "think_start"
    THINK_CONTENT = "think_content"
    THINK_END = "think_end"

    # Findings
    FINDING_SUMMARY = "finding_summary"
    FINDING_KEY_FACTS = "finding_key_facts"
    FINDING_SOURCES = "finding_sources"

    # Report generation
    REPORT_START = "report_start"
    REPORT_SECTION = "report_section"
    REPORT_TOKEN = "report_token"
    REPORT_CITATION = "report_citation"
    REPORT_END = "report_end"

    # Intermediate reports
    INTERMEDIATE_REPORT_START = "intermediate_report_start"
    INTERMEDIATE_REPORT_CONTENT = "intermediate_report_content"
    INTERMEDIATE_REPORT_END = "intermediate_report_end"

    # Debug/monitoring
    DEBUG_INFO = "debug_info"
    METRICS = "metrics"


class ResearchPhase(str, Enum):
    """Research phases."""
    INITIALIZATION = "initialization"
    CLARIFICATION = "clarification"
    PLANNING = "planning"
    RESEARCH = "research"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class BasePacket:
    """Base class for all streaming packets."""
    packet_type: PacketType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.packet_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self._get_data(),
            "metadata": self.metadata,
        }

    def _get_data(self) -> dict[str, Any]:
        """Get packet-specific data. Override in subclasses."""
        return {}


# =============================================================================
# LIFECYCLE PACKETS
# =============================================================================

@dataclass
class ResearchStartPacket(BasePacket):
    """Emitted when research begins."""
    question: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.packet_type = PacketType.RESEARCH_START

    def _get_data(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "config": self.config,
        }


@dataclass
class ResearchEndPacket(BasePacket):
    """Emitted when research completes."""
    success: bool = True
    duration_seconds: float = 0.0
    total_cycles: int = 0
    total_sources: int = 0

    def __post_init__(self) -> None:
        self.packet_type = PacketType.RESEARCH_END

    def _get_data(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "total_cycles": self.total_cycles,
            "total_sources": self.total_sources,
        }


@dataclass
class ResearchErrorPacket(BasePacket):
    """Emitted when research fails."""
    error: str = ""
    error_type: str = ""
    recoverable: bool = False

    def __post_init__(self) -> None:
        self.packet_type = PacketType.RESEARCH_ERROR

    def _get_data(self) -> dict[str, Any]:
        return {
            "error": self.error,
            "error_type": self.error_type,
            "recoverable": self.recoverable,
        }


# =============================================================================
# PHASE PACKETS
# =============================================================================

@dataclass
class PhaseStartPacket(BasePacket):
    """Emitted when a research phase begins."""
    phase: ResearchPhase = ResearchPhase.INITIALIZATION
    message: str = ""

    def __post_init__(self) -> None:
        self.packet_type = PacketType.PHASE_START

    def _get_data(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "message": self.message,
        }


@dataclass
class PhaseEndPacket(BasePacket):
    """Emitted when a research phase ends."""
    phase: ResearchPhase = ResearchPhase.INITIALIZATION
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.packet_type = PacketType.PHASE_END

    def _get_data(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "duration_seconds": self.duration_seconds,
        }


# =============================================================================
# CLARIFICATION PACKETS
# =============================================================================

@dataclass
class ClarificationQuestionPacket(BasePacket):
    """Emitted when clarification is needed."""
    questions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.packet_type = PacketType.CLARIFICATION_QUESTION

    def _get_data(self) -> dict[str, Any]:
        return {"questions": self.questions}


# =============================================================================
# PLANNING PACKETS
# =============================================================================

@dataclass
class ResearchPlanPacket(BasePacket):
    """Emitted with the research plan."""
    plan: str = ""
    sub_questions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.packet_type = PacketType.RESEARCH_PLAN

    def _get_data(self) -> dict[str, Any]:
        return {
            "plan": self.plan,
            "sub_questions": self.sub_questions,
        }


@dataclass
class SubQuestionsPacket(BasePacket):
    """Emitted with sub-questions to research."""
    sub_questions: list[str] = field(default_factory=list)
    cycle: int = 0

    def __post_init__(self) -> None:
        self.packet_type = PacketType.SUB_QUESTIONS

    def _get_data(self) -> dict[str, Any]:
        return {
            "sub_questions": self.sub_questions,
            "cycle": self.cycle,
        }


# =============================================================================
# RESEARCH CYCLE PACKETS
# =============================================================================

@dataclass
class CycleStartPacket(BasePacket):
    """Emitted when a research cycle begins."""
    cycle: int = 0
    max_cycles: int = 0
    sub_questions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.packet_type = PacketType.CYCLE_START

    def _get_data(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "max_cycles": self.max_cycles,
            "sub_questions": self.sub_questions,
        }


@dataclass
class CycleEndPacket(BasePacket):
    """Emitted when a research cycle ends."""
    cycle: int = 0
    agents_completed: int = 0
    sources_found: int = 0
    has_more_queries: bool = False

    def __post_init__(self) -> None:
        self.packet_type = PacketType.CYCLE_END

    def _get_data(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "agents_completed": self.agents_completed,
            "sources_found": self.sources_found,
            "has_more_queries": self.has_more_queries,
        }


# =============================================================================
# AGENT PACKETS
# =============================================================================

@dataclass
class AgentStartPacket(BasePacket):
    """Emitted when a research agent starts."""
    agent_id: str = ""
    sub_question: str = ""
    cycle: int = 0

    def __post_init__(self) -> None:
        self.packet_type = PacketType.AGENT_START

    def _get_data(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "sub_question": self.sub_question,
            "cycle": self.cycle,
        }


@dataclass
class AgentProgressPacket(BasePacket):
    """Emitted during agent progress."""
    agent_id: str = ""
    message: str = ""
    progress_percent: float = 0.0

    def __post_init__(self) -> None:
        self.packet_type = PacketType.AGENT_PROGRESS

    def _get_data(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "message": self.message,
            "progress_percent": self.progress_percent,
        }


@dataclass
class AgentSearchPacket(BasePacket):
    """Emitted when agent performs a search."""
    agent_id: str = ""
    query: str = ""
    search_type: str = "internal"  # internal, web, url

    def __post_init__(self) -> None:
        self.packet_type = PacketType.AGENT_SEARCH

    def _get_data(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "query": self.query,
            "search_type": self.search_type,
        }


@dataclass
class AgentResultPacket(BasePacket):
    """Emitted when agent gets search results."""
    agent_id: str = ""
    num_results: int = 0
    query: str = ""

    def __post_init__(self) -> None:
        self.packet_type = PacketType.AGENT_RESULT

    def _get_data(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "num_results": self.num_results,
            "query": self.query,
        }


@dataclass
class AgentEndPacket(BasePacket):
    """Emitted when agent completes."""
    agent_id: str = ""
    success: bool = True
    summary: str = ""
    num_sources: int = 0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        self.packet_type = PacketType.AGENT_END

    def _get_data(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "success": self.success,
            "summary": self.summary,
            "num_sources": self.num_sources,
            "confidence": self.confidence,
        }


# =============================================================================
# THINK TOOL PACKETS
# =============================================================================

@dataclass
class ThinkStartPacket(BasePacket):
    """Emitted when think tool starts."""
    question: str = ""

    def __post_init__(self) -> None:
        self.packet_type = PacketType.THINK_START

    def _get_data(self) -> dict[str, Any]:
        return {"question": self.question}


@dataclass
class ThinkContentPacket(BasePacket):
    """Emitted during think tool reasoning."""
    content: str = ""
    is_streaming: bool = True

    def __post_init__(self) -> None:
        self.packet_type = PacketType.THINK_CONTENT

    def _get_data(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "is_streaming": self.is_streaming,
        }


@dataclass
class ThinkEndPacket(BasePacket):
    """Emitted when think tool completes."""
    refined_queries: list[str] = field(default_factory=list)
    has_sufficient_info: bool = False

    def __post_init__(self) -> None:
        self.packet_type = PacketType.THINK_END

    def _get_data(self) -> dict[str, Any]:
        return {
            "refined_queries": self.refined_queries,
            "has_sufficient_info": self.has_sufficient_info,
        }


# =============================================================================
# FINDING PACKETS
# =============================================================================

@dataclass
class FindingSummaryPacket(BasePacket):
    """Emitted with finding summary."""
    agent_id: str = ""
    sub_question: str = ""
    summary: str = ""

    def __post_init__(self) -> None:
        self.packet_type = PacketType.FINDING_SUMMARY

    def _get_data(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "sub_question": self.sub_question,
            "summary": self.summary,
        }


@dataclass
class FindingKeyFactsPacket(BasePacket):
    """Emitted with key facts from findings."""
    agent_id: str = ""
    key_facts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.packet_type = PacketType.FINDING_KEY_FACTS

    def _get_data(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "key_facts": self.key_facts,
        }


@dataclass
class FindingSourcesPacket(BasePacket):
    """Emitted with sources from findings."""
    agent_id: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.packet_type = PacketType.FINDING_SOURCES

    def _get_data(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "sources": self.sources,
        }


# =============================================================================
# REPORT PACKETS
# =============================================================================

@dataclass
class ReportStartPacket(BasePacket):
    """Emitted when report generation starts."""
    total_findings: int = 0
    total_sources: int = 0

    def __post_init__(self) -> None:
        self.packet_type = PacketType.REPORT_START

    def _get_data(self) -> dict[str, Any]:
        return {
            "total_findings": self.total_findings,
            "total_sources": self.total_sources,
        }


@dataclass
class ReportSectionPacket(BasePacket):
    """Emitted when a report section is generated."""
    section_name: str = ""
    section_content: str = ""

    def __post_init__(self) -> None:
        self.packet_type = PacketType.REPORT_SECTION

    def _get_data(self) -> dict[str, Any]:
        return {
            "section_name": self.section_name,
            "section_content": self.section_content,
        }


@dataclass
class ReportTokenPacket(BasePacket):
    """Emitted during streaming report generation."""
    token: str = ""
    section: Optional[str] = None

    def __post_init__(self) -> None:
        self.packet_type = PacketType.REPORT_TOKEN

    def _get_data(self) -> dict[str, Any]:
        data = {"token": self.token}
        if self.section:
            data["section"] = self.section
        return data


@dataclass
class ReportCitationPacket(BasePacket):
    """Emitted when a citation is added to report."""
    citation_id: int = 0
    document_title: Optional[str] = None
    source_type: Optional[str] = None

    def __post_init__(self) -> None:
        self.packet_type = PacketType.REPORT_CITATION

    def _get_data(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "document_title": self.document_title,
            "source_type": self.source_type,
        }


@dataclass
class ReportEndPacket(BasePacket):
    """Emitted when report generation completes."""
    total_citations: int = 0
    word_count: int = 0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        self.packet_type = PacketType.REPORT_END

    def _get_data(self) -> dict[str, Any]:
        return {
            "total_citations": self.total_citations,
            "word_count": self.word_count,
            "confidence": self.confidence,
        }


# =============================================================================
# INTERMEDIATE REPORT PACKETS
# =============================================================================

@dataclass
class IntermediateReportStartPacket(BasePacket):
    """Emitted when intermediate report starts."""
    cycle: int = 0

    def __post_init__(self) -> None:
        self.packet_type = PacketType.INTERMEDIATE_REPORT_START

    def _get_data(self) -> dict[str, Any]:
        return {"cycle": self.cycle}


@dataclass
class IntermediateReportContentPacket(BasePacket):
    """Emitted with intermediate report content."""
    cycle: int = 0
    content: str = ""
    is_streaming: bool = True

    def __post_init__(self) -> None:
        self.packet_type = PacketType.INTERMEDIATE_REPORT_CONTENT

    def _get_data(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "content": self.content,
            "is_streaming": self.is_streaming,
        }


@dataclass
class IntermediateReportEndPacket(BasePacket):
    """Emitted when intermediate report completes."""
    cycle: int = 0
    sources_cited: int = 0

    def __post_init__(self) -> None:
        self.packet_type = PacketType.INTERMEDIATE_REPORT_END

    def _get_data(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "sources_cited": self.sources_cited,
        }


# =============================================================================
# DEBUG/METRICS PACKETS
# =============================================================================

@dataclass
class DebugInfoPacket(BasePacket):
    """Emitted with debug information."""
    component: str = ""
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.packet_type = PacketType.DEBUG_INFO

    def _get_data(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "message": self.message,
            "data": self.data,
        }


@dataclass
class MetricsPacket(BasePacket):
    """Emitted with metrics data."""
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.packet_type = PacketType.METRICS

    def _get_data(self) -> dict[str, Any]:
        return {"metrics": self.metrics}


# =============================================================================
# TYPE UNIONS
# =============================================================================

# All packet types for type hints
StreamPacket = Union[
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
    AgentProgressPacket,
    AgentSearchPacket,
    AgentResultPacket,
    AgentEndPacket,
    ThinkStartPacket,
    ThinkContentPacket,
    ThinkEndPacket,
    FindingSummaryPacket,
    FindingKeyFactsPacket,
    FindingSourcesPacket,
    ReportStartPacket,
    ReportSectionPacket,
    ReportTokenPacket,
    ReportCitationPacket,
    ReportEndPacket,
    IntermediateReportStartPacket,
    IntermediateReportContentPacket,
    IntermediateReportEndPacket,
    DebugInfoPacket,
    MetricsPacket,
]


# =============================================================================
# PACKET EMITTER
# =============================================================================

class PacketEmitter:
    """
    Helper class for emitting packets during research.

    Provides convenient methods for common packet patterns.
    """

    def __init__(self) -> None:
        self.packets: list[StreamPacket] = []

    def emit(self, packet: StreamPacket) -> StreamPacket:
        """Emit a packet and return it."""
        self.packets.append(packet)
        return packet

    def research_start(self, question: str, config: Optional[dict] = None) -> ResearchStartPacket:
        """Emit research start packet."""
        return self.emit(ResearchStartPacket(
            question=question,
            config=config or {},
        ))

    def research_end(
        self,
        success: bool,
        duration: float,
        cycles: int,
        sources: int,
    ) -> ResearchEndPacket:
        """Emit research end packet."""
        return self.emit(ResearchEndPacket(
            success=success,
            duration_seconds=duration,
            total_cycles=cycles,
            total_sources=sources,
        ))

    def research_error(self, error: str, error_type: str = "", recoverable: bool = False) -> ResearchErrorPacket:
        """Emit research error packet."""
        return self.emit(ResearchErrorPacket(
            error=error,
            error_type=error_type,
            recoverable=recoverable,
        ))

    def phase_start(self, phase: ResearchPhase, message: str = "") -> PhaseStartPacket:
        """Emit phase start packet."""
        return self.emit(PhaseStartPacket(phase=phase, message=message))

    def phase_end(self, phase: ResearchPhase, duration: float = 0.0) -> PhaseEndPacket:
        """Emit phase end packet."""
        return self.emit(PhaseEndPacket(phase=phase, duration_seconds=duration))

    def cycle_start(
        self,
        cycle: int,
        max_cycles: int,
        sub_questions: list[str],
    ) -> CycleStartPacket:
        """Emit cycle start packet."""
        return self.emit(CycleStartPacket(
            cycle=cycle,
            max_cycles=max_cycles,
            sub_questions=sub_questions,
        ))

    def agent_start(self, agent_id: str, sub_question: str, cycle: int) -> AgentStartPacket:
        """Emit agent start packet."""
        return self.emit(AgentStartPacket(
            agent_id=agent_id,
            sub_question=sub_question,
            cycle=cycle,
        ))

    def agent_end(
        self,
        agent_id: str,
        success: bool,
        summary: str,
        sources: int,
        confidence: float,
    ) -> AgentEndPacket:
        """Emit agent end packet."""
        return self.emit(AgentEndPacket(
            agent_id=agent_id,
            success=success,
            summary=summary,
            num_sources=sources,
            confidence=confidence,
        ))

    def report_token(self, token: str, section: Optional[str] = None) -> ReportTokenPacket:
        """Emit report token packet."""
        return self.emit(ReportTokenPacket(token=token, section=section))

    def get_all_packets(self) -> list[StreamPacket]:
        """Get all emitted packets."""
        return self.packets.copy()

    def clear(self) -> None:
        """Clear all packets."""
        self.packets.clear()
