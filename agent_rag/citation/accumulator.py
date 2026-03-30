"""Global Citation Accumulator for Deep Research.

Maintains a global citation mapping across multiple research agents,
ensuring consistent citation numbering and proper source deduplication.

Reference: backend/onyx/db/citations.py
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import threading

from agent_rag.core.models import Chunk, Citation
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GlobalCitationEntry:
    """Entry in the global citation mapping."""
    global_id: int  # The display ID shown to users (1, 2, 3, ...)
    chunk: Chunk
    document_id: str
    original_agent_ids: list[str] = field(default_factory=list)  # Which agents cited this
    original_local_ids: list[tuple[str, int]] = field(default_factory=list)  # (agent_id, local_id)
    citation_count: int = 0  # How many times this was cited


@dataclass
class AgentCitationMapping:
    """Mapping from agent's local citation IDs to global IDs."""
    agent_id: str
    local_to_global: dict[int, int] = field(default_factory=dict)  # local_id -> global_id


class GlobalCitationAccumulator:
    """
    Accumulates citations across multiple research agents.

    Key features:
    - Maintains consistent global numbering across all agents
    - Deduplicates citations to the same chunk/document
    - Tracks which agents cited which sources
    - Thread-safe for parallel agent execution

    Usage:
        accumulator = GlobalCitationAccumulator()

        # When agent 1 finds sources
        agent1_mapping = accumulator.register_agent_citations(
            agent_id="agent_1",
            chunks=[chunk1, chunk2, chunk3]
        )

        # When agent 2 finds sources (some may overlap)
        agent2_mapping = accumulator.register_agent_citations(
            agent_id="agent_2",
            chunks=[chunk2, chunk4, chunk5]  # chunk2 is shared
        )

        # Get final citation list for report
        citations = accumulator.get_all_citations()
    """

    def __init__(
        self,
        fold_by_document: bool = True,
        fold_by_chunk: bool = True,
    ) -> None:
        """
        Initialize the accumulator.

        Args:
            fold_by_document: Whether to fold citations to same document
            fold_by_chunk: Whether to fold citations to exact same chunk
        """
        self.fold_by_document = fold_by_document
        self.fold_by_chunk = fold_by_chunk

        # Global state
        self._entries: dict[int, GlobalCitationEntry] = {}  # global_id -> entry
        self._next_global_id: int = 1

        # Lookup indices for deduplication
        self._chunk_to_global: dict[str, int] = {}  # chunk_unique_id -> global_id
        self._doc_to_global: dict[str, int] = {}  # document_id -> first global_id

        # Agent mappings
        self._agent_mappings: dict[str, AgentCitationMapping] = {}

        # Thread safety
        self._lock = threading.RLock()

    def reset(self) -> None:
        """Reset all accumulated state."""
        with self._lock:
            self._entries.clear()
            self._next_global_id = 1
            self._chunk_to_global.clear()
            self._doc_to_global.clear()
            self._agent_mappings.clear()

    def register_agent_citations(
        self,
        agent_id: str,
        chunks: list[Chunk],
    ) -> AgentCitationMapping:
        """
        Register citations from a research agent.

        Maps the agent's local citation numbers (1, 2, 3...) to global IDs,
        handling deduplication.

        Args:
            agent_id: Unique identifier for the agent
            chunks: Chunks cited by the agent in order

        Returns:
            Mapping from agent's local IDs to global IDs
        """
        with self._lock:
            mapping = AgentCitationMapping(agent_id=agent_id)

            for local_id, chunk in enumerate(chunks, 1):
                global_id = self._get_or_create_global_id(chunk, agent_id, local_id)
                mapping.local_to_global[local_id] = global_id

            self._agent_mappings[agent_id] = mapping

            logger.debug(
                f"Registered {len(chunks)} citations from agent {agent_id}, "
                f"mapped to {len(set(mapping.local_to_global.values()))} unique global IDs"
            )

            return mapping

    def _get_or_create_global_id(
        self,
        chunk: Chunk,
        agent_id: str,
        local_id: int,
    ) -> int:
        """Get existing global ID or create new one for a chunk."""
        # Try to find existing citation

        # First check exact chunk match
        if self.fold_by_chunk:
            chunk_key = chunk.unique_id
            if chunk_key in self._chunk_to_global:
                global_id = self._chunk_to_global[chunk_key]
                entry = self._entries[global_id]
                entry.citation_count += 1
                if agent_id not in entry.original_agent_ids:
                    entry.original_agent_ids.append(agent_id)
                entry.original_local_ids.append((agent_id, local_id))
                return global_id

        # Then check document-level folding
        if self.fold_by_document:
            if chunk.document_id in self._doc_to_global:
                global_id = self._doc_to_global[chunk.document_id]
                entry = self._entries[global_id]
                entry.citation_count += 1
                if agent_id not in entry.original_agent_ids:
                    entry.original_agent_ids.append(agent_id)
                entry.original_local_ids.append((agent_id, local_id))

                # Also register this specific chunk for exact matching later
                if self.fold_by_chunk:
                    self._chunk_to_global[chunk.unique_id] = global_id

                return global_id

        # Create new entry
        global_id = self._next_global_id
        self._next_global_id += 1

        entry = GlobalCitationEntry(
            global_id=global_id,
            chunk=chunk,
            document_id=chunk.document_id,
            original_agent_ids=[agent_id],
            original_local_ids=[(agent_id, local_id)],
            citation_count=1,
        )
        self._entries[global_id] = entry

        # Update indices
        self._chunk_to_global[chunk.unique_id] = global_id
        if chunk.document_id not in self._doc_to_global:
            self._doc_to_global[chunk.document_id] = global_id

        return global_id

    def remap_text(
        self,
        text: str,
        agent_id: str,
    ) -> str:
        """
        Remap citations in text from agent's local IDs to global IDs.

        Args:
            text: Text containing citations like [1], [2]
            agent_id: ID of the agent whose local IDs are in the text

        Returns:
            Text with citations remapped to global IDs
        """
        import re

        with self._lock:
            mapping = self._agent_mappings.get(agent_id)
            if not mapping:
                logger.warning(f"No mapping found for agent {agent_id}")
                return text

            pattern = re.compile(r'\[(\d+(?:,\s*\d+)*)\]')

            def replace_citation(match: re.Match) -> str:
                refs = match.group(1).split(',')
                global_ids = []

                for ref in refs:
                    try:
                        local_id = int(ref.strip())
                        global_id = mapping.local_to_global.get(local_id)
                        if global_id is not None:
                            global_ids.append(str(global_id))
                    except ValueError:
                        continue

                if global_ids:
                    return f"[{','.join(global_ids)}]"
                return match.group(0)

            return pattern.sub(replace_citation, text)

    def get_agent_mapping(self, agent_id: str) -> Optional[AgentCitationMapping]:
        """Get the citation mapping for a specific agent."""
        with self._lock:
            return self._agent_mappings.get(agent_id)

    def get_global_citation(self, global_id: int) -> Optional[GlobalCitationEntry]:
        """Get a specific global citation entry."""
        with self._lock:
            return self._entries.get(global_id)

    def get_all_citations(self) -> list[Citation]:
        """
        Get all citations in global order.

        Returns:
            List of Citation objects ordered by global ID
        """
        with self._lock:
            citations = []

            for global_id in sorted(self._entries.keys()):
                entry = self._entries[global_id]
                chunk = entry.chunk

                citation = Citation(
                    citation_num=global_id,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    title=chunk.title,
                    link=chunk.link,
                    source_type=chunk.source_type,
                )
                citations.append(citation)

            return citations

    def get_citation_stats(self) -> dict[str, Any]:
        """
        Get statistics about accumulated citations.

        Returns:
            Dict with citation statistics
        """
        with self._lock:
            return {
                "total_citations": len(self._entries),
                "total_agents": len(self._agent_mappings),
                "citations_by_agent": {
                    agent_id: len(mapping.local_to_global)
                    for agent_id, mapping in self._agent_mappings.items()
                },
                "most_cited": sorted(
                    [
                        {
                            "global_id": entry.global_id,
                            "title": entry.chunk.title,
                            "count": entry.citation_count,
                            "agents": entry.original_agent_ids,
                        }
                        for entry in self._entries.values()
                    ],
                    key=lambda x: x["count"],
                    reverse=True,
                )[:10],
            }

    def merge_from(self, other: "GlobalCitationAccumulator") -> dict[int, int]:
        """
        Merge citations from another accumulator.

        Useful when combining results from sub-orchestrators.

        Args:
            other: Another accumulator to merge from

        Returns:
            Mapping from other's global IDs to this accumulator's global IDs
        """
        with self._lock:
            id_mapping: dict[int, int] = {}

            for other_global_id, entry in other._entries.items():
                # Check if we already have this chunk
                new_global_id = self._get_or_create_global_id(
                    chunk=entry.chunk,
                    agent_id=entry.original_agent_ids[0] if entry.original_agent_ids else "merged",
                    local_id=0,
                )
                id_mapping[other_global_id] = new_global_id

                # Update citation count
                self._entries[new_global_id].citation_count += entry.citation_count - 1

                # Merge agent tracking
                for agent_id in entry.original_agent_ids:
                    if agent_id not in self._entries[new_global_id].original_agent_ids:
                        self._entries[new_global_id].original_agent_ids.append(agent_id)

            return id_mapping


def create_global_accumulator(
    fold_by_document: bool = True,
    fold_by_chunk: bool = True,
) -> GlobalCitationAccumulator:
    """Factory function to create a global citation accumulator."""
    return GlobalCitationAccumulator(
        fold_by_document=fold_by_document,
        fold_by_chunk=fold_by_chunk,
    )
