"""LLM-driven selection and context expansion for search."""

from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import Optional

from agent_rag.core.config import ReasoningEffort
from agent_rag.core.models import Chunk, Section
from agent_rag.llm.interface import LLM, LLMMessage
from agent_rag.tools.builtin.search.prompts import (
    DOCUMENT_CONTEXT_SELECTION_PROMPT,
    DOCUMENT_SELECTION_PROMPT,
)
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


CONNECTOR_NAME_MAP = {
    "web": "Website",
    "requesttracker": "Request Tracker",
    "github": "GitHub",
    "file": "File Upload",
}


def clean_up_source(source_str: Optional[str]) -> str:
    if not source_str:
        return "Unknown"
    if source_str in CONNECTOR_NAME_MAP:
        return CONNECTOR_NAME_MAP[source_str]
    return source_str.replace("_", " ").title()


class ContextExpansionType(str, Enum):
    """How much context to include for a section."""
    NOT_RELEVANT = "NOT_RELEVANT"
    MAIN_SECTION_ONLY = "MAIN_SECTION_ONLY"
    INCLUDE_ADJACENT_SECTIONS = "INCLUDE_ADJACENT_SECTIONS"
    FULL_DOCUMENT = "FULL_DOCUMENT"


@dataclass
class SectionSelection:
    """Section selection result."""
    sections: list[Section]
    full_document_section_ids: list[int] = field(default_factory=list)
    reasoning: Optional[str] = None


def select_sections_for_expansion(
    sections: list[Section],
    query: str,
    llm: LLM,
    max_sections: int = 10,
    max_chunks_per_section: Optional[int] = None,
) -> SectionSelection:
    """Use LLM to select the most relevant sections for expansion."""
    if not sections:
        return SectionSelection(sections=[], reasoning=None)

    previews = []
    for idx, section in enumerate(sections):
        chunks = (
            select_chunks_for_relevance(section, max_chunks_per_section)
            if max_chunks_per_section
            else section.chunks
        )
        preview = " ".join(chunk.content for chunk in chunks)
        chunk = section.center_chunk
        title = chunk.semantic_identifier or chunk.title or "Untitled"

        authors = []
        if chunk.primary_owners:
            authors.extend(chunk.primary_owners)
        if chunk.secondary_owners:
            authors.extend(chunk.secondary_owners)

        # Filter metadata to only essential fields (avoid token explosion)
        filtered_metadata = {}
        if chunk.metadata:
            safe_keys = {"file_type", "page", "section", "language"}
            for key in safe_keys:
                if key in chunk.metadata:
                    value = chunk.metadata[key]
                    if isinstance(value, (int, float, bool)) or (isinstance(value, str) and len(value) < 50):
                        filtered_metadata[key] = value

        previews.append({
            "section_id": idx,
            "title": title[:100] if title else "Untitled",  # Limit title length
            "updated_at": chunk.updated_at.isoformat() if chunk.updated_at else None,
            "authors": authors[:3] if authors else None,  # Limit authors
            "source_type": str(chunk.source_type) if chunk.source_type else None,
            "metadata": filtered_metadata if filtered_metadata else None,
            "content": preview[:500],  # Reduce from 800 to 500
        })

    prompt = DOCUMENT_SELECTION_PROMPT.format(
        max_sections=max_sections,
        formatted_doc_sections=json.dumps(previews, ensure_ascii=False, indent=2),
        user_query=query,
    )
    messages = [LLMMessage(role="user", content=prompt)]

    try:
        # Use ReasoningEffort.OFF for fast document selection (auxiliary task)
        response = llm.chat(messages, max_tokens=150, reasoning_effort=ReasoningEffort.OFF)
        selected_indices = []
        full_doc_ids = []
        for part in response.content.replace(",", " ").split():
            try:
                cleaned = part.strip()
                is_full = cleaned.endswith("!")
                if is_full:
                    cleaned = cleaned[:-1]
                idx = int(cleaned)
                if 0 <= idx < len(sections):
                    selected_indices.append(idx)
                    if is_full:
                        full_doc_ids.append(idx)
            except ValueError:
                continue

        if selected_indices:
            chosen = [sections[i] for i in selected_indices[:max_sections]]
            return SectionSelection(
                sections=chosen,
                full_document_section_ids=full_doc_ids,
                reasoning=response.content,
            )
    except Exception as exc:
        logger.warning(f"Section selection failed: {exc}")

    return SectionSelection(sections=sections[:max_sections], full_document_section_ids=[], reasoning=None)


def select_chunks_for_relevance(
    section: Section,
    max_chunks: int = 3,
) -> list[Chunk]:
    """Select a subset of chunks based on center chunk position."""
    if max_chunks <= 0:
        return []

    center_chunk = section.center_chunk
    all_chunks = section.chunks

    # Find the index of the center chunk in the chunks list
    try:
        center_index = next(
            i for i, chunk in enumerate(all_chunks) if chunk.chunk_id == center_chunk.chunk_id
        )
    except StopIteration:
        return [center_chunk]

    if max_chunks == 1:
        return [center_chunk]

    chunks_needed = max_chunks - 1
    chunks_before_available = center_index
    chunks_after_available = len(all_chunks) - center_index - 1

    chunks_before = min(chunks_needed // 2, chunks_before_available)
    chunks_after = min(chunks_needed // 2, chunks_after_available)

    remaining = chunks_needed - chunks_before - chunks_after
    if remaining > 0:
        if chunks_before_available > chunks_before:
            additional_before = min(remaining, chunks_before_available - chunks_before)
            chunks_before += additional_before
            remaining -= additional_before
        if remaining > 0 and chunks_after_available > chunks_after:
            additional_after = min(remaining, chunks_after_available - chunks_after)
            chunks_after += additional_after

    start_index = center_index - chunks_before
    end_index = center_index + chunks_after + 1
    return all_chunks[start_index:end_index]


def classify_section_relevance(
    section: Section,
    query: str,
    llm: LLM,
    section_above: Optional[str],
    section_below: Optional[str],
) -> ContextExpansionType:
    """Classify how much context to include for a section."""
    prompt = DOCUMENT_CONTEXT_SELECTION_PROMPT.format(
        document_title=section.center_chunk.semantic_identifier
        or section.center_chunk.title
        or "Untitled",
        main_section=section.combined_content[:1200],
        section_above=section_above or "N/A",
        section_below=section_below or "N/A",
        user_query=query,
    )
    messages = [LLMMessage(role="user", content=prompt)]

    try:
        # Use ReasoningEffort.OFF for fast context classification (auxiliary task)
        response = llm.chat(messages, max_tokens=50, reasoning_effort=ReasoningEffort.OFF)
        raw = (response.content or "").strip()
        numbers = re.findall(r"\b[1-4]\b", raw)
        if numbers:
            situation = int(numbers[-1])
            mapping = {
                1: ContextExpansionType.NOT_RELEVANT,
                2: ContextExpansionType.MAIN_SECTION_ONLY,
                3: ContextExpansionType.INCLUDE_ADJACENT_SECTIONS,
                4: ContextExpansionType.FULL_DOCUMENT,
            }
            return mapping.get(situation, ContextExpansionType.MAIN_SECTION_ONLY)
        upper = raw.upper()
        for choice in ContextExpansionType:
            if choice.value in upper:
                return choice
    except Exception as exc:
        logger.warning(f"Context selection failed: {exc}")

    return ContextExpansionType.MAIN_SECTION_ONLY
