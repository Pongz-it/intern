"""Utility helpers for search result expansion."""

from collections import defaultdict
from collections.abc import Callable
from typing import Any, Optional, TypeVar

from agent_rag.core.models import Chunk, Section
from agent_rag.document_index.interface import ChunkRequest, DocumentIndex
from agent_rag.tools.builtin.search.document_filter import (
    ContextExpansionType,
    classify_section_relevance,
)

T = TypeVar("T")


def build_section_from_chunks(chunks: list[Chunk], center_chunk: Chunk) -> Section:
    """Create a section from chunks."""
    combined_content = "\n\n".join(chunk.content for chunk in chunks)
    return Section(center_chunk=center_chunk, chunks=chunks, combined_content=combined_content)


def weighted_reciprocal_rank_fusion(
    ranked_results: list[list[T]],
    weights: list[float],
    id_extractor: Callable[[T], str],
    k: int,
) -> list[T]:
    """Merge multiple ranked lists using weighted RRF with tie-breaking."""
    if len(ranked_results) != len(weights):
        raise ValueError("ranked_results and weights must have same length")

    rrf_scores: dict[str, float] = defaultdict(float)
    id_to_item: dict[str, T] = {}
    id_to_source_index: dict[str, int] = {}
    id_to_source_rank: dict[str, int] = {}

    for source_idx, (result_list, weight) in enumerate(zip(ranked_results, weights)):
        for rank, item in enumerate(result_list, start=1):
            item_id = id_extractor(item)
            rrf_scores[item_id] += weight / (k + rank)
            if item_id not in id_to_item:
                id_to_item[item_id] = item
                id_to_source_index[item_id] = source_idx
                id_to_source_rank[item_id] = rank

    sorted_ids = sorted(
        rrf_scores.keys(),
        key=lambda item_id: (
            -rrf_scores[item_id],
            id_to_source_rank[item_id],
            id_to_source_index[item_id],
        ),
    )
    return [id_to_item[item_id] for item_id in sorted_ids]


def merge_overlapping_sections(sections: list[Section]) -> list[Section]:
    """Merge sections from the same document that overlap or are adjacent."""
    if not sections:
        return []

    section_to_original_index: dict[tuple[str, int], int] = {}
    for idx, section in enumerate(sections):
        section_id = (section.center_chunk.document_id, section.center_chunk.chunk_id)
        section_to_original_index[section_id] = idx

    doc_sections: dict[str, list[Section]] = defaultdict(list)
    for section in sections:
        doc_sections[section.center_chunk.document_id].append(section)

    merged_sections: dict[tuple[str, int], Section] = {}

    for doc_id, doc_section_list in doc_sections.items():
        if not doc_section_list:
            continue

        doc_section_list.sort(key=lambda s: min(c.chunk_id for c in s.chunks))
        # Use dict keyed by chunk_id to avoid hashability issues with Chunk objects
        current_merged_chunks: dict[int, Chunk] = {c.chunk_id: c for c in doc_section_list[0].chunks}
        sections_in_current_group = [doc_section_list[0]]

        for current_section in doc_section_list[1:]:
            current_section_chunks = {c.chunk_id: c for c in current_section.chunks}
            merged_chunk_ids = set(current_merged_chunks.keys())
            current_chunk_ids = set(current_section_chunks.keys())

            min_merged = min(merged_chunk_ids)
            max_merged = max(merged_chunk_ids)
            min_current = min(current_chunk_ids)
            max_current = max(current_chunk_ids)

            is_adjacent = (min_current == max_merged + 1) or (min_merged == max_current + 1)
            is_overlapping = bool(merged_chunk_ids & current_chunk_ids)

            if is_adjacent or is_overlapping:
                current_merged_chunks.update(current_section_chunks)
                sections_in_current_group.append(current_section)
            else:
                first_section = min(
                    sections_in_current_group,
                    key=lambda s: section_to_original_index.get(
                        (s.center_chunk.document_id, s.center_chunk.chunk_id),
                        float("inf"),
                    ),
                )
                all_chunks = sorted(current_merged_chunks.values(), key=lambda c: c.chunk_id)
                merged_section = build_section_from_chunks(all_chunks, first_section.center_chunk)
                for section in sections_in_current_group:
                    section_id = (section.center_chunk.document_id, section.center_chunk.chunk_id)
                    merged_sections[section_id] = merged_section

                current_merged_chunks = current_section_chunks
                sections_in_current_group = [current_section]

        if sections_in_current_group:
            first_section = min(
                sections_in_current_group,
                key=lambda s: section_to_original_index.get(
                    (s.center_chunk.document_id, s.center_chunk.chunk_id),
                    float("inf"),
                ),
            )
            all_chunks = sorted(current_merged_chunks.values(), key=lambda c: c.chunk_id)
            merged_section = build_section_from_chunks(all_chunks, first_section.center_chunk)
            for section in sections_in_current_group:
                section_id = (section.center_chunk.document_id, section.center_chunk.chunk_id)
                merged_sections[section_id] = merged_section

    seen_section_ids: set[tuple[str, int]] = set()
    result: list[Section] = []
    for section in sections:
        section_id = (section.center_chunk.document_id, section.center_chunk.chunk_id)
        merged_section = merged_sections.get(section_id, section)
        merged_section_id = (
            merged_section.center_chunk.document_id,
            merged_section.center_chunk.chunk_id,
        )
        if merged_section_id not in seen_section_ids:
            seen_section_ids.add(merged_section_id)
            result.append(merged_section)

    return result


def expand_section_with_context(
    section: Section,
    user_query: str,
    llm: Any,
    document_index: DocumentIndex,
    full_doc_num_chunks_around: int,
    expand_override: bool = False,
) -> Optional[Section]:
    """Classify relevance and expand a section with appropriate context."""
    if expand_override:
        classification = ContextExpansionType.FULL_DOCUMENT
        chunks_above_for_prompt: list[Chunk] = []
        chunks_below_for_prompt: list[Chunk] = []
    else:
        chunks_above_for_prompt, chunks_below_for_prompt = _retrieve_adjacent_chunks(
            section=section,
            document_index=document_index,
            num_chunks_above=2,
            num_chunks_below=2,
        )

        section_above_text = (
            " ".join([c.content for c in chunks_above_for_prompt])
            if chunks_above_for_prompt
            else None
        )
        section_below_text = (
            " ".join([c.content for c in chunks_below_for_prompt])
            if chunks_below_for_prompt
            else None
        )

        classification = classify_section_relevance(
            section=section,
            query=user_query,
            llm=llm,
            section_above=section_above_text,
            section_below=section_below_text,
        )

    if classification == ContextExpansionType.NOT_RELEVANT:
        return None

    if classification == ContextExpansionType.MAIN_SECTION_ONLY:
        return section

    if classification == ContextExpansionType.INCLUDE_ADJACENT_SECTIONS:
        all_chunks = chunks_above_for_prompt + section.chunks + chunks_below_for_prompt
        if not all_chunks:
            return section
        return build_section_from_chunks(all_chunks, section.center_chunk)

    if classification == ContextExpansionType.FULL_DOCUMENT:
        chunks_above_full, chunks_below_full = _retrieve_adjacent_chunks(
            section=section,
            document_index=document_index,
            num_chunks_above=full_doc_num_chunks_around,
            num_chunks_below=full_doc_num_chunks_around,
        )
        all_chunks = chunks_above_full + section.chunks + chunks_below_full
        if not all_chunks:
            return section
        return build_section_from_chunks(all_chunks, section.center_chunk)

    return section


def _retrieve_adjacent_chunks(
    section: Section,
    document_index: DocumentIndex,
    num_chunks_above: int,
    num_chunks_below: int,
) -> tuple[list[Chunk], list[Chunk]]:
    """Retrieve adjacent chunks above and below a section."""
    document_id = section.center_chunk.document_id
    chunk_ids = [chunk.chunk_id for chunk in section.chunks]
    min_chunk_id = min(chunk_ids)
    max_chunk_id = max(chunk_ids)

    chunks_above: list[Chunk] = []
    chunks_below: list[Chunk] = []

    requests: list[ChunkRequest] = []
    if num_chunks_above > 0 and min_chunk_id > 0:
        above_min = max(0, min_chunk_id - num_chunks_above)
        above_max = min_chunk_id - 1
        requests.append(ChunkRequest(
            document_id=document_id,
            min_chunk_id=above_min,
            max_chunk_id=above_max,
        ))

    if num_chunks_below > 0:
        below_min = max_chunk_id + 1
        below_max = max_chunk_id + num_chunks_below
        requests.append(ChunkRequest(
            document_id=document_id,
            min_chunk_id=below_min,
            max_chunk_id=below_max,
        ))

    if requests:
        retrieved = document_index.id_based_retrieval(requests, batch_retrieval=True)
        for chunk in retrieved:
            if chunk.chunk_id < min_chunk_id:
                chunks_above.append(chunk)
            elif chunk.chunk_id > max_chunk_id:
                chunks_below.append(chunk)

    chunks_above.sort(key=lambda c: c.chunk_id)
    chunks_below.sort(key=lambda c: c.chunk_id)

    return chunks_above, chunks_below




def trim_sections_by_tokens(
    sections: list[Section],
    token_counter: Any,
    max_tokens: int,
    max_chunks_per_section: Optional[int] = None,
) -> list[Section]:
    """Trim sections to fit within a token budget."""
    if not sections or max_tokens <= 0:
        return sections

    trimmed: list[Section] = []
    total_tokens = 0

    for section in sections:
        if max_chunks_per_section is not None:
            selected = section.chunks[:max_chunks_per_section]
            content = "\n".join(chunk.content for chunk in selected)
        else:
            content = section.combined_content

        section_tokens = token_counter(content)
        if total_tokens + section_tokens <= max_tokens:
            trimmed.append(section)
            total_tokens += section_tokens
        else:
            break

    return trimmed


def score_chunks_by_query(
    query: str,
    chunks: list[Chunk],
) -> list[Chunk]:
    """Score and sort chunks by relevance to the query.

    Args:
        query: The search query
        chunks: List of chunks to score

    Returns:
        Chunks sorted by relevance score (descending)
    """
    if not chunks:
        return []

    def get_score(c: Chunk) -> float:
        try:
            if c.score is None:
                return 0.0
            return float(c.score)
        except (TypeError, ValueError):
            return 0.0

    sorted_chunks = sorted(chunks, key=get_score, reverse=True)
    return sorted_chunks


def dedupe_chunks_by_title(
    chunks: list[Chunk],
) -> list[Chunk]:
    """Remove duplicate chunks by title, keeping the highest scoring one.

    Args:
        chunks: List of chunks to deduplicate

    Returns:
        Deduplicated list of chunks
    """
    if not chunks:
        return []

    def get_score(c: Chunk) -> float:
        try:
            if c.score is None:
                return 0.0
            return float(c.score)
        except (TypeError, ValueError):
            return 0.0

    seen_titles: dict[str, Chunk] = {}

    for chunk in chunks:
        title = chunk.title or chunk.document_id or chunk.unique_id
        current_score = get_score(chunk)
        existing_score = get_score(seen_titles[title]) if title in seen_titles else 0.0
        
        if title not in seen_titles or current_score > existing_score:
            seen_titles[title] = chunk

    return list(seen_titles.values())
