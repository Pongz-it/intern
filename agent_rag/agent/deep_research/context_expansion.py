"""Dynamic Context Expansion for Deep Research.

Implements intelligent context window expansion based on:
- Chunk boundaries and coherence
- Semantic continuity
- Token budget constraints
- Relevance-based prioritization

Reference: backend/onyx/agents/agent_search/shared/expanded_retrieval.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from agent_rag.core.models import Chunk
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class ExpansionDirection(str, Enum):
    """Direction of context expansion."""
    BEFORE = "before"
    AFTER = "after"
    BOTH = "both"


class ExpansionReason(str, Enum):
    """Reason for context expansion."""
    SENTENCE_BOUNDARY = "sentence_boundary"
    PARAGRAPH_BOUNDARY = "paragraph_boundary"
    SEMANTIC_CONTINUITY = "semantic_continuity"
    REFERENCE_RESOLUTION = "reference_resolution"
    INCOMPLETE_THOUGHT = "incomplete_thought"


@dataclass
class ExpandedChunk:
    """A chunk with expanded context."""
    original_chunk: Chunk
    expanded_content: str
    context_before: str
    context_after: str
    expansion_tokens: int
    expansion_reason: ExpansionReason
    relevance_score: float


@dataclass
class ContextWindow:
    """A window of context around a chunk."""
    chunks: list[Chunk]
    center_chunk_index: int
    total_tokens: int
    relevance_weighted_score: float


@dataclass
class ContextExpansionConfig:
    """Configuration for context expansion."""
    # Token budgets
    max_context_tokens: int = 1000  # Max tokens for context expansion
    min_context_tokens: int = 50  # Min tokens to add
    target_context_tokens: int = 200  # Target tokens per side

    # Expansion rules
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    expand_incomplete_sentences: bool = True

    # Direction preferences
    default_direction: ExpansionDirection = ExpansionDirection.BOTH
    before_weight: float = 0.4  # Weight for before context
    after_weight: float = 0.6  # Weight for after context (usually more important)

    # Relevance thresholds
    min_relevance_for_expansion: float = 0.3
    relevance_decay_per_sentence: float = 0.1


class DynamicContextExpander:
    """
    Dynamically expands context around chunks for better comprehension.

    Features:
    - Sentence/paragraph boundary detection
    - Token budget management
    - Relevance-weighted expansion
    - Semantic continuity analysis
    """

    def __init__(
        self,
        config: Optional[ContextExpansionConfig] = None,
    ) -> None:
        """
        Initialize the context expander.

        Args:
            config: Configuration options
        """
        self.config = config or ContextExpansionConfig()

    def expand_chunk(
        self,
        chunk: Chunk,
        full_document_content: str,
        chunk_start_offset: int,
        question: Optional[str] = None,
        token_budget: Optional[int] = None,
    ) -> ExpandedChunk:
        """
        Expand a chunk with surrounding context.

        Args:
            chunk: The chunk to expand
            full_document_content: Full content of the source document
            chunk_start_offset: Character offset where chunk starts in document
            question: Research question for relevance-based expansion
            token_budget: Optional token budget override

        Returns:
            ExpandedChunk with added context
        """
        budget = token_budget or self.config.max_context_tokens

        # Calculate expansion for before and after
        before_budget = int(budget * self.config.before_weight)
        after_budget = int(budget * self.config.after_weight)

        # Get context before
        context_before, before_reason = self._expand_before(
            content=full_document_content,
            offset=chunk_start_offset,
            token_budget=before_budget,
        )

        # Get context after
        chunk_end_offset = chunk_start_offset + len(chunk.content)
        context_after, after_reason = self._expand_after(
            content=full_document_content,
            offset=chunk_end_offset,
            token_budget=after_budget,
        )

        # Combine expanded content
        expanded_content = context_before + chunk.content + context_after

        # Estimate tokens
        expansion_tokens = self._estimate_tokens(context_before + context_after)

        # Calculate relevance score
        relevance_score = self._calculate_expansion_relevance(
            original=chunk.content,
            context_before=context_before,
            context_after=context_after,
            question=question,
        )

        # Determine primary expansion reason
        expansion_reason = before_reason if before_reason else after_reason
        if not expansion_reason:
            expansion_reason = ExpansionReason.SEMANTIC_CONTINUITY

        return ExpandedChunk(
            original_chunk=chunk,
            expanded_content=expanded_content,
            context_before=context_before,
            context_after=context_after,
            expansion_tokens=expansion_tokens,
            expansion_reason=expansion_reason,
            relevance_score=relevance_score,
        )

    def expand_chunks(
        self,
        chunks: list[Chunk],
        full_document_content: str,
        chunk_offsets: list[int],
        question: Optional[str] = None,
        total_token_budget: Optional[int] = None,
    ) -> list[ExpandedChunk]:
        """
        Expand multiple chunks with budget allocation.

        Args:
            chunks: Chunks to expand
            full_document_content: Full document content
            chunk_offsets: Start offsets for each chunk
            question: Research question
            total_token_budget: Total token budget for all expansions

        Returns:
            List of ExpandedChunk objects
        """
        if not chunks:
            return []

        # Calculate per-chunk budget
        budget = total_token_budget or (self.config.max_context_tokens * len(chunks))
        per_chunk_budget = budget // len(chunks)

        expanded = []
        for chunk, offset in zip(chunks, chunk_offsets):
            expanded_chunk = self.expand_chunk(
                chunk=chunk,
                full_document_content=full_document_content,
                chunk_start_offset=offset,
                question=question,
                token_budget=per_chunk_budget,
            )
            expanded.append(expanded_chunk)

        return expanded

    def create_context_window(
        self,
        chunks: list[Chunk],
        center_index: int,
        window_size: int = 3,
        token_budget: Optional[int] = None,
    ) -> ContextWindow:
        """
        Create a context window around a central chunk.

        Args:
            chunks: All available chunks
            center_index: Index of the central chunk
            window_size: Number of chunks on each side
            token_budget: Optional token budget

        Returns:
            ContextWindow with surrounding chunks
        """
        budget = token_budget or self.config.max_context_tokens

        # Calculate window boundaries
        start_idx = max(0, center_index - window_size)
        end_idx = min(len(chunks), center_index + window_size + 1)

        # Get window chunks
        window_chunks = chunks[start_idx:end_idx]

        # Calculate total tokens (estimate)
        total_tokens = sum(
            self._estimate_tokens(c.content)
            for c in window_chunks
        )

        # Trim if over budget
        if total_tokens > budget:
            window_chunks = self._trim_to_budget(
                chunks=window_chunks,
                center_index=center_index - start_idx,
                budget=budget,
            )
            total_tokens = sum(
                self._estimate_tokens(c.content)
                for c in window_chunks
            )

        # Calculate relevance-weighted score
        relevance_score = self._calculate_window_relevance(
            window_chunks,
            center_index - start_idx,
        )

        return ContextWindow(
            chunks=window_chunks,
            center_chunk_index=center_index - start_idx,
            total_tokens=total_tokens,
            relevance_weighted_score=relevance_score,
        )

    def _expand_before(
        self,
        content: str,
        offset: int,
        token_budget: int,
    ) -> tuple[str, Optional[ExpansionReason]]:
        """Expand context before the chunk."""
        if offset <= 0 or token_budget <= 0:
            return "", None

        # Get available content before
        available = content[:offset]
        if not available:
            return "", None

        # Find good boundary points
        expansion_text = ""
        reason = None

        # First, try paragraph boundary
        if self.config.respect_paragraph_boundaries:
            para_boundary = available.rfind('\n\n')
            if para_boundary >= 0 and len(available) - para_boundary < token_budget * 4:
                expansion_text = available[para_boundary:].lstrip('\n')
                reason = ExpansionReason.PARAGRAPH_BOUNDARY

        # If no paragraph boundary or too far, try sentence boundary
        if not expansion_text and self.config.respect_sentence_boundaries:
            sentence_boundary = self._find_sentence_boundary_before(available)
            if sentence_boundary >= 0:
                expansion_text = available[sentence_boundary:].lstrip()
                reason = ExpansionReason.SENTENCE_BOUNDARY

        # If still no good boundary, just take what fits in budget
        if not expansion_text:
            # Estimate characters from token budget
            char_budget = token_budget * 4  # Rough estimate
            expansion_text = available[-char_budget:]
            reason = ExpansionReason.SEMANTIC_CONTINUITY

        # Trim to budget
        expansion_text = self._trim_to_token_budget(expansion_text, token_budget)

        return expansion_text, reason

    def _expand_after(
        self,
        content: str,
        offset: int,
        token_budget: int,
    ) -> tuple[str, Optional[ExpansionReason]]:
        """Expand context after the chunk."""
        if offset >= len(content) or token_budget <= 0:
            return "", None

        # Get available content after
        available = content[offset:]
        if not available:
            return "", None

        expansion_text = ""
        reason = None

        # First, try paragraph boundary
        if self.config.respect_paragraph_boundaries:
            para_boundary = available.find('\n\n')
            if para_boundary >= 0 and para_boundary < token_budget * 4:
                expansion_text = available[:para_boundary].rstrip('\n')
                reason = ExpansionReason.PARAGRAPH_BOUNDARY

        # If no paragraph boundary or too far, try sentence boundary
        if not expansion_text and self.config.respect_sentence_boundaries:
            sentence_boundary = self._find_sentence_boundary_after(available)
            if sentence_boundary >= 0:
                expansion_text = available[:sentence_boundary + 1].rstrip()
                reason = ExpansionReason.SENTENCE_BOUNDARY

        # If still no good boundary, just take what fits in budget
        if not expansion_text:
            char_budget = token_budget * 4
            expansion_text = available[:char_budget]
            reason = ExpansionReason.SEMANTIC_CONTINUITY

        # Check for incomplete sentence at end and extend
        if self.config.expand_incomplete_sentences:
            if not expansion_text.rstrip().endswith(('.', '!', '?', '"', "'")):
                # Try to complete the sentence
                remaining = available[len(expansion_text):]
                sentence_end = self._find_sentence_boundary_after(remaining[:200])
                if sentence_end >= 0:
                    expansion_text += remaining[:sentence_end + 1]
                    reason = ExpansionReason.INCOMPLETE_THOUGHT

        # Trim to budget
        expansion_text = self._trim_to_token_budget(expansion_text, token_budget)

        return expansion_text, reason

    def _find_sentence_boundary_before(self, text: str) -> int:
        """Find the last sentence boundary in text."""
        # Look for sentence-ending punctuation followed by space or end
        import re
        matches = list(re.finditer(r'[.!?]["\']?\s', text))
        if matches:
            return matches[-1].end()
        return -1

    def _find_sentence_boundary_after(self, text: str) -> int:
        """Find the first sentence boundary in text."""
        import re
        match = re.search(r'[.!?]["\']?\s', text)
        if match:
            return match.end() - 1  # Position of the punctuation
        return -1

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple heuristic: ~4 characters per token for English
        return len(text) // 4

    def _trim_to_token_budget(self, text: str, token_budget: int) -> str:
        """Trim text to fit within token budget."""
        estimated_tokens = self._estimate_tokens(text)
        if estimated_tokens <= token_budget:
            return text

        # Calculate target character count
        target_chars = token_budget * 4
        if len(text) <= target_chars:
            return text

        return text[:target_chars]

    def _trim_to_budget(
        self,
        chunks: list[Chunk],
        center_index: int,
        budget: int,
    ) -> list[Chunk]:
        """Trim chunks to fit within budget, keeping center."""
        if not chunks:
            return []

        # Always keep the center chunk
        result = [chunks[center_index]]
        remaining_budget = budget - self._estimate_tokens(chunks[center_index].content)

        # Alternately add before and after chunks
        before_idx = center_index - 1
        after_idx = center_index + 1

        while remaining_budget > 0 and (before_idx >= 0 or after_idx < len(chunks)):
            # Try to add after chunk (weighted preference)
            if after_idx < len(chunks):
                after_tokens = self._estimate_tokens(chunks[after_idx].content)
                if after_tokens <= remaining_budget:
                    result.append(chunks[after_idx])
                    remaining_budget -= after_tokens
                after_idx += 1

            # Try to add before chunk
            if before_idx >= 0:
                before_tokens = self._estimate_tokens(chunks[before_idx].content)
                if before_tokens <= remaining_budget:
                    result.insert(0, chunks[before_idx])
                    remaining_budget -= before_tokens
                before_idx -= 1

            # Break if we can't add anything
            if remaining_budget < self.config.min_context_tokens:
                break

        return result

    def _calculate_expansion_relevance(
        self,
        original: str,
        context_before: str,
        context_after: str,
        question: Optional[str],
    ) -> float:
        """Calculate relevance of the expanded content."""
        if not question:
            return 0.5  # Default moderate relevance

        # Simple keyword overlap scoring
        question_words = set(question.lower().split())

        original_words = set(original.lower().split())
        before_words = set(context_before.lower().split()) if context_before else set()
        after_words = set(context_after.lower().split()) if context_after else set()

        # Calculate overlaps
        original_overlap = len(original_words & question_words)
        before_overlap = len(before_words & question_words)
        after_overlap = len(after_words & question_words)

        total_overlap = original_overlap + before_overlap + after_overlap
        if total_overlap == 0:
            return 0.0

        # Weight the original content more heavily
        relevance = (
            original_overlap * 0.6 +
            (before_overlap + after_overlap) * 0.4
        ) / len(question_words)

        return min(1.0, relevance)

    def _calculate_window_relevance(
        self,
        chunks: list[Chunk],
        center_index: int,
    ) -> float:
        """Calculate relevance-weighted score for a window."""
        if not chunks:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for i, chunk in enumerate(chunks):
            # Distance from center
            distance = abs(i - center_index)

            # Weight decays with distance
            weight = 1.0 / (1 + distance * self.config.relevance_decay_per_sentence)

            # Use chunk's relevance score if available
            chunk_score = getattr(chunk, 'relevance_score', 0.5)

            total_score += chunk_score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0


def create_context_expander(
    config: Optional[ContextExpansionConfig] = None,
) -> DynamicContextExpander:
    """Factory function to create a context expander."""
    return DynamicContextExpander(config=config)
