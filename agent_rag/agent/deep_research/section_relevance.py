"""Section Relevance Scoring for Deep Research.

Implements intelligent section-level relevance scoring with 4 expansion strategies:
1. Most Relevant Section Only - Focus on highest scoring section
2. Relevant Sections Only - Include all sections above threshold
3. All Sections with Relevance Weights - Include all but weight by relevance
4. Dynamic Context Expansion - Expand adjacent sections for context

Reference: backend/onyx/agents/agent_search/deep/shared/expanded_retrieval.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from agent_rag.core.models import Chunk
from agent_rag.llm.interface import LLM, LLMMessage
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class ExpansionStrategy(str, Enum):
    """Section expansion strategies."""
    MOST_RELEVANT_ONLY = "most_relevant_only"
    RELEVANT_SECTIONS = "relevant_sections"
    ALL_WITH_WEIGHTS = "all_with_weights"
    DYNAMIC_CONTEXT = "dynamic_context"


@dataclass
class SectionScore:
    """Relevance score for a document section."""
    section_id: str
    content: str
    relevance_score: float  # 0.0 to 1.0
    semantic_similarity: float
    keyword_overlap: float
    position_weight: float  # Earlier sections may be more important
    is_adjacent_context: bool = False  # If added for context expansion
    expansion_reason: Optional[str] = None


@dataclass
class ScoredDocument:
    """Document with scored sections."""
    document_id: str
    title: str
    sections: list[SectionScore]
    overall_relevance: float
    selected_sections: list[SectionScore] = field(default_factory=list)
    expansion_strategy: ExpansionStrategy = ExpansionStrategy.RELEVANT_SECTIONS


@dataclass
class SectionScoringConfig:
    """Configuration for section relevance scoring."""
    expansion_strategy: ExpansionStrategy = ExpansionStrategy.RELEVANT_SECTIONS
    relevance_threshold: float = 0.5  # Minimum score to include
    max_sections_per_doc: int = 5
    context_window: int = 1  # Adjacent sections to include for context
    use_llm_scoring: bool = True  # Use LLM for semantic scoring
    position_decay: float = 0.9  # Decay factor for later sections
    combine_adjacent: bool = True  # Combine adjacent relevant sections


# =============================================================================
# PROMPTS
# =============================================================================

SECTION_RELEVANCE_PROMPT = """You are a relevance assessment expert. Score the relevance of document sections to a research question.

Research Question: {question}

Document Title: {doc_title}

Sections to Score:
{sections_text}

For each section, provide a relevance score from 0.0 to 1.0:
- 1.0: Directly and comprehensively answers the question
- 0.7-0.9: Highly relevant, contains key information
- 0.4-0.6: Moderately relevant, contains related information
- 0.1-0.3: Marginally relevant, tangentially related
- 0.0: Not relevant

Format your response as:
Section 1: [score] | [brief justification]
Section 2: [score] | [brief justification]
...
"""

SECTION_RELATIONSHIP_PROMPT = """Analyze the relationships between document sections to determine which provide important context.

Research Question: {question}
Document Title: {doc_title}

Sections:
{sections_text}

Identify:
1. Which sections provide crucial context for understanding other sections
2. Which sections should be read together
3. Any logical dependencies between sections

Format as:
## Context Relationships
- Section X provides context for Section Y because...
- Sections A and B should be read together because...
...
"""


class SectionRelevanceScorer:
    """
    Scores document sections for relevance to research questions.

    Implements 4 expansion strategies:
    1. MOST_RELEVANT_ONLY: Only use the highest-scoring section
    2. RELEVANT_SECTIONS: Include all sections above threshold
    3. ALL_WITH_WEIGHTS: Include all sections but weight by relevance
    4. DYNAMIC_CONTEXT: Expand to include adjacent context sections
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        config: Optional[SectionScoringConfig] = None,
    ) -> None:
        """
        Initialize the scorer.

        Args:
            llm: LLM for semantic scoring (optional)
            config: Scoring configuration
        """
        self.llm = llm
        self.config = config or SectionScoringConfig()

    def score_document(
        self,
        document_id: str,
        title: str,
        sections: list[str],
        question: str,
    ) -> ScoredDocument:
        """
        Score all sections of a document for relevance.

        Args:
            document_id: Unique document identifier
            title: Document title
            sections: List of section contents
            question: Research question

        Returns:
            ScoredDocument with scored and selected sections
        """
        # Score each section
        scored_sections = []
        for i, content in enumerate(sections):
            section_id = f"{document_id}_section_{i}"

            # Calculate base scores
            keyword_score = self._calculate_keyword_overlap(content, question)
            position_weight = self._calculate_position_weight(i, len(sections))

            # Get semantic score (LLM or heuristic)
            if self.config.use_llm_scoring and self.llm:
                semantic_score = self._get_llm_semantic_score(
                    content, question, title
                )
            else:
                semantic_score = self._calculate_heuristic_semantic_score(
                    content, question
                )

            # Combine scores
            relevance_score = self._combine_scores(
                semantic=semantic_score,
                keyword=keyword_score,
                position=position_weight,
            )

            scored_sections.append(SectionScore(
                section_id=section_id,
                content=content,
                relevance_score=relevance_score,
                semantic_similarity=semantic_score,
                keyword_overlap=keyword_score,
                position_weight=position_weight,
            ))

        # Apply expansion strategy
        selected_sections = self._apply_expansion_strategy(scored_sections)

        # Calculate overall document relevance
        overall_relevance = self._calculate_overall_relevance(selected_sections)

        return ScoredDocument(
            document_id=document_id,
            title=title,
            sections=scored_sections,
            overall_relevance=overall_relevance,
            selected_sections=selected_sections,
            expansion_strategy=self.config.expansion_strategy,
        )

    def score_chunks(
        self,
        chunks: list[Chunk],
        question: str,
    ) -> list[tuple[Chunk, float]]:
        """
        Score a list of chunks for relevance.

        Args:
            chunks: List of chunks to score
            question: Research question

        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        scored_chunks = []

        for chunk in chunks:
            # Calculate component scores
            keyword_score = self._calculate_keyword_overlap(chunk.content, question)

            if self.config.use_llm_scoring and self.llm:
                semantic_score = self._get_llm_semantic_score(
                    chunk.content, question, chunk.title or ""
                )
            else:
                semantic_score = self._calculate_heuristic_semantic_score(
                    chunk.content, question
                )

            # No position weight for standalone chunks
            relevance_score = 0.6 * semantic_score + 0.4 * keyword_score
            scored_chunks.append((chunk, relevance_score))

        # Sort by relevance
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks

    def _apply_expansion_strategy(
        self,
        sections: list[SectionScore],
    ) -> list[SectionScore]:
        """Apply the configured expansion strategy."""
        strategy = self.config.expansion_strategy

        if strategy == ExpansionStrategy.MOST_RELEVANT_ONLY:
            return self._strategy_most_relevant_only(sections)

        elif strategy == ExpansionStrategy.RELEVANT_SECTIONS:
            return self._strategy_relevant_sections(sections)

        elif strategy == ExpansionStrategy.ALL_WITH_WEIGHTS:
            return self._strategy_all_with_weights(sections)

        elif strategy == ExpansionStrategy.DYNAMIC_CONTEXT:
            return self._strategy_dynamic_context(sections)

        else:
            return self._strategy_relevant_sections(sections)

    def _strategy_most_relevant_only(
        self,
        sections: list[SectionScore],
    ) -> list[SectionScore]:
        """
        Strategy 1: Most Relevant Section Only.

        Only include the single highest-scoring section.
        Best for: Focused queries with clear answers.
        """
        if not sections:
            return []

        # Find highest scoring section
        best_section = max(sections, key=lambda s: s.relevance_score)

        # Only include if above threshold
        if best_section.relevance_score >= self.config.relevance_threshold:
            return [best_section]
        return []

    def _strategy_relevant_sections(
        self,
        sections: list[SectionScore],
    ) -> list[SectionScore]:
        """
        Strategy 2: Relevant Sections Only.

        Include all sections above the relevance threshold.
        Best for: Comprehensive queries needing multiple aspects.
        """
        relevant = [
            s for s in sections
            if s.relevance_score >= self.config.relevance_threshold
        ]

        # Sort by relevance and limit
        relevant.sort(key=lambda s: s.relevance_score, reverse=True)
        return relevant[:self.config.max_sections_per_doc]

    def _strategy_all_with_weights(
        self,
        sections: list[SectionScore],
    ) -> list[SectionScore]:
        """
        Strategy 3: All Sections with Relevance Weights.

        Include all sections but weight their importance by relevance.
        Best for: Exploratory research where context matters.
        """
        # Include all sections, sorted by relevance
        all_sections = sorted(sections, key=lambda s: s.relevance_score, reverse=True)

        # Limit to max sections
        return all_sections[:self.config.max_sections_per_doc]

    def _strategy_dynamic_context(
        self,
        sections: list[SectionScore],
    ) -> list[SectionScore]:
        """
        Strategy 4: Dynamic Context Expansion.

        Include relevant sections plus adjacent sections for context.
        Best for: Complex topics where section boundaries might split relevant info.
        """
        # First, get relevant sections
        relevant_indices = [
            i for i, s in enumerate(sections)
            if s.relevance_score >= self.config.relevance_threshold
        ]

        if not relevant_indices:
            # Fall back to most relevant if nothing meets threshold
            if sections:
                best_idx = max(range(len(sections)),
                              key=lambda i: sections[i].relevance_score)
                relevant_indices = [best_idx]
            else:
                return []

        # Expand to include context window
        expanded_indices: set[int] = set()
        for idx in relevant_indices:
            # Add the relevant section
            expanded_indices.add(idx)

            # Add adjacent sections for context
            for offset in range(1, self.config.context_window + 1):
                # Previous sections
                if idx - offset >= 0:
                    expanded_indices.add(idx - offset)
                # Following sections
                if idx + offset < len(sections):
                    expanded_indices.add(idx + offset)

        # Build result with context marking
        result = []
        for idx in sorted(expanded_indices):
            section = sections[idx]
            is_context = idx not in relevant_indices

            if is_context:
                # Mark as context section with adjusted weight
                context_section = SectionScore(
                    section_id=section.section_id,
                    content=section.content,
                    relevance_score=section.relevance_score * 0.7,  # Reduce weight
                    semantic_similarity=section.semantic_similarity,
                    keyword_overlap=section.keyword_overlap,
                    position_weight=section.position_weight,
                    is_adjacent_context=True,
                    expansion_reason="Adjacent to relevant section",
                )
                result.append(context_section)
            else:
                result.append(section)

        # Limit and sort
        result.sort(key=lambda s: s.relevance_score, reverse=True)
        return result[:self.config.max_sections_per_doc]

    def _calculate_keyword_overlap(self, content: str, question: str) -> float:
        """Calculate keyword overlap score."""
        import re

        # Tokenize and normalize
        def tokenize(text: str) -> set[str]:
            words = re.findall(r'\b\w+\b', text.lower())
            # Remove common stopwords
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                        'been', 'being', 'have', 'has', 'had', 'do', 'does',
                        'did', 'will', 'would', 'could', 'should', 'may',
                        'might', 'must', 'shall', 'can', 'to', 'of', 'in',
                        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                        'through', 'during', 'before', 'after', 'above',
                        'below', 'between', 'under', 'again', 'further',
                        'then', 'once', 'here', 'there', 'when', 'where',
                        'why', 'how', 'all', 'each', 'few', 'more', 'most',
                        'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                        'own', 'same', 'so', 'than', 'too', 'very', 'just',
                        'and', 'but', 'if', 'or', 'because', 'until', 'while',
                        'what', 'which', 'who', 'this', 'that', 'these',
                        'those', 'it', 'its'}
            return {w for w in words if w not in stopwords and len(w) > 2}

        content_tokens = tokenize(content)
        question_tokens = tokenize(question)

        if not question_tokens:
            return 0.0

        # Calculate Jaccard-like overlap
        overlap = len(content_tokens & question_tokens)
        return overlap / len(question_tokens)

    def _calculate_position_weight(self, position: int, total: int) -> float:
        """Calculate position-based weight (earlier = higher)."""
        if total <= 1:
            return 1.0

        # Exponential decay from position 0
        decay = self.config.position_decay ** position
        return decay

    def _calculate_heuristic_semantic_score(
        self,
        content: str,
        question: str,
    ) -> float:
        """
        Calculate semantic similarity using heuristics (no LLM).

        Uses:
        - N-gram overlap
        - Key phrase matching
        - Length normalization
        """
        import re

        # N-gram overlap
        def get_ngrams(text: str, n: int) -> set[str]:
            words = text.lower().split()
            return {' '.join(words[i:i+n]) for i in range(len(words)-n+1)}

        # Calculate overlap for different n-grams
        scores = []
        for n in [2, 3]:
            content_ngrams = get_ngrams(content, n)
            question_ngrams = get_ngrams(question, n)
            if question_ngrams:
                overlap = len(content_ngrams & question_ngrams)
                scores.append(min(1.0, overlap / len(question_ngrams)))

        # Keyword overlap (already calculated, but include for semantic)
        keyword_score = self._calculate_keyword_overlap(content, question)
        scores.append(keyword_score)

        # Length bonus (moderate length sections are often more informative)
        content_words = len(content.split())
        if 50 <= content_words <= 500:
            length_bonus = 0.1
        elif content_words > 500:
            length_bonus = 0.05
        else:
            length_bonus = 0.0

        # Combine scores
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return min(1.0, avg_score + length_bonus)

    def _get_llm_semantic_score(
        self,
        content: str,
        question: str,
        doc_title: str,
    ) -> float:
        """Get semantic relevance score from LLM."""
        if not self.llm:
            return self._calculate_heuristic_semantic_score(content, question)

        # Truncate content if too long
        max_content_length = 1000
        truncated_content = content[:max_content_length]
        if len(content) > max_content_length:
            truncated_content += "..."

        prompt = f"""Rate the relevance of this content to the research question.

Research Question: {question}

Document: {doc_title}

Content:
{truncated_content}

Provide only a single number from 0.0 to 1.0 indicating relevance.
1.0 = Directly answers the question
0.5 = Somewhat relevant
0.0 = Not relevant

Score:"""

        try:
            messages = [LLMMessage(role="user", content=prompt)]
            response = self.llm.generate(messages)

            # Parse score from response
            import re
            match = re.search(r'\b([01]\.?\d*)\b', response.content)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}, falling back to heuristic")

        return self._calculate_heuristic_semantic_score(content, question)

    def _combine_scores(
        self,
        semantic: float,
        keyword: float,
        position: float,
    ) -> float:
        """Combine component scores into final relevance score."""
        # Weighted combination
        weights = {
            'semantic': 0.5,
            'keyword': 0.35,
            'position': 0.15,
        }

        combined = (
            weights['semantic'] * semantic +
            weights['keyword'] * keyword +
            weights['position'] * position
        )

        return min(1.0, combined)

    def _calculate_overall_relevance(
        self,
        sections: list[SectionScore],
    ) -> float:
        """Calculate overall document relevance from sections."""
        if not sections:
            return 0.0

        # Weight by section relevance
        total_weight = sum(s.relevance_score for s in sections)
        if total_weight == 0:
            return 0.0

        # Weighted average with bonus for multiple relevant sections
        avg_relevance = total_weight / len(sections)

        # Bonus for documents with multiple relevant sections
        num_relevant = sum(1 for s in sections
                         if s.relevance_score >= self.config.relevance_threshold)
        coverage_bonus = min(0.1, num_relevant * 0.02)

        return min(1.0, avg_relevance + coverage_bonus)


def create_section_scorer(
    llm: Optional[LLM] = None,
    config: Optional[SectionScoringConfig] = None,
) -> SectionRelevanceScorer:
    """Factory function to create a section relevance scorer."""
    return SectionRelevanceScorer(llm=llm, config=config)


def get_expansion_strategy_description(strategy: ExpansionStrategy) -> str:
    """Get a human-readable description of an expansion strategy."""
    descriptions = {
        ExpansionStrategy.MOST_RELEVANT_ONLY:
            "Only use the single highest-scoring section (focused retrieval)",
        ExpansionStrategy.RELEVANT_SECTIONS:
            "Include all sections above the relevance threshold (comprehensive)",
        ExpansionStrategy.ALL_WITH_WEIGHTS:
            "Include all sections weighted by relevance (exploratory)",
        ExpansionStrategy.DYNAMIC_CONTEXT:
            "Include relevant sections plus adjacent context (context-aware)",
    }
    return descriptions.get(strategy, "Unknown strategy")
