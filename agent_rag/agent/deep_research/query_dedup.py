"""Query Deduplication and Weight Accumulation for Deep Research.

Implements intelligent query deduplication with:
- Semantic similarity-based deduplication
- Weight accumulation for repeated concepts
- Query history tracking
- Diminishing returns handling

Reference: backend/onyx/agents/agent_search/deep/shared/expanded_retrieval.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import hashlib
import re

from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryRecord:
    """Record of a query with metadata."""
    query: str
    normalized: str
    hash_key: str
    weight: float = 1.0
    execution_count: int = 1
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    source_agents: list[str] = field(default_factory=list)
    results_count: int = 0
    is_duplicate: bool = False
    merged_from: list[str] = field(default_factory=list)


@dataclass
class DeduplicationResult:
    """Result of query deduplication."""
    original_count: int
    deduplicated_count: int
    merged_count: int
    unique_queries: list[QueryRecord]
    duplicate_queries: list[QueryRecord]
    weight_adjustments: dict[str, float]


@dataclass
class QueryDedupConfig:
    """Configuration for query deduplication."""
    # Similarity thresholds
    exact_match_threshold: float = 1.0
    semantic_similarity_threshold: float = 0.85
    keyword_overlap_threshold: float = 0.7

    # Weight accumulation
    weight_accumulation_factor: float = 0.3  # How much to add per duplicate
    max_weight: float = 3.0  # Maximum accumulated weight
    diminishing_returns_factor: float = 0.8  # Decay for repeated queries

    # Normalization
    normalize_case: bool = True
    remove_stopwords: bool = True
    stem_words: bool = False  # Requires stemmer

    # History tracking
    max_history_size: int = 1000
    track_execution_counts: bool = True


class QueryDeduplicator:
    """
    Deduplicates and manages search queries.

    Features:
    - Exact match deduplication
    - Fuzzy/semantic deduplication
    - Weight accumulation for important concepts
    - Diminishing returns for over-queried topics
    - Query history tracking
    """

    def __init__(
        self,
        config: Optional[QueryDedupConfig] = None,
    ) -> None:
        """
        Initialize the deduplicator.

        Args:
            config: Configuration options
        """
        self.config = config or QueryDedupConfig()

        # Query history (hash -> QueryRecord)
        self._history: dict[str, QueryRecord] = {}

        # N-gram index for fast similarity lookup
        self._ngram_index: dict[str, set[str]] = {}  # ngram -> set of query hashes

        # Stopwords for normalization
        self._stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
            'because', 'until', 'while', 'what', 'which', 'who', 'this',
            'that', 'these', 'those', 'it', 'its',
        }

    def reset(self) -> None:
        """Reset the deduplicator state."""
        self._history.clear()
        self._ngram_index.clear()

    def deduplicate(
        self,
        queries: list[str],
        source_agent: Optional[str] = None,
    ) -> DeduplicationResult:
        """
        Deduplicate a list of queries.

        Args:
            queries: List of queries to deduplicate
            source_agent: Optional agent identifier for tracking

        Returns:
            DeduplicationResult with unique and duplicate queries
        """
        unique_queries: list[QueryRecord] = []
        duplicate_queries: list[QueryRecord] = []
        weight_adjustments: dict[str, float] = {}
        merged_count = 0

        for query in queries:
            normalized = self._normalize_query(query)
            hash_key = self._compute_hash(normalized)

            # Check for exact match in history
            if hash_key in self._history:
                existing = self._history[hash_key]
                self._update_existing_query(existing, query, source_agent)
                duplicate_queries.append(QueryRecord(
                    query=query,
                    normalized=normalized,
                    hash_key=hash_key,
                    is_duplicate=True,
                    merged_from=[existing.query],
                ))
                weight_adjustments[query] = existing.weight
                merged_count += 1
                continue

            # Check for semantic similarity
            similar_hash = self._find_similar_query(normalized)
            if similar_hash:
                existing = self._history[similar_hash]
                self._update_existing_query(existing, query, source_agent)
                existing.merged_from.append(query)
                duplicate_queries.append(QueryRecord(
                    query=query,
                    normalized=normalized,
                    hash_key=hash_key,
                    is_duplicate=True,
                    merged_from=[existing.query],
                ))
                weight_adjustments[query] = existing.weight
                merged_count += 1
                continue

            # New unique query
            record = QueryRecord(
                query=query,
                normalized=normalized,
                hash_key=hash_key,
                weight=1.0,
                source_agents=[source_agent] if source_agent else [],
            )

            # Add to history and index
            self._history[hash_key] = record
            self._add_to_ngram_index(normalized, hash_key)

            unique_queries.append(record)

        return DeduplicationResult(
            original_count=len(queries),
            deduplicated_count=len(unique_queries),
            merged_count=merged_count,
            unique_queries=unique_queries,
            duplicate_queries=duplicate_queries,
            weight_adjustments=weight_adjustments,
        )

    def get_weighted_queries(
        self,
        queries: list[str],
        base_weights: Optional[dict[str, float]] = None,
    ) -> list[tuple[str, float]]:
        """
        Get queries with accumulated weights.

        Args:
            queries: Queries to weight
            base_weights: Optional base weights per query

        Returns:
            List of (query, weight) tuples sorted by weight
        """
        base_weights = base_weights or {}
        weighted: list[tuple[str, float]] = []

        for query in queries:
            normalized = self._normalize_query(query)
            hash_key = self._compute_hash(normalized)

            # Get accumulated weight from history
            history_weight = 1.0
            if hash_key in self._history:
                history_weight = self._history[hash_key].weight

            # Check for similar queries
            elif similar_hash := self._find_similar_query(normalized):
                history_weight = self._history[similar_hash].weight * 0.9

            # Combine with base weight
            base_weight = base_weights.get(query, 1.0)
            final_weight = base_weight * history_weight

            weighted.append((query, final_weight))

        # Sort by weight descending
        weighted.sort(key=lambda x: x[1], reverse=True)
        return weighted

    def should_skip_query(self, query: str) -> tuple[bool, str]:
        """
        Check if a query should be skipped due to diminishing returns.

        Args:
            query: Query to check

        Returns:
            Tuple of (should_skip, reason)
        """
        normalized = self._normalize_query(query)
        hash_key = self._compute_hash(normalized)

        if hash_key in self._history:
            record = self._history[hash_key]

            # Check execution count
            if record.execution_count >= 3:
                return True, f"Query executed {record.execution_count} times"

            # Check if results were consistently low
            if record.execution_count >= 2 and record.results_count == 0:
                return True, "Query consistently returned no results"

        return False, ""

    def record_execution(
        self,
        query: str,
        results_count: int,
    ) -> None:
        """
        Record query execution results.

        Args:
            query: The executed query
            results_count: Number of results returned
        """
        if not self.config.track_execution_counts:
            return

        normalized = self._normalize_query(query)
        hash_key = self._compute_hash(normalized)

        if hash_key in self._history:
            record = self._history[hash_key]
            record.execution_count += 1
            record.results_count += results_count
            record.last_seen = datetime.now()

            # Apply diminishing returns to weight
            if record.execution_count > 1:
                record.weight *= self.config.diminishing_returns_factor

    def get_query_stats(self, query: str) -> Optional[dict[str, Any]]:
        """Get statistics for a query."""
        normalized = self._normalize_query(query)
        hash_key = self._compute_hash(normalized)

        if hash_key not in self._history:
            return None

        record = self._history[hash_key]
        return {
            "query": record.query,
            "weight": record.weight,
            "execution_count": record.execution_count,
            "results_count": record.results_count,
            "first_seen": record.first_seen.isoformat(),
            "last_seen": record.last_seen.isoformat(),
            "source_agents": record.source_agents,
            "merged_from": record.merged_from,
        }

    def get_history_summary(self) -> dict[str, Any]:
        """Get summary of query history."""
        if not self._history:
            return {
                "total_queries": 0,
                "total_executions": 0,
            }

        total_executions = sum(r.execution_count for r in self._history.values())
        total_results = sum(r.results_count for r in self._history.values())
        avg_weight = sum(r.weight for r in self._history.values()) / len(self._history)

        # Find most executed queries
        top_executed = sorted(
            self._history.values(),
            key=lambda r: r.execution_count,
            reverse=True
        )[:5]

        # Find highest weighted queries
        top_weighted = sorted(
            self._history.values(),
            key=lambda r: r.weight,
            reverse=True
        )[:5]

        return {
            "total_unique_queries": len(self._history),
            "total_executions": total_executions,
            "total_results": total_results,
            "average_weight": avg_weight,
            "most_executed": [
                {"query": r.query, "count": r.execution_count}
                for r in top_executed
            ],
            "highest_weighted": [
                {"query": r.query, "weight": r.weight}
                for r in top_weighted
            ],
        }

    def _normalize_query(self, query: str) -> str:
        """Normalize a query for comparison."""
        normalized = query.strip()

        if self.config.normalize_case:
            normalized = normalized.lower()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)

        if self.config.remove_stopwords:
            words = normalized.split()
            words = [w for w in words if w not in self._stopwords]
            normalized = ' '.join(words)

        return normalized.strip()

    def _compute_hash(self, normalized: str) -> str:
        """Compute hash for a normalized query."""
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _get_ngrams(self, text: str, n: int = 2) -> set[str]:
        """Get character n-grams from text."""
        text = text.replace(' ', '_')
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    def _add_to_ngram_index(self, normalized: str, hash_key: str) -> None:
        """Add query to n-gram index for similarity lookup."""
        ngrams = self._get_ngrams(normalized)
        for ngram in ngrams:
            if ngram not in self._ngram_index:
                self._ngram_index[ngram] = set()
            self._ngram_index[ngram].add(hash_key)

    def _find_similar_query(self, normalized: str) -> Optional[str]:
        """Find a similar query in history using n-gram similarity."""
        if not self._ngram_index:
            return None

        query_ngrams = self._get_ngrams(normalized)
        if not query_ngrams:
            return None

        # Find candidates that share n-grams
        candidates: dict[str, int] = {}
        for ngram in query_ngrams:
            if ngram in self._ngram_index:
                for hash_key in self._ngram_index[ngram]:
                    candidates[hash_key] = candidates.get(hash_key, 0) + 1

        if not candidates:
            return None

        # Calculate Jaccard similarity
        best_match = None
        best_similarity = 0.0

        for hash_key, shared_count in candidates.items():
            record = self._history.get(hash_key)
            if not record:
                continue

            record_ngrams = self._get_ngrams(record.normalized)
            if not record_ngrams:
                continue

            # Jaccard similarity
            intersection = shared_count
            union = len(query_ngrams) + len(record_ngrams) - intersection
            similarity = intersection / union if union > 0 else 0

            if similarity >= self.config.semantic_similarity_threshold:
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = hash_key

        return best_match

    def _calculate_keyword_overlap(self, query1: str, query2: str) -> float:
        """Calculate keyword overlap between two queries."""
        words1 = set(query1.split())
        words2 = set(query2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _update_existing_query(
        self,
        existing: QueryRecord,
        new_query: str,
        source_agent: Optional[str],
    ) -> None:
        """Update an existing query record with new occurrence."""
        existing.last_seen = datetime.now()

        # Accumulate weight (with cap)
        new_weight = existing.weight + self.config.weight_accumulation_factor
        existing.weight = min(new_weight, self.config.max_weight)

        # Track source agent
        if source_agent and source_agent not in existing.source_agents:
            existing.source_agents.append(source_agent)

        logger.debug(
            f"Query merged: '{new_query}' -> '{existing.query}', "
            f"weight now {existing.weight:.2f}"
        )


def create_query_deduplicator(
    config: Optional[QueryDedupConfig] = None,
) -> QueryDeduplicator:
    """Factory function to create a query deduplicator."""
    return QueryDeduplicator(config=config)
