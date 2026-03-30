"""Reciprocal Rank Fusion (RRF) Optimizer for Deep Research.

Implements adaptive RRF with:
- Configurable k parameter
- Source-type weighted fusion
- Dynamic parameter tuning
- Multi-query result combination

Reference: backend/onyx/agents/agent_search/deep/shared/expanded_retrieval.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

from agent_rag.core.models import Chunk
from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


T = TypeVar('T')


class SourceType(str, Enum):
    """Types of search result sources."""
    INTERNAL_KNOWLEDGE = "internal"
    WEB_SEARCH = "web"
    URL_CONTENT = "url"
    SEMANTIC_SEARCH = "semantic"
    KEYWORD_SEARCH = "keyword"
    ENTITY_SEARCH = "entity"


@dataclass
class RankedResult(Generic[T]):
    """A result with ranking information."""
    item: T
    original_rank: int
    source_type: SourceType
    query: str
    raw_score: float = 0.0
    rrf_score: float = 0.0
    combined_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedResult(Generic[T]):
    """A result after RRF fusion."""
    item: T
    final_score: float
    contributing_queries: list[str]
    source_types: list[SourceType]
    rank_contributions: dict[str, int]  # query -> original_rank
    score_breakdown: dict[str, float]  # component -> score


@dataclass
class RRFConfig:
    """Configuration for RRF optimization."""
    # Standard RRF parameter
    k: int = 60  # Classic RRF uses k=60

    # Source type weights (multiply RRF score by these)
    source_weights: dict[str, float] = field(default_factory=lambda: {
        SourceType.INTERNAL_KNOWLEDGE.value: 1.2,  # Boost internal docs
        SourceType.WEB_SEARCH.value: 1.0,
        SourceType.URL_CONTENT.value: 1.1,
        SourceType.SEMANTIC_SEARCH.value: 1.0,
        SourceType.KEYWORD_SEARCH.value: 0.9,
        SourceType.ENTITY_SEARCH.value: 0.95,
    })

    # Query weights (for multi-query fusion)
    default_query_weight: float = 1.0

    # Fusion parameters
    min_occurrences: int = 1  # Min queries a result must appear in
    score_normalization: bool = True
    dedup_by_content: bool = True

    # Adaptive tuning
    adaptive_k: bool = True
    k_min: int = 30
    k_max: int = 120


class RRFOptimizer(Generic[T]):
    """
    Optimizes Reciprocal Rank Fusion for combining search results.

    Features:
    - Configurable k parameter
    - Source-type weighted fusion
    - Multi-query result combination
    - Adaptive parameter tuning
    - Content-based deduplication
    """

    def __init__(
        self,
        config: Optional[RRFConfig] = None,
    ) -> None:
        """
        Initialize the RRF optimizer.

        Args:
            config: Configuration options
        """
        self.config = config or RRFConfig()

    def fuse_results(
        self,
        result_lists: list[list[T]],
        queries: list[str],
        source_types: list[SourceType],
        query_weights: Optional[list[float]] = None,
        get_id_func: Optional[callable] = None,
    ) -> list[FusedResult[T]]:
        """
        Fuse multiple ranked result lists using RRF.

        Args:
            result_lists: List of ranked result lists
            queries: Corresponding queries for each result list
            source_types: Source type for each result list
            query_weights: Optional weights per query
            get_id_func: Function to get unique ID from item (for dedup)

        Returns:
            Fused and re-ranked results
        """
        if not result_lists or len(result_lists) != len(queries):
            return []

        # Default weights
        if query_weights is None:
            query_weights = [self.config.default_query_weight] * len(queries)

        # Get effective k (adaptive or fixed)
        k = self._get_effective_k(result_lists)

        # Calculate RRF scores for each result
        score_map: dict[str, dict[str, Any]] = {}  # id -> {scores, item, metadata}

        for i, (results, query, source_type, weight) in enumerate(
            zip(result_lists, queries, source_types, query_weights)
        ):
            source_weight = self.config.source_weights.get(
                source_type.value, 1.0
            )

            for rank, item in enumerate(results, 1):
                # Get item ID for deduplication
                item_id = self._get_item_id(item, get_id_func)

                # Calculate RRF score
                rrf_score = 1.0 / (k + rank)
                weighted_score = rrf_score * weight * source_weight

                if item_id not in score_map:
                    score_map[item_id] = {
                        "item": item,
                        "total_score": 0.0,
                        "queries": [],
                        "source_types": [],
                        "rank_contributions": {},
                        "score_breakdown": {},
                    }

                score_map[item_id]["total_score"] += weighted_score
                score_map[item_id]["queries"].append(query)
                score_map[item_id]["source_types"].append(source_type)
                score_map[item_id]["rank_contributions"][query] = rank
                score_map[item_id]["score_breakdown"][f"{query}_{source_type.value}"] = weighted_score

        # Filter by minimum occurrences
        filtered = {
            id_: data for id_, data in score_map.items()
            if len(data["queries"]) >= self.config.min_occurrences
        }

        # Normalize scores if configured
        if self.config.score_normalization and filtered:
            max_score = max(d["total_score"] for d in filtered.values())
            if max_score > 0:
                for data in filtered.values():
                    data["total_score"] /= max_score

        # Create fused results
        fused_results = [
            FusedResult(
                item=data["item"],
                final_score=data["total_score"],
                contributing_queries=list(set(data["queries"])),
                source_types=list(set(data["source_types"])),
                rank_contributions=data["rank_contributions"],
                score_breakdown=data["score_breakdown"],
            )
            for data in filtered.values()
        ]

        # Sort by final score
        fused_results.sort(key=lambda x: x.final_score, reverse=True)

        return fused_results

    def fuse_chunks(
        self,
        chunk_lists: list[list[Chunk]],
        queries: list[str],
        source_types: list[SourceType],
        query_weights: Optional[list[float]] = None,
    ) -> list[FusedResult[Chunk]]:
        """
        Convenience method for fusing chunk results.

        Args:
            chunk_lists: List of chunk result lists
            queries: Corresponding queries
            source_types: Source types
            query_weights: Optional query weights

        Returns:
            Fused chunk results
        """
        return self.fuse_results(
            result_lists=chunk_lists,
            queries=queries,
            source_types=source_types,
            query_weights=query_weights,
            get_id_func=lambda c: c.unique_id,
        )

    def calculate_rrf_score(
        self,
        rank: int,
        source_type: Optional[SourceType] = None,
        query_weight: float = 1.0,
        k: Optional[int] = None,
    ) -> float:
        """
        Calculate RRF score for a single result.

        Args:
            rank: Position in ranked list (1-indexed)
            source_type: Type of source
            query_weight: Weight for the query
            k: Optional k parameter override

        Returns:
            RRF score
        """
        k = k or self.config.k
        base_score = 1.0 / (k + rank)

        # Apply source weight
        if source_type:
            source_weight = self.config.source_weights.get(
                source_type.value, 1.0
            )
            base_score *= source_weight

        return base_score * query_weight

    def optimize_k(
        self,
        result_lists: list[list[T]],
        relevance_labels: Optional[list[list[bool]]] = None,
    ) -> int:
        """
        Find optimal k parameter for given results.

        Args:
            result_lists: Result lists to optimize for
            relevance_labels: Optional ground truth relevance labels

        Returns:
            Optimal k value
        """
        if not self.config.adaptive_k:
            return self.config.k

        # Heuristic-based optimization
        total_results = sum(len(r) for r in result_lists)
        num_lists = len(result_lists)

        if num_lists == 0:
            return self.config.k

        avg_list_length = total_results / num_lists

        # Adaptive k based on result characteristics
        # Shorter lists benefit from smaller k
        # Longer lists need larger k to differentiate
        if avg_list_length < 10:
            k = self.config.k_min
        elif avg_list_length > 100:
            k = self.config.k_max
        else:
            # Linear interpolation
            ratio = (avg_list_length - 10) / 90
            k = int(self.config.k_min + ratio * (self.config.k_max - self.config.k_min))

        # If we have relevance labels, could do more sophisticated optimization
        # For now, return the heuristic k
        return k

    def get_score_distribution(
        self,
        results: list[FusedResult[T]],
    ) -> dict[str, Any]:
        """
        Analyze score distribution of fused results.

        Args:
            results: Fused results to analyze

        Returns:
            Distribution statistics
        """
        if not results:
            return {"count": 0}

        scores = [r.final_score for r in results]

        return {
            "count": len(scores),
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
            "median": sorted(scores)[len(scores) // 2],
            "top_5_scores": scores[:5],
            "query_overlap": {
                "single_query": sum(1 for r in results if len(r.contributing_queries) == 1),
                "multi_query": sum(1 for r in results if len(r.contributing_queries) > 1),
            },
        }

    def rerank_by_source_diversity(
        self,
        results: list[FusedResult[T]],
        diversity_weight: float = 0.3,
    ) -> list[FusedResult[T]]:
        """
        Re-rank results to promote source diversity.

        Args:
            results: Fused results
            diversity_weight: Weight for diversity bonus

        Returns:
            Re-ranked results
        """
        if not results:
            return []

        # Track source type usage
        source_counts: dict[SourceType, int] = {}

        reranked = []
        for result in results:
            # Calculate diversity bonus
            primary_source = result.source_types[0] if result.source_types else None
            if primary_source:
                count = source_counts.get(primary_source, 0)
                diversity_bonus = diversity_weight / (1 + count)
                source_counts[primary_source] = count + 1
            else:
                diversity_bonus = 0

            # Adjust score
            adjusted_score = result.final_score * (1 + diversity_bonus)

            reranked.append(FusedResult(
                item=result.item,
                final_score=adjusted_score,
                contributing_queries=result.contributing_queries,
                source_types=result.source_types,
                rank_contributions=result.rank_contributions,
                score_breakdown={
                    **result.score_breakdown,
                    "diversity_bonus": diversity_bonus,
                },
            ))

        # Re-sort
        reranked.sort(key=lambda x: x.final_score, reverse=True)
        return reranked

    def _get_effective_k(self, result_lists: list[list[T]]) -> int:
        """Get effective k parameter."""
        if self.config.adaptive_k:
            return self.optimize_k(result_lists)
        return self.config.k

    def _get_item_id(
        self,
        item: T,
        get_id_func: Optional[callable],
    ) -> str:
        """Get unique ID for an item."""
        if get_id_func:
            return str(get_id_func(item))

        # Try common ID attributes
        if hasattr(item, 'unique_id'):
            return str(item.unique_id)
        if hasattr(item, 'id'):
            return str(item.id)
        if hasattr(item, 'document_id'):
            return str(item.document_id)

        # Fallback to hash
        return str(hash(str(item)))


@dataclass
class RRFParameterTuner:
    """
    Tunes RRF parameters based on result characteristics.

    Can be used to find optimal parameters for specific use cases.
    """

    def tune_for_precision(self) -> RRFConfig:
        """Get config optimized for precision (fewer, more relevant results)."""
        return RRFConfig(
            k=30,  # Lower k emphasizes top ranks
            min_occurrences=2,  # Require multiple query matches
            source_weights={
                SourceType.INTERNAL_KNOWLEDGE.value: 1.3,
                SourceType.WEB_SEARCH.value: 0.9,
                SourceType.URL_CONTENT.value: 1.0,
                SourceType.SEMANTIC_SEARCH.value: 1.1,
                SourceType.KEYWORD_SEARCH.value: 0.8,
                SourceType.ENTITY_SEARCH.value: 0.9,
            },
        )

    def tune_for_recall(self) -> RRFConfig:
        """Get config optimized for recall (more comprehensive results)."""
        return RRFConfig(
            k=100,  # Higher k flattens rank differences
            min_occurrences=1,  # Include single-query matches
            source_weights={
                SourceType.INTERNAL_KNOWLEDGE.value: 1.0,
                SourceType.WEB_SEARCH.value: 1.1,
                SourceType.URL_CONTENT.value: 1.0,
                SourceType.SEMANTIC_SEARCH.value: 1.0,
                SourceType.KEYWORD_SEARCH.value: 1.0,
                SourceType.ENTITY_SEARCH.value: 1.0,
            },
        )

    def tune_for_diversity(self) -> RRFConfig:
        """Get config optimized for result diversity."""
        return RRFConfig(
            k=60,
            min_occurrences=1,
            source_weights={
                # Equal weights for diversity
                SourceType.INTERNAL_KNOWLEDGE.value: 1.0,
                SourceType.WEB_SEARCH.value: 1.0,
                SourceType.URL_CONTENT.value: 1.0,
                SourceType.SEMANTIC_SEARCH.value: 1.0,
                SourceType.KEYWORD_SEARCH.value: 1.0,
                SourceType.ENTITY_SEARCH.value: 1.0,
            },
            dedup_by_content=True,
        )

    def tune_for_research(self) -> RRFConfig:
        """Get config optimized for deep research use case."""
        return RRFConfig(
            k=60,
            min_occurrences=1,
            source_weights={
                SourceType.INTERNAL_KNOWLEDGE.value: 1.2,  # Boost internal
                SourceType.WEB_SEARCH.value: 1.0,
                SourceType.URL_CONTENT.value: 1.1,  # URL content is valuable
                SourceType.SEMANTIC_SEARCH.value: 1.05,
                SourceType.KEYWORD_SEARCH.value: 0.95,
                SourceType.ENTITY_SEARCH.value: 1.0,
            },
            adaptive_k=True,
            k_min=40,
            k_max=80,
        )


def create_rrf_optimizer(
    config: Optional[RRFConfig] = None,
) -> RRFOptimizer:
    """Factory function to create an RRF optimizer."""
    return RRFOptimizer(config=config)


def create_research_rrf_optimizer() -> RRFOptimizer:
    """Create an RRF optimizer tuned for research."""
    tuner = RRFParameterTuner()
    config = tuner.tune_for_research()
    return RRFOptimizer(config=config)
