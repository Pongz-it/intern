"""Token Budget Management for Deep Research.

Implements comprehensive token budget tracking and allocation for:
- Context window management
- Per-component budget allocation
- Dynamic reallocation based on phase
- Overflow handling strategies

Reference: backend/onyx/agents/agent_search/deep/shared/utils.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class BudgetCategory(str, Enum):
    """Categories of token budget allocation."""
    SYSTEM_PROMPT = "system_prompt"
    USER_CONTEXT = "user_context"
    SEARCH_RESULTS = "search_results"
    FINDINGS = "findings"
    CITATIONS = "citations"
    THINK_ANALYSIS = "think_analysis"
    REPORT_GENERATION = "report_generation"
    CONVERSATION_HISTORY = "conversation_history"
    RESERVED = "reserved"  # Safety buffer


class BudgetPhase(str, Enum):
    """Research phases with different budget priorities."""
    INITIALIZATION = "initialization"
    RESEARCH = "research"
    SYNTHESIS = "synthesis"
    REPORT = "report"


class OverflowStrategy(str, Enum):
    """Strategies for handling budget overflow."""
    TRUNCATE = "truncate"  # Simple truncation
    SUMMARIZE = "summarize"  # LLM-based summarization
    PRIORITIZE = "prioritize"  # Keep highest priority content
    SLIDE_WINDOW = "slide_window"  # Sliding window approach
    COMPRESS = "compress"  # Remove redundancy


@dataclass
class BudgetAllocation:
    """Token budget allocation for a category."""
    category: BudgetCategory
    allocated: int
    used: int = 0
    priority: int = 1  # Higher = more important
    can_overflow: bool = False  # Can borrow from other categories
    overflow_source: Optional[BudgetCategory] = None

    @property
    def remaining(self) -> int:
        return max(0, self.allocated - self.used)

    @property
    def utilization(self) -> float:
        return self.used / self.allocated if self.allocated > 0 else 0.0

    @property
    def is_exceeded(self) -> bool:
        return self.used > self.allocated


@dataclass
class BudgetSnapshot:
    """Snapshot of budget state at a point in time."""
    timestamp: datetime
    phase: BudgetPhase
    total_budget: int
    total_used: int
    allocations: dict[str, dict[str, Any]]
    overflow_events: int = 0


@dataclass
class TokenBudgetConfig:
    """Configuration for token budget management."""
    # Total budget
    total_context_limit: int = 128000  # Model's context window
    reserved_for_output: int = 4000  # Reserved for model output

    # Phase allocations (percentages of available budget)
    initialization_allocation: dict[str, float] = field(default_factory=lambda: {
        BudgetCategory.SYSTEM_PROMPT.value: 0.05,
        BudgetCategory.USER_CONTEXT.value: 0.10,
        BudgetCategory.RESERVED.value: 0.05,
    })

    research_allocation: dict[str, float] = field(default_factory=lambda: {
        BudgetCategory.SYSTEM_PROMPT.value: 0.05,
        BudgetCategory.SEARCH_RESULTS.value: 0.40,
        BudgetCategory.FINDINGS.value: 0.25,
        BudgetCategory.THINK_ANALYSIS.value: 0.15,
        BudgetCategory.RESERVED.value: 0.05,
    })

    synthesis_allocation: dict[str, float] = field(default_factory=lambda: {
        BudgetCategory.SYSTEM_PROMPT.value: 0.05,
        BudgetCategory.FINDINGS.value: 0.50,
        BudgetCategory.CITATIONS.value: 0.20,
        BudgetCategory.RESERVED.value: 0.05,
    })

    report_allocation: dict[str, float] = field(default_factory=lambda: {
        BudgetCategory.SYSTEM_PROMPT.value: 0.05,
        BudgetCategory.FINDINGS.value: 0.35,
        BudgetCategory.CITATIONS.value: 0.15,
        BudgetCategory.REPORT_GENERATION.value: 0.35,
        BudgetCategory.RESERVED.value: 0.05,
    })

    # Overflow settings
    default_overflow_strategy: OverflowStrategy = OverflowStrategy.PRIORITIZE
    allow_overflow_borrowing: bool = True
    max_overflow_ratio: float = 0.2  # Max 20% overflow

    # Warning thresholds
    warning_threshold: float = 0.8  # Warn at 80% usage
    critical_threshold: float = 0.95  # Critical at 95% usage


class TokenBudgetManager:
    """
    Manages token budgets across deep research phases.

    Features:
    - Phase-aware budget allocation
    - Dynamic reallocation
    - Overflow handling
    - Usage tracking and warnings
    """

    def __init__(
        self,
        config: Optional[TokenBudgetConfig] = None,
    ) -> None:
        """
        Initialize the budget manager.

        Args:
            config: Configuration options
        """
        self.config = config or TokenBudgetConfig()

        # Available budget (total - reserved for output)
        self.available_budget = (
            self.config.total_context_limit -
            self.config.reserved_for_output
        )

        # Current allocations
        self.allocations: dict[BudgetCategory, BudgetAllocation] = {}

        # State
        self.current_phase = BudgetPhase.INITIALIZATION
        self.overflow_events: list[dict[str, Any]] = []
        self.snapshots: list[BudgetSnapshot] = []

        # Initialize with default phase
        self._apply_phase_allocation(BudgetPhase.INITIALIZATION)

    def reset(self) -> None:
        """Reset the budget manager."""
        self.allocations.clear()
        self.overflow_events.clear()
        self.snapshots.clear()
        self.current_phase = BudgetPhase.INITIALIZATION
        self._apply_phase_allocation(BudgetPhase.INITIALIZATION)

    def set_phase(self, phase: BudgetPhase) -> None:
        """
        Set the current research phase and reallocate budget.

        Args:
            phase: The new phase
        """
        if phase != self.current_phase:
            # Take snapshot before changing
            self._take_snapshot()

            # Apply new phase allocation
            self._apply_phase_allocation(phase)
            self.current_phase = phase

            logger.info(f"Budget phase changed to {phase.value}")

    def allocate(
        self,
        category: BudgetCategory,
        tokens: int,
        priority: int = 1,
        can_overflow: bool = False,
    ) -> bool:
        """
        Allocate tokens to a category.

        Args:
            category: Budget category
            tokens: Number of tokens to allocate
            priority: Priority level
            can_overflow: Whether this can borrow from other categories

        Returns:
            True if allocation succeeded
        """
        if category not in self.allocations:
            self.allocations[category] = BudgetAllocation(
                category=category,
                allocated=tokens,
                priority=priority,
                can_overflow=can_overflow,
            )
        else:
            self.allocations[category].allocated = tokens
            self.allocations[category].priority = priority
            self.allocations[category].can_overflow = can_overflow

        return True

    def use(
        self,
        category: BudgetCategory,
        tokens: int,
        content_id: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Use tokens from a category.

        Args:
            category: Budget category
            tokens: Number of tokens to use
            content_id: Optional identifier for the content

        Returns:
            Tuple of (success, warning_message)
        """
        if category not in self.allocations:
            # Auto-allocate if not exists
            self.allocate(category, tokens)

        allocation = self.allocations[category]
        old_used = allocation.used
        allocation.used += tokens

        warning = None

        # Check for overflow
        if allocation.is_exceeded:
            warning = self._handle_overflow(category, allocation)

        # Check thresholds
        elif allocation.utilization >= self.config.critical_threshold:
            warning = f"Critical: {category.value} at {allocation.utilization:.0%} capacity"
        elif allocation.utilization >= self.config.warning_threshold:
            warning = f"Warning: {category.value} at {allocation.utilization:.0%} capacity"

        if warning:
            logger.warning(warning)

        return True, warning

    def release(
        self,
        category: BudgetCategory,
        tokens: int,
    ) -> None:
        """
        Release tokens back to a category.

        Args:
            category: Budget category
            tokens: Number of tokens to release
        """
        if category in self.allocations:
            self.allocations[category].used = max(
                0,
                self.allocations[category].used - tokens
            )

    def get_remaining(self, category: BudgetCategory) -> int:
        """Get remaining tokens for a category."""
        if category in self.allocations:
            return self.allocations[category].remaining
        return 0

    def get_total_remaining(self) -> int:
        """Get total remaining tokens across all categories."""
        return sum(a.remaining for a in self.allocations.values())

    def get_total_used(self) -> int:
        """Get total used tokens across all categories."""
        return sum(a.used for a in self.allocations.values())

    def can_use(
        self,
        category: BudgetCategory,
        tokens: int,
        allow_overflow: bool = False,
    ) -> bool:
        """
        Check if tokens can be used from a category.

        Args:
            category: Budget category
            tokens: Number of tokens
            allow_overflow: Whether to allow overflow

        Returns:
            True if tokens are available
        """
        if category not in self.allocations:
            return True  # Will be auto-allocated

        allocation = self.allocations[category]

        if allocation.remaining >= tokens:
            return True

        if allow_overflow and self.config.allow_overflow_borrowing:
            # Check if we can borrow from other categories
            borrowable = self._get_borrowable_tokens(category)
            return (allocation.remaining + borrowable) >= tokens

        return False

    def get_budget_for_content(
        self,
        category: BudgetCategory,
        priority: int = 1,
    ) -> int:
        """
        Get available budget for new content.

        Args:
            category: Budget category
            priority: Content priority

        Returns:
            Available tokens
        """
        if category not in self.allocations:
            return 0

        allocation = self.allocations[category]
        available = allocation.remaining

        # Higher priority content can get more
        if priority > 1 and self.config.allow_overflow_borrowing:
            borrowable = self._get_borrowable_tokens(category)
            available += borrowable * (priority / 10)  # Scale by priority

        return int(available)

    def suggest_content_length(
        self,
        category: BudgetCategory,
        num_items: int,
    ) -> int:
        """
        Suggest content length per item given budget.

        Args:
            category: Budget category
            num_items: Number of items to fit

        Returns:
            Suggested tokens per item
        """
        available = self.get_remaining(category)
        if num_items <= 0:
            return available

        per_item = available // num_items
        return max(100, per_item)  # Minimum 100 tokens per item

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Simple heuristic: ~4 characters per token for English
        return len(text) // 4

    def get_status(self) -> dict[str, Any]:
        """Get current budget status."""
        status = {
            "phase": self.current_phase.value,
            "available_budget": self.available_budget,
            "total_used": self.get_total_used(),
            "total_remaining": self.get_total_remaining(),
            "utilization": self.get_total_used() / self.available_budget,
            "categories": {},
            "warnings": [],
            "overflow_events": len(self.overflow_events),
        }

        for category, allocation in self.allocations.items():
            status["categories"][category.value] = {
                "allocated": allocation.allocated,
                "used": allocation.used,
                "remaining": allocation.remaining,
                "utilization": allocation.utilization,
                "priority": allocation.priority,
            }

            if allocation.utilization >= self.config.critical_threshold:
                status["warnings"].append(
                    f"{category.value}: CRITICAL ({allocation.utilization:.0%})"
                )
            elif allocation.utilization >= self.config.warning_threshold:
                status["warnings"].append(
                    f"{category.value}: Warning ({allocation.utilization:.0%})"
                )

        return status

    def _apply_phase_allocation(self, phase: BudgetPhase) -> None:
        """Apply budget allocation for a phase."""
        # Get phase-specific allocation
        if phase == BudgetPhase.INITIALIZATION:
            allocation_config = self.config.initialization_allocation
        elif phase == BudgetPhase.RESEARCH:
            allocation_config = self.config.research_allocation
        elif phase == BudgetPhase.SYNTHESIS:
            allocation_config = self.config.synthesis_allocation
        elif phase == BudgetPhase.REPORT:
            allocation_config = self.config.report_allocation
        else:
            allocation_config = self.config.research_allocation

        # Calculate and apply allocations
        for category_str, percentage in allocation_config.items():
            category = BudgetCategory(category_str)
            tokens = int(self.available_budget * percentage)

            # Preserve usage if category exists
            current_used = 0
            if category in self.allocations:
                current_used = self.allocations[category].used

            self.allocations[category] = BudgetAllocation(
                category=category,
                allocated=tokens,
                used=current_used,
                priority=self._get_category_priority(category),
                can_overflow=category not in [
                    BudgetCategory.SYSTEM_PROMPT,
                    BudgetCategory.RESERVED,
                ],
            )

    def _get_category_priority(self, category: BudgetCategory) -> int:
        """Get priority for a budget category."""
        priorities = {
            BudgetCategory.SYSTEM_PROMPT: 10,
            BudgetCategory.USER_CONTEXT: 8,
            BudgetCategory.FINDINGS: 7,
            BudgetCategory.SEARCH_RESULTS: 6,
            BudgetCategory.CITATIONS: 5,
            BudgetCategory.THINK_ANALYSIS: 4,
            BudgetCategory.REPORT_GENERATION: 7,
            BudgetCategory.CONVERSATION_HISTORY: 3,
            BudgetCategory.RESERVED: 9,
        }
        return priorities.get(category, 5)

    def _handle_overflow(
        self,
        category: BudgetCategory,
        allocation: BudgetAllocation,
    ) -> str:
        """Handle budget overflow."""
        overflow_amount = allocation.used - allocation.allocated

        self.overflow_events.append({
            "timestamp": datetime.now().isoformat(),
            "category": category.value,
            "overflow_amount": overflow_amount,
            "strategy": self.config.default_overflow_strategy.value,
        })

        if self.config.allow_overflow_borrowing:
            # Try to borrow from lower priority categories
            borrowed = self._borrow_from_others(category, overflow_amount)
            if borrowed >= overflow_amount:
                return f"Overflow in {category.value}: borrowed {borrowed} tokens"

        return f"Overflow in {category.value}: {overflow_amount} tokens over budget"

    def _get_borrowable_tokens(self, exclude_category: BudgetCategory) -> int:
        """Get tokens that can be borrowed from other categories."""
        borrowable = 0
        for cat, alloc in self.allocations.items():
            if cat == exclude_category:
                continue
            if cat in [BudgetCategory.SYSTEM_PROMPT, BudgetCategory.RESERVED]:
                continue
            if alloc.remaining > 0:
                # Can borrow up to max_overflow_ratio of remaining
                borrowable += int(alloc.remaining * self.config.max_overflow_ratio)
        return borrowable

    def _borrow_from_others(
        self,
        borrower: BudgetCategory,
        amount: int,
    ) -> int:
        """Borrow tokens from other categories."""
        borrowed = 0
        borrower_priority = self.allocations[borrower].priority

        # Sort categories by priority (lower first, so we borrow from less important)
        sorted_cats = sorted(
            self.allocations.items(),
            key=lambda x: x[1].priority
        )

        for cat, alloc in sorted_cats:
            if cat == borrower:
                continue
            if cat in [BudgetCategory.SYSTEM_PROMPT, BudgetCategory.RESERVED]:
                continue
            if alloc.priority >= borrower_priority:
                continue  # Don't borrow from higher/equal priority

            available = alloc.remaining
            if available <= 0:
                continue

            can_borrow = int(available * self.config.max_overflow_ratio)
            to_borrow = min(can_borrow, amount - borrowed)

            if to_borrow > 0:
                alloc.allocated -= to_borrow  # Reduce allocation
                self.allocations[borrower].allocated += to_borrow  # Add to borrower
                borrowed += to_borrow

            if borrowed >= amount:
                break

        return borrowed

    def _take_snapshot(self) -> None:
        """Take a snapshot of current budget state."""
        snapshot = BudgetSnapshot(
            timestamp=datetime.now(),
            phase=self.current_phase,
            total_budget=self.available_budget,
            total_used=self.get_total_used(),
            allocations={
                cat.value: {
                    "allocated": alloc.allocated,
                    "used": alloc.used,
                    "remaining": alloc.remaining,
                }
                for cat, alloc in self.allocations.items()
            },
            overflow_events=len(self.overflow_events),
        )
        self.snapshots.append(snapshot)


def create_token_budget_manager(
    config: Optional[TokenBudgetConfig] = None,
) -> TokenBudgetManager:
    """Factory function to create a token budget manager."""
    return TokenBudgetManager(config=config)


def estimate_tokens(text: str) -> int:
    """Utility function to estimate token count."""
    return len(text) // 4
