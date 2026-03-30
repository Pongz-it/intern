"""Think Tool Chain Enforcement for Deep Research.

Implements mandatory think tool usage between research operations for non-reasoning models.
Ensures systematic analysis and prevents skipping reflection steps.

Reference: backend/onyx/agents/agent_search/deep/initial/tool_use_check.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from agent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class OperationType(str, Enum):
    """Types of research operations."""
    SEARCH = "search"
    READ = "read"
    THINK = "think"
    REPORT = "report"
    CYCLE_END = "cycle_end"


class ThinkChainViolation(str, Enum):
    """Types of think chain violations."""
    CONSECUTIVE_SEARCHES = "consecutive_searches"
    SEARCH_BEFORE_THINK = "search_before_think"
    REPORT_WITHOUT_THINK = "report_without_think"
    TOO_MANY_SEARCHES = "too_many_searches"


@dataclass
class OperationRecord:
    """Record of a research operation."""
    operation_type: OperationType
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThinkChainViolationRecord:
    """Record of a think chain violation."""
    violation_type: ThinkChainViolation
    message: str
    timestamp: datetime
    operations_since_think: int
    severity: str = "warning"  # warning, error


@dataclass
class ThinkChainConfig:
    """Configuration for think chain enforcement."""
    max_operations_before_think: int = 3  # Max operations between thinks
    require_think_before_report: bool = True
    require_think_after_search: bool = True
    allow_consecutive_reads: bool = True  # Reads don't require intervening thinks
    strict_mode: bool = False  # If True, violations raise exceptions
    log_violations: bool = True


class ThinkChainEnforcer:
    """
    Enforces mandatory think tool usage between research operations.

    For non-reasoning models, ensures that:
    1. Think tool is called between every set of searches
    2. Think tool is called before generating reports
    3. Maximum operations without thinking is enforced
    4. Violations are logged and optionally raise exceptions

    For reasoning models:
    - The enforcer is essentially disabled (permissive mode)
    - Reasoning models have built-in chain-of-thought
    """

    def __init__(
        self,
        config: Optional[ThinkChainConfig] = None,
        is_reasoning_model: bool = False,
    ) -> None:
        """
        Initialize the enforcer.

        Args:
            config: Configuration options
            is_reasoning_model: Whether using a reasoning model (disables enforcement)
        """
        self.config = config or ThinkChainConfig()
        self.is_reasoning_model = is_reasoning_model

        # State tracking
        self.operation_history: list[OperationRecord] = []
        self.violations: list[ThinkChainViolationRecord] = []
        self.operations_since_last_think: int = 0
        self.searches_since_last_think: int = 0
        self.last_operation_type: Optional[OperationType] = None

    def reset(self) -> None:
        """Reset the enforcer state."""
        self.operation_history.clear()
        self.violations.clear()
        self.operations_since_last_think = 0
        self.searches_since_last_think = 0
        self.last_operation_type = None

    def record_operation(
        self,
        operation_type: OperationType,
        details: Optional[dict[str, Any]] = None,
    ) -> list[ThinkChainViolationRecord]:
        """
        Record an operation and check for violations.

        Args:
            operation_type: Type of operation being performed
            details: Optional details about the operation

        Returns:
            List of any violations detected
        """
        now = datetime.now()
        record = OperationRecord(
            operation_type=operation_type,
            timestamp=now,
            details=details or {},
        )
        self.operation_history.append(record)

        # Skip enforcement for reasoning models
        if self.is_reasoning_model:
            self.last_operation_type = operation_type
            return []

        # Check for violations
        new_violations = self._check_violations(operation_type)

        # Update counters
        if operation_type == OperationType.THINK:
            self.operations_since_last_think = 0
            self.searches_since_last_think = 0
        else:
            self.operations_since_last_think += 1
            if operation_type == OperationType.SEARCH:
                self.searches_since_last_think += 1

        self.last_operation_type = operation_type

        # Handle violations
        for violation in new_violations:
            self.violations.append(violation)

            if self.config.log_violations:
                log_msg = f"Think chain violation: {violation.message}"
                if violation.severity == "error":
                    logger.error(log_msg)
                else:
                    logger.warning(log_msg)

            if self.config.strict_mode and violation.severity == "error":
                raise ThinkChainError(violation.message)

        return new_violations

    def _check_violations(
        self,
        operation_type: OperationType,
    ) -> list[ThinkChainViolationRecord]:
        """Check for think chain violations."""
        violations = []
        now = datetime.now()

        # Check consecutive searches without think
        if operation_type == OperationType.SEARCH:
            if self.config.require_think_after_search and self.last_operation_type == OperationType.SEARCH:
                violations.append(ThinkChainViolationRecord(
                    violation_type=ThinkChainViolation.CONSECUTIVE_SEARCHES,
                    message="Consecutive searches without intervening think operation",
                    timestamp=now,
                    operations_since_think=self.operations_since_last_think,
                    severity="warning",
                ))

            # Check too many searches without think
            if self.searches_since_last_think >= self.config.max_operations_before_think:
                violations.append(ThinkChainViolationRecord(
                    violation_type=ThinkChainViolation.TOO_MANY_SEARCHES,
                    message=f"Exceeded max searches ({self.searches_since_last_think}) without think",
                    timestamp=now,
                    operations_since_think=self.operations_since_last_think,
                    severity="error",
                ))

        # Check report without prior think
        if operation_type == OperationType.REPORT:
            if self.config.require_think_before_report:
                # Check if there's been a think after any searches
                if self.searches_since_last_think > 0:
                    violations.append(ThinkChainViolationRecord(
                        violation_type=ThinkChainViolation.REPORT_WITHOUT_THINK,
                        message="Attempting to generate report without thinking about findings",
                        timestamp=now,
                        operations_since_think=self.operations_since_last_think,
                        severity="error",
                    ))

        # Check max operations before think
        if self.operations_since_last_think >= self.config.max_operations_before_think:
            if operation_type not in [OperationType.THINK, OperationType.READ]:
                # Reads can accumulate without requiring thinks
                if not (self.config.allow_consecutive_reads and operation_type == OperationType.READ):
                    violations.append(ThinkChainViolationRecord(
                        violation_type=ThinkChainViolation.TOO_MANY_SEARCHES,
                        message=f"Exceeded max operations ({self.operations_since_last_think}) without think",
                        timestamp=now,
                        operations_since_think=self.operations_since_last_think,
                        severity="warning",
                    ))

        return violations

    def should_think(self) -> bool:
        """
        Check if a think operation should be performed.

        Returns:
            True if think tool should be called
        """
        if self.is_reasoning_model:
            return False

        # Suggest think after any search
        if self.searches_since_last_think > 0:
            return True

        # Suggest think if approaching limit
        if self.operations_since_last_think >= self.config.max_operations_before_think - 1:
            return True

        return False

    def can_report(self) -> bool:
        """
        Check if report generation is allowed.

        Returns:
            True if report can be generated without violation
        """
        if self.is_reasoning_model:
            return True

        if not self.config.require_think_before_report:
            return True

        # Report is allowed if no pending searches without think
        return self.searches_since_last_think == 0

    def get_violation_summary(self) -> dict[str, Any]:
        """Get a summary of all violations."""
        if not self.violations:
            return {"total_violations": 0, "by_type": {}}

        by_type: dict[str, int] = {}
        for v in self.violations:
            by_type[v.violation_type.value] = by_type.get(v.violation_type.value, 0) + 1

        return {
            "total_violations": len(self.violations),
            "by_type": by_type,
            "error_count": sum(1 for v in self.violations if v.severity == "error"),
            "warning_count": sum(1 for v in self.violations if v.severity == "warning"),
        }

    def get_operation_stats(self) -> dict[str, Any]:
        """Get statistics about recorded operations."""
        if not self.operation_history:
            return {"total_operations": 0}

        by_type: dict[str, int] = {}
        for op in self.operation_history:
            by_type[op.operation_type.value] = by_type.get(op.operation_type.value, 0) + 1

        think_count = by_type.get(OperationType.THINK.value, 0)
        search_count = by_type.get(OperationType.SEARCH.value, 0)

        return {
            "total_operations": len(self.operation_history),
            "by_type": by_type,
            "think_ratio": think_count / max(1, search_count),
            "operations_since_last_think": self.operations_since_last_think,
            "searches_since_last_think": self.searches_since_last_think,
        }


class ThinkChainError(Exception):
    """Exception raised when think chain is violated in strict mode."""
    pass


@dataclass
class ThinkChainMiddleware:
    """
    Middleware for integrating think chain enforcement into the research loop.

    Provides hooks for before/after operations to enforce thinking.
    """

    enforcer: ThinkChainEnforcer

    def before_search(self, query: str) -> Optional[str]:
        """
        Called before a search operation.

        Returns:
            Warning message if think should happen first, None otherwise
        """
        if self.enforcer.should_think():
            return (
                f"Warning: Think tool should be called before next search. "
                f"Operations since last think: {self.enforcer.operations_since_last_think}"
            )
        return None

    def after_search(self, query: str, num_results: int) -> list[ThinkChainViolationRecord]:
        """Called after a search operation."""
        return self.enforcer.record_operation(
            OperationType.SEARCH,
            details={"query": query, "num_results": num_results},
        )

    def before_report(self) -> Optional[str]:
        """
        Called before report generation.

        Returns:
            Error message if report is not allowed, None otherwise
        """
        if not self.enforcer.can_report():
            return (
                f"Error: Cannot generate report without thinking about findings. "
                f"Pending searches: {self.enforcer.searches_since_last_think}"
            )
        return None

    def after_report(self) -> list[ThinkChainViolationRecord]:
        """Called after report generation."""
        return self.enforcer.record_operation(OperationType.REPORT)

    def record_think(self, analysis: str) -> list[ThinkChainViolationRecord]:
        """Record a think operation."""
        return self.enforcer.record_operation(
            OperationType.THINK,
            details={"analysis_length": len(analysis)},
        )

    def record_read(self, document_id: str) -> list[ThinkChainViolationRecord]:
        """Record a read operation."""
        return self.enforcer.record_operation(
            OperationType.READ,
            details={"document_id": document_id},
        )


def create_think_chain_enforcer(
    config: Optional[ThinkChainConfig] = None,
    is_reasoning_model: bool = False,
) -> ThinkChainEnforcer:
    """Factory function to create a think chain enforcer."""
    return ThinkChainEnforcer(config=config, is_reasoning_model=is_reasoning_model)


def create_think_chain_middleware(
    enforcer: Optional[ThinkChainEnforcer] = None,
    config: Optional[ThinkChainConfig] = None,
    is_reasoning_model: bool = False,
) -> ThinkChainMiddleware:
    """Factory function to create think chain middleware."""
    if enforcer is None:
        enforcer = create_think_chain_enforcer(
            config=config,
            is_reasoning_model=is_reasoning_model,
        )
    return ThinkChainMiddleware(enforcer=enforcer)
