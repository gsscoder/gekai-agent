"""Exception types for llmstitch."""

from __future__ import annotations


class MaxIterationsExceeded(RuntimeError):
    """Raised when the agent loop hits `max_iterations` without producing a final response."""


class CostCeilingExceeded(RuntimeError):
    """Raised when an agent's accumulated cost exceeds its configured ceiling.

    Carries `spent` and `ceiling` (USD) so callers can surface them without
    re-parsing the message.
    """

    def __init__(self, spent: float, ceiling: float) -> None:
        super().__init__(f"Cost ceiling exceeded: spent ${spent:.6f} > ${ceiling:.6f}")
        self.spent = spent
        self.ceiling = ceiling


__all__ = ["CostCeilingExceeded", "MaxIterationsExceeded"]
