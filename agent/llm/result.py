"""Non-raising agent result types."""

from __future__ import annotations

from dataclasses import dataclass

from .events import StopReason
from .types import Cost, Message, UsageTally


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Structured outcome of an `Agent.run_with_result(...)` invocation."""

    messages: list[Message]
    text: str
    stop_reason: StopReason
    turns: int
    usage: UsageTally
    cost: Cost | None
    error: Exception | None = None
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class AgentResultEvent:
    """Terminal event from `Agent.run_stream_with_result(...)`."""

    result: AgentResult


__all__ = ["AgentResult", "AgentResultEvent"]
