from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AgentEvent:
    pass


@dataclass
class SubAgentStartEvent(AgentEvent):
    """Always the first event. Carries the subagent name for the TUI header."""
    name: str = ""
    description: str = ""
    color: str = ""


@dataclass
class LogEvent(AgentEvent):
    """A timestep worth showing in the conversation (file read, scan step, etc.)."""
    message: str = ""
    tool_name: str = ""


@dataclass
class DiffEvent(AgentEvent):
    """Diff of an edit_file mutation; emitted after ToolExecutionCompleted."""
    path: str = ""
    diff_lines: list = field(default_factory=list)  # list[DiffLine]


@dataclass
class InferEndEvent(AgentEvent):
    """An LLM call finished. TUI should show token counts."""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class StatusUpdateEvent(AgentEvent):
    """Progress update. TUI shows a determinate ProgressBar; removed on DoneEvent."""
    progress: int = 0
    total: int | None = None


@dataclass
class ThinkingTokenEvent(AgentEvent):
    """A chunk of thinking/reasoning tokens from the LLM."""
    text: str = ""


@dataclass
class DoneEvent(AgentEvent):
    """Always the last event."""
    thinking_chars: int = 0
    files_touched: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SubagentResult:
    """Result of one hosted delegation: a subagent's summary, the files it touched, and whether it completed normally."""
    summary: str = ""
    files_touched: list[str] = field(default_factory=list)
    status: Literal["ok", "failed"] = "ok"
    budget_exhausted: bool = False


@dataclass
class DelegationStartEvent(AgentEvent):
    """A `delegate` tool call is starting a nested specialist run — the TUI
    opens a distinct badge block for it, separate from the outer header."""
    agent_name: str = ""
    task: str = ""
    mission: str = ""


@dataclass
class DelegationDoneEvent(AgentEvent):
    """The nested run from a `DelegationStartEvent` has finished — the TUI
    closes that badge block and resumes attributing events to the outer
    (pre-delegation) renderer."""
    agent_name: str = ""


@dataclass
class MaxIterationsEvent(AgentEvent):
    """Emitted instead of a text response when the agent hit its iteration limit."""
    pass


@dataclass
class BudgetExhaustedEvent(AgentEvent):
    """Emitted when the agent exhausted its iteration budget but a salvage turn (with tools suppressed) was attempted."""
    pass


@dataclass
class EstimateEvent(AgentEvent):
    """Result of the trivial-vs-mutate scope estimate (plan 27 improvement 5)."""
    decision: str = ""  # "trivial" | "mutate" | "seeded" | "skipped"
    specialists: list[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class PlanStartedEvent(AgentEvent):
    """Telemetry: the planner produced a validated plan (plan 27 improvement 6)."""
    step_count: int = 0
    agents: list[str] = field(default_factory=list)
    verify_placements: int = 0


@dataclass
class PlanHaltedEvent(AgentEvent):
    """A plan step failed verification twice (or dispatched empty output);
    the interpreter halted in place — completed steps' work is kept, no rollback."""
    step_index: int = 0
    agent: str = ""
    reason: str = ""
