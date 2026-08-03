from __future__ import annotations

from dataclasses import dataclass, field


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
class TaskGraphStartedEvent(AgentEvent):
    """Telemetry: the sequencer produced a validated task graph (plan 27 improvement 6)."""
    step_count: int = 0
    agents: list[str] = field(default_factory=list)
    verify_placements: int = 0


@dataclass
class TaskGraphHaltedEvent(AgentEvent):
    """A task graph step failed verification twice (or dispatched empty output);
    the interpreter halted in place — completed steps' work is kept, no rollback."""
    step_index: int = 0
    agent: str = ""
    reason: str = ""


@dataclass
class ScaleEvent(AgentEvent):
    """Telemetry only (plan 28 Phase 2, hard problem 4): emitted to events-*.jsonl
    when the harness moves a component off its configured default tier for one
    dispatch. Never emitted on a no-op (chosen_tier == default_tier)."""
    component: str = ""      # e.g. "root-dispatch" | "subagent-dispatch" | "sequencer"
    default_tier: str = ""   # e.g. "supp"
    chosen_tier: str = ""    # e.g. "core"
    reason: str = ""


@dataclass
class ToolScopeEvent(AgentEvent):
    """Telemetry only (plan 31 Phase 1): emitted to events-*.jsonl when
    assignment-time tool scoping (`harness/tool_scope.py`) narrows a unit's
    tool grant below its `ToolPolicy` ceiling for one dispatch. Never
    emitted on a no-op (chosen_rung == "default")."""
    unit: str = ""            # e.g. "root" | "subagent-dispatch"
    chosen_rung: str = ""     # e.g. "read" | "edit" | "fs"
    reason: str = ""


@dataclass
class DirectivePumpEvent(AgentEvent):
    """Telemetry only (plan 28 Phase 3, hard problem 3): emitted when the
    harness pumps domain-craft directives into root's system prompt for one
    dispatch. Never emitted when no domain was detected (empty pump)."""
    domains: list[str] = field(default_factory=list)


@dataclass
class ResponderEvent(AgentEvent):
    """Telemetry only: the responder synthesized (or failed to synthesize,
    falling back to the mechanical recap) the turn's final answer from the
    task graph's actual step outputs, reversing plan 27 decision 15's
    summary-only recap for mutate turns."""
    duration_ms: int = 0
    fell_back: bool = False
