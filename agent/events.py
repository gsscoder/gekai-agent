from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import ClassVar


@dataclass
class AgentEvent:
    # When set, this event is telemetry: `harness/turn.py` forwards it to
    # events-*.jsonl under this name, with its fields as the payload, instead
    # of each event needing a hand-written emit call. A field opts out of the
    # payload with `metadata={"telemetry": False}`.
    telemetry: ClassVar[str | None] = None

    def telemetry_payload(self) -> dict:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.metadata.get("telemetry", True)
        }


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
    via: str = ""  # "edit_file" | "write_file" | "overwrite" — lets the verifier's gate tell an edit to existing code (or an overwrite via write_file) from a brand-new file write


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
class TextChunkEvent(AgentEvent):
    """A chunk of answer text from the LLM, streamed as it generates (plan 34
    Phase 2). Direct (no-graph) root dispatch only — display-only side
    channel; the persisted answer still comes from the assembled response,
    never from these chunks."""
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
    telemetry: ClassVar[str] = "estimate"
    decision: str = ""  # "chat" | "solo" | "mutate" | "dispatch" | "skipped"
    specialists: list[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class TaskGraphStartedEvent(AgentEvent):
    """Telemetry: the sequencer produced a validated task graph (plan 27 improvement 6)."""
    telemetry: ClassVar[str] = "task_graph"
    step_count: int = 0
    agents: list[str] = field(default_factory=list)
    verify_placements: int = 0
    summary: str = ""
    steps: list[str] = field(default_factory=list)


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
    telemetry: ClassVar[str] = "scale"
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
    telemetry: ClassVar[str] = "tool_scope"
    unit: str = ""            # e.g. "root" | "subagent-dispatch"
    chosen_rung: str = ""     # e.g. "read" | "edit" | "fs"
    reason: str = ""


@dataclass
class DirectivePumpEvent(AgentEvent):
    """Telemetry only (plan 28 Phase 3, hard problem 3): emitted when the
    harness pumps domain-craft directives into root's system prompt for one
    dispatch. Never emitted when no domain was detected (empty pump)."""
    telemetry: ClassVar[str] = "directive_pump"
    domains: list[str] = field(default_factory=list)


@dataclass
class DirectiveAuditEvent(AgentEvent):
    """Telemetry only (plan 35 v3): the directive auditor asked its one
    cheap yes/no question about an ingested or foreign markdown file. The
    verdict itself never enters any model's context (plan 35 concept 5) —
    this event and the TUI notice are its only two consumers."""
    path: str = ""
    has_directives: bool = False
    cached: bool = False
    duration_ms: int = 0
    file_bytes: int = 0


@dataclass
class ForeignFileDetectedEvent(AgentEvent):
    """Plan 35 v3 Phase A: a root-dispatched `read_file` call returned
    markdown text — no prefilter gates this any more (v3 deletes it); every
    root-dispatched `.md` read fires. Carries the file's rel_path and full
    text so a caller can fire the same tracked, cached, swallow-all
    directive audit GEKAI.md gets (`GekaiAgent.start_foreign_file_audit`) —
    never the verdict itself, which stays out of any model's context
    (concept 5)."""
    rel_path: str = ""
    text: str = ""


@dataclass
class VerifyEvent(AgentEvent):
    """Telemetry only: emitted when the coding-step verifier actually ran
    (gate-skipped steps — e.g. pure filesystem scaffolding — emit nothing,
    same never-emitted-on-no-op convention as ScaleEvent/ToolScopeEvent)."""
    telemetry: ClassVar[str] = "verify"
    step_index: int = 0
    agent: str = ""
    ok: bool = True
    violation_count: int = 0
    violations: list[str] = field(default_factory=list, metadata={"telemetry": False})
    chosen_tier: str = ""
    duration_ms: int = 0
    gate: str = "ran"  # "ran" | "skipped_with_mutations" | "repair_no_op" — "skipped_with_mutations" flags a step that mutated files but the coding-diff gate skipped anyway; "repair_no_op" flags a repair dispatch (attempt>=1) that produced no coding diff at all


@dataclass
class ResponderEvent(AgentEvent):
    """Telemetry only: the responder synthesized (or failed to synthesize,
    falling back to the mechanical recap) the turn's final answer from the
    task graph's actual step outputs, reversing plan 27 decision 15's
    summary-only recap for mutate turns."""
    telemetry: ClassVar[str] = "responder"
    duration_ms: int = 0
    fell_back: bool = False
