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
    status: Literal["ok", "max_iterations"] = "ok"


@dataclass
class MaxIterationsEvent(AgentEvent):
    """Emitted instead of a text response when the agent hit its iteration limit."""
    pass
