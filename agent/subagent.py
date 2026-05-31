from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class SubAgentEvent:
    pass


@dataclass
class SubAgentStartEvent(SubAgentEvent):
    """Always the first event. Carries the subagent name for the TUI header."""
    name: str = ""
    description: str = ""
    color: str = ""


@dataclass
class LogEvent(SubAgentEvent):
    """A timestep worth showing in the conversation (file read, scan step, etc.)."""
    message: str = ""
    tool_name: str = ""


@dataclass
class InferEndEvent(SubAgentEvent):
    """An LLM call finished. TUI should show token counts."""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class StatusUpdateEvent(SubAgentEvent):
    """Progress update. TUI shows a determinate ProgressBar; removed on DoneEvent."""
    progress: int = 0
    total: int | None = None


@dataclass
class ThinkingTokenEvent(SubAgentEvent):
    """A chunk of thinking/reasoning tokens from the LLM."""
    text: str = ""


@dataclass
class DoneEvent(SubAgentEvent):
    """Always the last event."""
    thinking_chars: int = 0


class SubAgent(ABC):
    name: str
    color: str = ""

    @property
    def description(self) -> str:
        return ""

    @abstractmethod
    def run(self) -> AsyncIterator[SubAgentEvent]:
        """
        Async generator of events. Convention:
        - First yield: SubAgentStartEvent(name=self.name)
        - Last yield: DoneEvent()
        """
        ...
