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
class InferStartEvent(SubAgentEvent):
    """An LLM call is about to start. TUI should show verb+spinner."""
    pass


@dataclass
class InferDeltaEvent(SubAgentEvent):
    """A streamed chunk arrived mid-LLM-call. Carries running completion-token estimate."""
    completion_tokens: int | None = None


@dataclass
class InferEndEvent(SubAgentEvent):
    """An LLM call finished. TUI should show token counts."""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class DoneEvent(SubAgentEvent):
    """Always the last event."""
    pass


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
