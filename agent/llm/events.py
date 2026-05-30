"""Event bus and event dataclasses for agent-run observability."""

from __future__ import annotations

import asyncio
import contextlib
import warnings
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Literal

from .types import (
    CompletionResponse,
    Message,
    ToolResultBlock,
    ToolUseBlock,
    UsageTally,
)

StopReason = Literal["complete", "max_iterations", "cost_ceiling", "error", "cancelled", "timeout"]


@dataclass(frozen=True, slots=True)
class AgentStarted:
    prompt: str | list[Message]
    model: str
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class TurnStarted:
    turn: int
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class ModelRequestSent:
    turn: int
    messages: list[Message]
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class ModelResponseReceived:
    turn: int
    response: CompletionResponse
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class ToolExecutionStarted:
    turn: int
    call: ToolUseBlock
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class ToolExecutionCompleted:
    turn: int
    call: ToolUseBlock
    result: ToolResultBlock
    duration_s: float
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class UsageUpdated:
    turn: int
    usage: UsageTally
    delta: dict[str, int] | None
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class RetryAttemptEvent:
    turn: int
    attempt: int
    delay: float
    exc_type: str
    exc_message: str
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class AgentStopped:
    stop_reason: StopReason
    turns: int
    error: Exception | None = None
    run_id: str = ""


Event = (
    AgentStarted
    | TurnStarted
    | ModelRequestSent
    | ModelResponseReceived
    | ToolExecutionStarted
    | ToolExecutionCompleted
    | UsageUpdated
    | RetryAttemptEvent
    | AgentStopped
)


class _StreamEnd:
    """Sentinel pushed to streams after an AgentStopped to terminate iteration."""


_STREAM_END = _StreamEnd()


@dataclass(slots=True)
class EventBus:
    """Fan-out hub for agent run events."""

    _subscribers: list[Callable[[Event], None]] = field(default_factory=list)
    _streams: list[asyncio.Queue[Event | _StreamEnd]] = field(default_factory=list)

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        self._subscribers.append(callback)

        def _unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._subscribers.remove(callback)

        return _unsubscribe

    def emit(self, event: Event) -> None:
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"EventBus subscriber raised {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        for queue in list(self._streams):
            queue.put_nowait(event)
            if isinstance(event, AgentStopped):
                queue.put_nowait(_STREAM_END)

    def stream(self) -> AsyncIterator[Event]:
        queue: asyncio.Queue[Event | _StreamEnd] = asyncio.Queue()
        self._streams.append(queue)

        async def _iter() -> AsyncIterator[Event]:
            try:
                while True:
                    item = await queue.get()
                    if isinstance(item, _StreamEnd):
                        return
                    yield item
            finally:
                with contextlib.suppress(ValueError):
                    self._streams.remove(queue)

        return _iter()


__all__ = [
    "AgentStarted",
    "AgentStopped",
    "Event",
    "EventBus",
    "ModelRequestSent",
    "ModelResponseReceived",
    "RetryAttemptEvent",
    "StopReason",
    "ToolExecutionCompleted",
    "ToolExecutionStarted",
    "TurnStarted",
    "UsageUpdated",
]
