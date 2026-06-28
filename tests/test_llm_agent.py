from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent.llm.agent import Agent
from agent.llm.errors import MaxIterationsExceeded
from agent.llm.events import AgentStopped, Event, EventBus
from agent.llm.providers.base import ProviderAdapter
from agent.llm.result import AgentResultEvent
from agent.llm.tools import ToolRegistry, tool
from agent.llm.types import (
    CompletionResponse,
    Message,
    StreamDone,
    StreamEvent,
    TextBlock,
    ToolUseBlock,
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class _ScriptedProvider(ProviderAdapter):
    """Fake provider that replays a scripted sequence of responses."""

    def __init__(self, responses: list[CompletionResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    async def stream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        yield StreamDone(response=self._responses[index])


async def _noop_read_file(path: str) -> str:
    return ""


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool(_noop_read_file, name="read_file"))
    return registry


def _tool_call_response(call_id: str) -> CompletionResponse:
    return CompletionResponse(
        content=[ToolUseBlock(id=call_id, name="read_file", input={"path": "x.py"})],
        stop_reason="tool_use",
    )


def _collect_agent_stopped(bus: EventBus) -> list[AgentStopped]:
    collected: list[AgentStopped] = []

    def _on_event(event: Event) -> None:
        if isinstance(event, AgentStopped):
            collected.append(event)

    bus.subscribe(_on_event)
    return collected


def test_run_loop_normal_completion_has_budget_exhausted_false() -> None:
    responses = [CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")]
    provider = _ScriptedProvider(responses)
    bus = EventBus()
    agent = Agent(
        provider=provider,
        model="test-model",
        tools=_build_registry(),
        max_iterations=3,
        event_bus=bus,
    )
    stopped = _collect_agent_stopped(bus)

    run(agent.run("do the thing"))

    assert len(stopped) == 1
    assert stopped[0].stop_reason == "complete"
    assert stopped[0].budget_exhausted is False


def test_run_loop_salvages_closing_text_when_budget_exhausted() -> None:
    max_iterations = 3
    responses = [_tool_call_response(f"call-{i}") for i in range(1, max_iterations + 1)]
    responses.append(
        CompletionResponse(
            content=[TextBlock(text="done: salvaged")],
            stop_reason="end_turn",
        )
    )
    provider = _ScriptedProvider(responses)
    bus = EventBus()
    agent = Agent(
        provider=provider,
        model="test-model",
        tools=_build_registry(),
        max_iterations=max_iterations,
        event_bus=bus,
    )
    stopped = _collect_agent_stopped(bus)

    messages = run(agent.run("do the thing"))

    assert "done: salvaged" in agent._assistant_text(messages)
    assert len(provider.calls) == max_iterations + 1
    assert provider.calls[-1]["tools"] is None
    assert len(stopped) == 1
    assert stopped[0].stop_reason == "complete"
    assert stopped[0].budget_exhausted is True


def test_run_loop_raises_when_salvage_also_empty() -> None:
    max_iterations = 3
    responses = [_tool_call_response(f"call-{i}") for i in range(1, max_iterations + 1)]
    responses.append(CompletionResponse(content=[], stop_reason="end_turn"))
    provider = _ScriptedProvider(responses)
    bus = EventBus()
    agent = Agent(
        provider=provider,
        model="test-model",
        tools=_build_registry(),
        max_iterations=max_iterations,
        event_bus=bus,
    )
    stopped = _collect_agent_stopped(bus)

    with pytest.raises(MaxIterationsExceeded):
        run(agent.run("do the thing"))

    assert len(stopped) == 1
    assert stopped[0].stop_reason == "max_iterations"
    assert stopped[0].budget_exhausted is True


def test_run_stream_salvages_closing_text_when_budget_exhausted() -> None:
    max_iterations = 3
    responses = [_tool_call_response(f"call-{i}") for i in range(1, max_iterations + 1)]
    responses.append(
        CompletionResponse(
            content=[TextBlock(text="done: salvaged")],
            stop_reason="end_turn",
        )
    )
    provider = _ScriptedProvider(responses)
    bus = EventBus()
    agent = Agent(
        provider=provider,
        model="test-model",
        tools=_build_registry(),
        max_iterations=max_iterations,
        event_bus=bus,
    )
    stopped = _collect_agent_stopped(bus)

    async def _drain() -> None:
        async for _ in agent.run_stream("do the thing"):
            pass

    run(_drain())

    assert len(provider.calls) == max_iterations + 1
    assert provider.calls[-1]["tools"] is None
    assert len(stopped) == 1
    assert stopped[0].stop_reason == "complete"
    assert stopped[0].budget_exhausted is True


def test_run_stream_with_result_salvages_closing_text_when_budget_exhausted() -> None:
    max_iterations = 3
    responses = [_tool_call_response(f"call-{i}") for i in range(1, max_iterations + 1)]
    responses.append(
        CompletionResponse(
            content=[TextBlock(text="done: salvaged")],
            stop_reason="end_turn",
        )
    )
    provider = _ScriptedProvider(responses)
    bus = EventBus()
    agent = Agent(
        provider=provider,
        model="test-model",
        tools=_build_registry(),
        max_iterations=max_iterations,
        event_bus=bus,
    )
    stopped = _collect_agent_stopped(bus)

    async def _drain() -> AgentResultEvent:
        result_event: AgentResultEvent | None = None
        async for event in agent.run_stream_with_result("do the thing"):
            if isinstance(event, AgentResultEvent):
                result_event = event
        assert result_event is not None
        return result_event

    result_event = run(_drain())

    assert result_event.result.stop_reason == "complete"
    assert "done: salvaged" in result_event.result.text
    assert len(provider.calls) == max_iterations + 1
    assert provider.calls[-1]["tools"] is None
    assert len(stopped) == 1
    assert stopped[0].stop_reason == "complete"
    assert stopped[0].budget_exhausted is True


def test_run_loop_nudges_when_early_complete_is_textless() -> None:
    responses = [
        CompletionResponse(content=[], stop_reason="end_turn"),
        CompletionResponse(content=[TextBlock(text="nudged answer")], stop_reason="end_turn"),
    ]
    provider = _ScriptedProvider(responses)
    agent = Agent(
        provider=provider,
        model="test-model",
        tools=_build_registry(),
        max_iterations=3,
    )

    messages = run(agent.run("do the thing"))

    assert len(provider.calls) == 2
    assert provider.calls[1]["tools"] is None
    assert "nudged answer" in agent._assistant_text(messages)


def test_run_stream_nudges_when_early_complete_is_textless() -> None:
    responses = [
        CompletionResponse(content=[], stop_reason="end_turn"),
        CompletionResponse(content=[TextBlock(text="nudged answer")], stop_reason="end_turn"),
    ]
    provider = _ScriptedProvider(responses)
    agent = Agent(
        provider=provider,
        model="test-model",
        tools=_build_registry(),
        max_iterations=3,
    )

    async def _drain() -> None:
        async for _ in agent.run_stream("do the thing"):
            pass

    run(_drain())

    assert len(provider.calls) == 2
    assert provider.calls[1]["tools"] is None


def test_run_stream_with_result_raises_when_salvage_also_empty() -> None:
    max_iterations = 3
    responses = [_tool_call_response(f"call-{i}") for i in range(1, max_iterations + 1)]
    responses.append(CompletionResponse(content=[], stop_reason="end_turn"))
    provider = _ScriptedProvider(responses)
    bus = EventBus()
    agent = Agent(
        provider=provider,
        model="test-model",
        tools=_build_registry(),
        max_iterations=max_iterations,
        event_bus=bus,
    )
    stopped = _collect_agent_stopped(bus)

    async def _drain() -> AgentResultEvent:
        result_event: AgentResultEvent | None = None
        async for event in agent.run_stream_with_result("do the thing"):
            if isinstance(event, AgentResultEvent):
                result_event = event
        assert result_event is not None
        return result_event

    result_event = run(_drain())

    assert result_event.result.stop_reason == "max_iterations"
    assert len(provider.calls) == max_iterations + 1
    assert provider.calls[-1]["tools"] is None
    assert len(stopped) == 1
    assert stopped[0].stop_reason == "max_iterations"
    assert stopped[0].budget_exhausted is True
