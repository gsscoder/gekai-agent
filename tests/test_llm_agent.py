from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent.llm.agent import Agent
from agent.llm.errors import MaxIterationsExceeded
from agent.llm.events import AgentStopped, Event, EventBus, TextChunkReceived
from agent.llm.result import AgentResultEvent
from agent.llm.tools import ToolRegistry, tool
from agent.llm.types import (
    CompletionResponse,
    Message,
    StreamDone,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class _ScriptedProvider:
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


def _tool_call_response_with_narration(call_id: str, text: str) -> CompletionResponse:
    return CompletionResponse(
        content=[
            TextBlock(text=text),
            ToolUseBlock(id=call_id, name="read_file", input={"path": "x.py"}),
        ],
        stop_reason="tool_use",
    )


def _collect_agent_stopped(bus: EventBus) -> list[AgentStopped]:
    collected: list[AgentStopped] = []

    def _on_event(event: Event) -> None:
        if isinstance(event, AgentStopped):
            collected.append(event)

    bus.subscribe(_on_event)
    return collected


def _make_agent(
    responses: list[CompletionResponse], max_iterations: int = 3
) -> tuple[Agent, "_ScriptedProvider", list[AgentStopped]]:
    provider = _ScriptedProvider(responses)
    bus = EventBus()
    agent = Agent(
        provider=provider,
        model="test-model",
        tools=_build_registry(),
        max_iterations=max_iterations,
        event_bus=bus,
    )
    return agent, provider, _collect_agent_stopped(bus)


def _assert_stopped_once(stopped: list[AgentStopped], stop_reason: str, budget_exhausted: bool) -> None:
    assert len(stopped) == 1
    assert stopped[0].stop_reason == stop_reason
    assert stopped[0].budget_exhausted is budget_exhausted


def _assert_final_call_dropped_tools(provider: "_ScriptedProvider", max_iterations: int) -> None:
    assert len(provider.calls) == max_iterations + 1
    assert provider.calls[-1]["tools"] is None


def _make_agent_no_bus(responses: list[CompletionResponse]) -> tuple[Agent, "_ScriptedProvider"]:
    provider = _ScriptedProvider(responses)
    agent = Agent(
        provider=provider,
        model="test-model",
        tools=_build_registry(),
        max_iterations=3,
    )
    return agent, provider


def _assert_no_tools_on_call(provider: "_ScriptedProvider", index: int) -> None:
    assert len(provider.calls) == 2
    assert provider.calls[index]["tools"] is None


def test_run_loop_normal_completion_has_budget_exhausted_false() -> None:
    responses = [CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")]
    agent, provider, stopped = _make_agent(responses)

    run(agent.run("do the thing"))

    _assert_stopped_once(stopped, "complete", False)


def test_run_loop_salvages_closing_text_when_budget_exhausted() -> None:
    max_iterations = 3
    responses = [_tool_call_response(f"call-{i}") for i in range(1, max_iterations + 1)]
    responses.append(
        CompletionResponse(
            content=[TextBlock(text="done: salvaged")],
            stop_reason="end_turn",
        )
    )
    agent, provider, stopped = _make_agent(responses, max_iterations)

    messages = run(agent.run("do the thing"))

    assert "done: salvaged" in agent._assistant_text(messages)
    _assert_final_call_dropped_tools(provider, max_iterations)
    _assert_stopped_once(stopped, "complete", True)


def test_run_loop_salvages_when_tool_calls_carry_narration_text() -> None:
    max_iterations = 3
    responses = [
        _tool_call_response_with_narration(f"call-{i}", f"Now let me check {i}")
        for i in range(1, max_iterations + 1)
    ]
    responses.append(
        CompletionResponse(
            content=[TextBlock(text="done: salvaged")],
            stop_reason="end_turn",
        )
    )
    agent, provider, stopped = _make_agent(responses, max_iterations)

    messages = run(agent.run("do the thing"))

    assert "done: salvaged" in agent._assistant_text(messages)
    _assert_final_call_dropped_tools(provider, max_iterations)
    _assert_stopped_once(stopped, "complete", True)


def test_run_loop_raises_when_salvage_also_empty() -> None:
    max_iterations = 3
    responses = [_tool_call_response(f"call-{i}") for i in range(1, max_iterations + 1)]
    responses.append(CompletionResponse(content=[], stop_reason="end_turn"))
    agent, provider, stopped = _make_agent(responses, max_iterations)

    with pytest.raises(MaxIterationsExceeded):
        run(agent.run("do the thing"))

    _assert_stopped_once(stopped, "max_iterations", True)


def test_run_stream_salvages_closing_text_when_budget_exhausted() -> None:
    max_iterations = 3
    responses = [_tool_call_response(f"call-{i}") for i in range(1, max_iterations + 1)]
    responses.append(
        CompletionResponse(
            content=[TextBlock(text="done: salvaged")],
            stop_reason="end_turn",
        )
    )
    agent, provider, stopped = _make_agent(responses, max_iterations)

    async def _drain() -> None:
        async for _ in agent.run_stream("do the thing"):
            pass

    run(_drain())

    _assert_final_call_dropped_tools(provider, max_iterations)
    _assert_stopped_once(stopped, "complete", True)


def test_run_stream_salvages_when_tool_calls_carry_narration_text() -> None:
    max_iterations = 3
    responses = [
        _tool_call_response_with_narration(f"call-{i}", f"Now let me check {i}")
        for i in range(1, max_iterations + 1)
    ]
    responses.append(
        CompletionResponse(
            content=[TextBlock(text="done: salvaged")],
            stop_reason="end_turn",
        )
    )
    agent, provider, stopped = _make_agent(responses, max_iterations)

    async def _drain() -> None:
        async for _ in agent.run_stream("do the thing"):
            pass

    run(_drain())

    _assert_final_call_dropped_tools(provider, max_iterations)
    _assert_stopped_once(stopped, "complete", True)


def test_run_stream_with_result_salvages_closing_text_when_budget_exhausted() -> None:
    max_iterations = 3
    responses = [_tool_call_response(f"call-{i}") for i in range(1, max_iterations + 1)]
    responses.append(
        CompletionResponse(
            content=[TextBlock(text="done: salvaged")],
            stop_reason="end_turn",
        )
    )
    agent, provider, stopped = _make_agent(responses, max_iterations)

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
    _assert_final_call_dropped_tools(provider, max_iterations)
    _assert_stopped_once(stopped, "complete", True)


def test_run_stream_with_result_salvages_when_tool_calls_carry_narration_text() -> None:
    max_iterations = 3
    responses = [
        _tool_call_response_with_narration(f"call-{i}", f"Now let me check {i}")
        for i in range(1, max_iterations + 1)
    ]
    responses.append(
        CompletionResponse(
            content=[TextBlock(text="done: salvaged")],
            stop_reason="end_turn",
        )
    )
    agent, provider, stopped = _make_agent(responses, max_iterations)

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
    _assert_final_call_dropped_tools(provider, max_iterations)
    _assert_stopped_once(stopped, "complete", True)


def test_run_loop_nudges_when_early_complete_is_textless() -> None:
    responses = [
        CompletionResponse(content=[], stop_reason="end_turn"),
        CompletionResponse(content=[TextBlock(text="nudged answer")], stop_reason="end_turn"),
    ]
    agent, provider = _make_agent_no_bus(responses)

    messages = run(agent.run("do the thing"))

    _assert_no_tools_on_call(provider, 1)
    assert "nudged answer" in agent._assistant_text(messages)


def test_run_stream_nudges_when_early_complete_is_textless() -> None:
    responses = [
        CompletionResponse(content=[], stop_reason="end_turn"),
        CompletionResponse(content=[TextBlock(text="nudged answer")], stop_reason="end_turn"),
    ]
    agent, provider = _make_agent_no_bus(responses)

    async def _drain() -> None:
        async for _ in agent.run_stream("do the thing"):
            pass

    run(_drain())

    _assert_no_tools_on_call(provider, 1)


def test_run_stream_with_result_raises_when_salvage_also_empty() -> None:
    max_iterations = 3
    responses = [_tool_call_response(f"call-{i}") for i in range(1, max_iterations + 1)]
    responses.append(CompletionResponse(content=[], stop_reason="end_turn"))
    agent, provider, stopped = _make_agent(responses, max_iterations)

    async def _drain() -> AgentResultEvent:
        result_event: AgentResultEvent | None = None
        async for event in agent.run_stream_with_result("do the thing"):
            if isinstance(event, AgentResultEvent):
                result_event = event
        assert result_event is not None
        return result_event

    result_event = run(_drain())

    assert result_event.result.stop_reason == "max_iterations"
    _assert_final_call_dropped_tools(provider, max_iterations)
    _assert_stopped_once(stopped, "max_iterations", True)


class _ChunkedTextProvider:
    """Fake provider that streams a response as multiple `TextDelta` chunks
    (mirrors `OpenAIAdapter.stream()`'s real shape) before the final
    `StreamDone` — used to prove `_run_loop` forwards `TextDelta` as
    `TextChunkReceived` (plan 34 Phase 2), in the right order, without
    disturbing the assembled final response."""

    def __init__(self, chunks: list[str], final: CompletionResponse) -> None:
        self._chunks = chunks
        self._final = final

    async def complete(self, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    async def stream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        for chunk in self._chunks:
            yield TextDelta(text=chunk)
        yield StreamDone(response=self._final)


def test_run_loop_streams_text_chunks_and_persists_assembled_answer() -> None:
    """The single most important test in plan 34 Phase 2: a scripted
    multi-chunk `TextDelta` response must emit `TextChunkReceived` events in
    delivery order, AND the text that ends up persisted in the returned
    message history (what `Harness.stream` derives its final answer and
    `session.jsonl` write from) must be byte-identical to what a
    non-streamed assembly of the same chunks would produce — streaming is a
    display-only side channel, never the source of the persisted answer."""
    chunks = ["Hi", "! ", "What can ", "I help ", "you with?"]
    final_text = "".join(chunks)
    responses = [CompletionResponse(content=[TextBlock(text=final_text)], stop_reason="end_turn")]
    provider = _ChunkedTextProvider(chunks, responses[0])
    bus = EventBus()
    received: list[TextChunkReceived] = []

    def _on_event(event: Event) -> None:
        if isinstance(event, TextChunkReceived):
            received.append(event)

    bus.subscribe(_on_event)
    agent = Agent(provider=provider, model="test-model", event_bus=bus)

    messages = run(agent.run("hi"))

    assert [e.text for e in received] == chunks

    assistant_text = "".join(
        b.text for b in messages[-1].content if isinstance(b, TextBlock)
    )
    assert assistant_text == final_text
    assert assistant_text == "".join(e.text for e in received)


def _salvage_call(provider: "_ScriptedProvider", max_iterations: int) -> dict:
    """The closing call the agent makes once the iteration budget is spent."""
    assert len(provider.calls) == max_iterations + 1
    return provider.calls[-1]


def _assert_salvage_shaped(call: dict) -> None:
    """The closing call must be able to actually produce an answer: no tools to
    keep chasing, an explicit instruction to write up now (the transcript ends
    on a tool result, so nothing else signals the budget is gone), and a token
    allowance large enough to hold the write-up."""
    assert call["tools"] is None
    last = call["messages"][-1]
    assert last.role == "user"
    assert "tool-call budget" in last.content
    assert call["max_tokens"] >= 8192


def test_run_loop_salvage_call_instructs_and_widens_budget() -> None:
    max_iterations = 3
    responses = [_tool_call_response(f"call-{i}") for i in range(1, max_iterations + 1)]
    responses.append(
        CompletionResponse(content=[TextBlock(text="done: salvaged")], stop_reason="end_turn")
    )
    agent, provider, _ = _make_agent(responses, max_iterations)

    run(agent.run("do the thing"))

    _assert_salvage_shaped(_salvage_call(provider, max_iterations))


def test_run_stream_salvage_call_instructs_and_widens_budget() -> None:
    max_iterations = 3
    responses = [_tool_call_response(f"call-{i}") for i in range(1, max_iterations + 1)]
    responses.append(
        CompletionResponse(content=[TextBlock(text="done: salvaged")], stop_reason="end_turn")
    )
    agent, provider, _ = _make_agent(responses, max_iterations)

    async def _drain() -> None:
        async for _ in agent.run_stream("do the thing"):
            pass

    run(_drain())

    _assert_salvage_shaped(_salvage_call(provider, max_iterations))


def test_run_stream_with_result_salvage_call_instructs_and_widens_budget() -> None:
    max_iterations = 3
    responses = [_tool_call_response(f"call-{i}") for i in range(1, max_iterations + 1)]
    responses.append(
        CompletionResponse(content=[TextBlock(text="done: salvaged")], stop_reason="end_turn")
    )
    agent, provider, _ = _make_agent(responses, max_iterations)

    async def _drain() -> None:
        async for _ in agent.run_stream_with_result("do the thing"):
            pass

    run(_drain())

    _assert_salvage_shaped(_salvage_call(provider, max_iterations))


def test_salvage_instruction_does_not_persist_in_history() -> None:
    """The nudge is scaffolding for the closing call only — it must not end up
    in the returned transcript, where it would read as something the user said."""
    max_iterations = 3
    responses = [_tool_call_response(f"call-{i}") for i in range(1, max_iterations + 1)]
    responses.append(
        CompletionResponse(content=[TextBlock(text="done: salvaged")], stop_reason="end_turn")
    )
    agent, _, _ = _make_agent(responses, max_iterations)

    messages = run(agent.run("do the thing"))

    assert not any(
        isinstance(m.content, str) and "tool-call budget" in m.content for m in messages
    )
