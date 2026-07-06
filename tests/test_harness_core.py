"""Coverage for `Harness.stream`'s event bridge (`_bridge_llm_event`) in
agent/harness/core.py, specifically the `BudgetExhaustedEvent` branch added
in plan 17b (reserved-final-response fix):

    elif isinstance(event, AgentStopped) and event.budget_exhausted:
        await queue.put(BudgetExhaustedEvent())

Source: agent/llm/agent.py's `Agent._run_loop` (the producer side) sets
`AgentStopped.budget_exhausted=True` whenever the iteration budget is
exhausted and a salvage call (tools suppressed) is attempted — regardless of
whether that salvage call finds any text. This file proves the bridge
forwards that signal into Harness.stream's yielded item sequence; the
salvage mechanism itself is already covered by tests/test_llm_agent.py.

`Harness.stream` hardwires `OpenAIAdapter` inside `_build_agent`
(agent/harness/core.py), so `agent.harness.core.OpenAIAdapter` is
monkeypatched with a fake adapter class for the duration of each test —
mirroring tests/test_llm_agent.py's `_ScriptedProvider` pattern but adapted to
`ProviderAdapter`'s interface (agent/llm/providers/base.py).

Plan 27 note: no `supp_model` is passed to `Harness(...)` in this file, so
`self._estimator` is None and every `Harness.stream(subagent=None)` call here
takes the "trivial (no estimator wired)" branch straight to the single-agent
path — exactly the pre-plan-27 flat behavior these tests were written
against. The mutate/planner path is covered separately in
tests/test_harness_stream_plan.py.

ASSUMPTION: `_build_agent`/`Harness.__init__` expose no way to override
`Agent.max_iterations` (default 10, agent/llm/agent.py:53), so the scripted
response list is 11 entries long (10 tool-use turns + 1 salvage turn) to
reach the budget-exhaustion path. This makes the test verbose but avoids
guessing at an unexposed seam.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from agent.events import AgentEvent, BudgetExhaustedEvent, DiffEvent, MaxIterationsEvent
from agent.harness import core as harness_core
from agent.harness.core import Harness
from agent.llm.events import AgentStopped, EventBus
from agent.llm.providers.base import ProviderAdapter
from agent.llm.tools import Tool
from agent.llm.types import CompletionResponse, StreamDone, StreamEvent, TextBlock, ToolUseBlock
from agent.session import Session
from agent.settings import Permissions


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class _ScriptedAdapter(ProviderAdapter):
    """Fake provider adapter that replays a scripted sequence of responses.

    Mirrors tests/test_llm_agent.py's `_ScriptedProvider`, but constructed as
    a zero-arg-instantiable class so it can stand in for `OpenAIAdapter` at
    the `agent.harness.core.OpenAIAdapter(api_key=..., base_url=...)` call
    site via a class-level response script.
    """

    responses: list[CompletionResponse] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._responses = list(type(self).responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    async def stream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        yield StreamDone(response=self._responses[index])


def _tool_call_response(call_id: str) -> CompletionResponse:
    return CompletionResponse(
        content=[ToolUseBlock(id=call_id, name="read_file", input={"path": "x.py"})],
        stop_reason="tool_use",
    )


_MAX_ITERATIONS = 10  # agent/llm/agent.py: Agent.max_iterations default


def _make_session(tmp_path: Path) -> Session:
    return Session(
        working_dir=tmp_path,
        permissions=Permissions(read=True, write=True, exec=True),
    )


async def _drain(harness: Harness, session: Session, prompt: str) -> list[AgentEvent | str]:
    collected: list[AgentEvent | str] = []
    async for item in harness.stream(session, prompt):
        collected.append(item)
    return collected


def test_budget_exhausted_event_reaches_stream_when_salvage_finds_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    responses = [_tool_call_response(f"call-{i}") for i in range(1, _MAX_ITERATIONS + 1)]
    responses.append(
        CompletionResponse(content=[TextBlock(text="salvaged answer")], stop_reason="end_turn")
    )
    _ScriptedAdapter.responses = responses
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    harness = Harness(model="test-model", api_key="key", api_base="http://localhost")
    collected = run(_drain(harness, session, "do something"))

    budget_events = [e for e in collected if isinstance(e, BudgetExhaustedEvent)]
    assert len(budget_events) == 1
    assert not any(isinstance(e, MaxIterationsEvent) for e in collected)
    assert collected[-1] == "salvaged answer"


def test_budget_exhausted_event_reaches_stream_when_salvage_also_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    responses = [_tool_call_response(f"call-{i}") for i in range(1, _MAX_ITERATIONS + 1)]
    responses.append(CompletionResponse(content=[], stop_reason="end_turn"))
    _ScriptedAdapter.responses = responses
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    harness = Harness(model="test-model", api_key="key", api_base="http://localhost")
    collected = run(_drain(harness, session, "do something"))

    budget_events = [e for e in collected if isinstance(e, BudgetExhaustedEvent)]
    assert len(budget_events) == 1
    assert any(isinstance(e, MaxIterationsEvent) for e in collected)


def test_write_file_emits_diff_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    file_content = "print('hello')\n"
    responses = [
        CompletionResponse(
            content=[ToolUseBlock(id="wf-1", name="write_file", input={"path": "hello.py", "content": file_content})],
            stop_reason="tool_use",
        ),
        CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn"),
    ]
    _ScriptedAdapter.responses = responses
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    harness = Harness(model="test-model", api_key="key", api_base="http://localhost")
    collected = run(_drain(harness, session, "write hello.py"))

    diff_events = [e for e in collected if isinstance(e, DiffEvent)]
    assert len(diff_events) == 1
    assert diff_events[0].path == "hello.py"


class _CapturingEventBus(EventBus):
    """`EventBus` subclass that records every instance constructed, so a test
    can grab the exact `bus` object `Harness.stream` builds internally (it's
    otherwise a local variable, not exposed to callers) and emit an extra,
    out-of-band event on it mid-run — simulating a nested subagent run
    completing on the same shared bus (problem 1's scenario).
    """

    captured: ClassVar[list[EventBus]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        type(self).captured.append(self)


def test_nested_agent_stopped_with_different_run_id_does_not_end_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: Harness.stream's bridge must only terminate the outer consumer
    # loop (queue.put_nowait(None)) on an AgentStopped whose run_id matches
    # the outer agent's own run (`main_run_id`). A stray AgentStopped carrying
    # a *different* run_id — exactly what a nested subagent run emits on the
    # same shared bus when it finishes mid-turn — must be ignored, not
    # treated as "the whole stream is done".
    _CapturingEventBus.captured = []
    monkeypatch.setattr(harness_core, "EventBus", _CapturingEventBus)

    call_count = 0

    async def _tool_fn(**kwargs: Any) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            bus = _CapturingEventBus.captured[0]
            bus.emit(AgentStopped(stop_reason="complete", turns=1, run_id="stray-nested-run-id"))
        return "ok"

    stray_tool = Tool(
        name="read_file",
        description="fake tool; first call emits a stray AgentStopped as a side effect",
        input_schema={"type": "object", "properties": {}},
        fn=_tool_fn,
        is_async=True,
        required_permission="none",
    )
    monkeypatch.setattr(harness_core, "make_tools", lambda *a, **kw: [stray_tool])

    responses = [
        _tool_call_response("call-1"),
        _tool_call_response("call-2"),
        CompletionResponse(content=[TextBlock(text="final answer")], stop_reason="end_turn"),
    ]
    _ScriptedAdapter.responses = responses
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    harness = Harness(model="test-model", api_key="key", api_base="http://localhost")
    collected = run(_drain(harness, session, "do something"))

    from agent.events import LogEvent
    read_file_logs = [
        item for item in collected
        if isinstance(item, LogEvent) and item.tool_name == "read_file"
    ]
    assert len(read_file_logs) == 2, (
        "expected both the pre-stray (turn 1) and post-stray (turn 2) tool "
        f"calls to surface; got {len(read_file_logs)} — the outer stream "
        f"likely terminated early on the stray AgentStopped: {collected}"
    )
    assert collected[-1] == "final answer"
