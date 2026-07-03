"""Coverage for `Harness.stream`'s event bridge (`_consume_bus`) in
agent/harness/core.py, specifically the `BudgetExhaustedEvent` branch added
in plan 17b (reserved-final-response fix):

    elif isinstance(event, AgentStopped) and event.budget_exhausted:
        await queue.put(BudgetExhaustedEvent())

Source: agent/harness/core.py:221-222 (the new branch); agent/llm/agent.py's
`Agent._run_loop` (the producer side) sets `AgentStopped.budget_exhausted=True`
whenever the iteration budget is exhausted and a salvage call (tools
suppressed) is attempted — regardless of whether that salvage call finds any
text. This file proves the bridge forwards that signal into Harness.stream's
yielded item sequence; the salvage mechanism itself is already covered by
tests/test_llm_agent.py.

`Harness.stream` hardwires `OpenAIAdapter` inside `_build_agent`
(agent/harness/core.py:118), so `agent.harness.core.OpenAIAdapter` is
monkeypatched with a fake adapter class for the duration of each test —
mirroring tests/test_llm_agent.py's `_ScriptedProvider` pattern but adapted to
`ProviderAdapter`'s interface (agent/llm/providers/base.py).

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
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest

from agent.events import AgentEvent, BudgetExhaustedEvent, DiffEvent, LogEvent, MaxIterationsEvent
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
    site (agent/harness/core.py:118) via a class-level response script.
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
    # REQ: agent/harness/core.py:221-222 — AgentStopped(budget_exhausted=True)
    # is bridged to a BudgetExhaustedEvent on the Harness.stream queue.
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
    # salvage succeeded -> no MaxIterationsEvent, and the salvaged text is
    # yielded as the final string item.
    assert not any(isinstance(e, MaxIterationsEvent) for e in collected)
    assert collected[-1] == "salvaged answer"


def test_budget_exhausted_event_reaches_stream_when_salvage_also_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: per the producer-side contract (agent/llm/agent.py:139-141),
    # budget_exhausted=True is set even when the salvage call itself raises
    # MaxIterationsExceeded (no tool calls and no text in the final response).
    # Confirms the pre-existing MaxIterationsEvent bridge still fires AND
    # that BudgetExhaustedEvent now also fires alongside it.
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


class _InspectableAdapter(_ScriptedAdapter):
    """`_ScriptedAdapter` subclass that records each instance created inside
    `_build_agent`, so the test can inspect `.calls[0]["system"]` afterward —
    `_ScriptedAdapter` itself is left untouched since other tests share it.
    """

    instances: ClassVar[list["_InspectableAdapter"]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        type(self).instances.append(self)


def _ok_response() -> CompletionResponse:
    return CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn")


def test_subagent_mention_injects_subagents_request_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: agent/harness/core.py `Harness.stream` — when subagent is None (direct
    # mode) and the user's raw text explicitly names an invocable subagent,
    # a `<subagents_request>` block naming it is injected into the composed
    # system prompt sent to the provider.
    _InspectableAdapter.instances = []
    _InspectableAdapter.responses = [_ok_response()]
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _InspectableAdapter)

    session = _make_session(tmp_path)
    harness = Harness(model="test-model", api_key="key", api_base="http://localhost")
    run(_drain(harness, session, "use test-fixer to fix the failing tests"))

    adapter = _InspectableAdapter.instances[0]
    system = adapter.calls[0]["system"]
    assert "<subagents_request>" in system
    assert "test-fixer" in system


def test_no_subagent_mention_omits_subagents_request_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # Regression guard: a prompt with no agent mention must not spuriously
    # trigger the injection.
    _InspectableAdapter.instances = []
    _InspectableAdapter.responses = [_ok_response()]
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _InspectableAdapter)

    session = _make_session(tmp_path)
    harness = Harness(model="test-model", api_key="key", api_base="http://localhost")
    run(_drain(harness, session, "list the files in this repo"))

    adapter = _InspectableAdapter.instances[0]
    system = adapter.calls[0]["system"]
    assert "<subagents_request>" not in system


def _make_harness_with_estimator(monkeypatch: pytest.MonkeyPatch) -> Harness:
    """A Harness wired with an Estimator (plan 26 Improvement 1) whose SUPP
    client can be monkeypatched post-construction, mirroring
    tests/test_gate.py's `patch("agent.pipeline.gate.AsyncOpenAI")` pattern
    but applied at the estimate module (constructing `AsyncOpenAI` performs
    no network call, so no patch is needed for construction itself)."""
    return Harness(
        model="test-model", api_key="key", api_base="http://localhost",
        supp_model="supp-model", supp_api_key="k", supp_api_base=None,
    )


def _mock_estimate_response(text: str):
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice])


def test_implementation_estimate_injects_implies_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: plan 26 Improvement 1 — an implementation-sized estimate with
    # named specialists injects the "implies" `<subagents_request>` block,
    # even though the raw user text names no subagent explicitly.
    _InspectableAdapter.instances = []
    _InspectableAdapter.responses = [_ok_response()]
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _InspectableAdapter)

    session = _make_session(tmp_path)
    harness = _make_harness_with_estimator(monkeypatch)
    harness._estimator._client.chat.completions.create = AsyncMock(
        return_value=_mock_estimate_response("IMPLEMENTATION code-expert")
    )
    run(_drain(harness, session, "build a normalization module"))

    adapter = _InspectableAdapter.instances[0]
    system = adapter.calls[0]["system"]
    assert "<subagents_request>" in system
    assert "implies" in system
    assert "code-expert" in system


class _CapturingEventBus(EventBus):
    """`EventBus` subclass that records every instance constructed, so a test
    can grab the exact `bus` object `Harness.stream` builds internally (it's
    otherwise a local variable, not exposed to callers) and emit an extra,
    out-of-band event on it mid-run — simulating a nested `delegate()`-invoked
    subagent completing on the same shared bus (problem 1's scenario).
    """

    captured: ClassVar[list[EventBus]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        type(self).captured.append(self)


def test_nested_agent_stopped_with_different_run_id_does_not_end_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: Harness.stream's _on_event must only terminate the outer consumer
    # loop (queue.put_nowait(None)) on an AgentStopped whose run_id matches
    # the outer agent's own run (`main_run_id`). A stray AgentStopped carrying
    # a *different* run_id — exactly what a nested delegate()-invoked agent
    # emits on the same shared bus when it finishes mid-turn — must be
    # ignored, not treated as "the whole stream is done".
    #
    # The stray event is injected from inside a stubbed tool call (a fake
    # "read_file" tool registered via a patched `make_tools`) on its FIRST
    # invocation only; a second, later invocation of the same tool (turn 2)
    # stays clean. This is essential to the test actually discriminating
    # fixed-vs-broken behavior: `collected[-1] == "final answer"` alone is
    # NOT a valid assertion here — that text comes from `history = await
    # agent_task`, which resolves independently of whether the outer
    # queue-draining loop broke early, so it is present either way. What
    # only survives if the run_id guard works is whether turn 2's LogEvent
    # (queued strictly *after* the stray sentinel, in FIFO order on the same
    # asyncio.Queue) ever gets drained — if `_on_event` treated the stray
    # AgentStopped as terminal, the consumer loop breaks on it and every
    # later queued item, including turn 2's LogEvent, is never yielded.
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

    read_file_logs = [
        item for item in collected
        if isinstance(item, LogEvent) and item.tool_name == "read_file"
    ]
    assert len(read_file_logs) == 2, (
        "expected both the pre-stray (turn 1) and post-stray (turn 2) tool "
        f"calls to surface; got {len(read_file_logs)} — the outer stream "
        f"likely terminated early on the stray AgentStopped: {collected}"
    )
    # The stream must have kept going past the stray event and yielded main's
    # own real completion.
    assert collected[-1] == "final answer"


def test_trivial_estimate_injects_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ: a trivial estimate must not inject any `<subagents_request>` block.
    _InspectableAdapter.instances = []
    _InspectableAdapter.responses = [_ok_response()]
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _InspectableAdapter)

    session = _make_session(tmp_path)
    harness = _make_harness_with_estimator(monkeypatch)
    harness._estimator._client.chat.completions.create = AsyncMock(
        return_value=_mock_estimate_response("TRIVIAL")
    )
    run(_drain(harness, session, "sum 10 random numbers"))

    adapter = _InspectableAdapter.instances[0]
    system = adapter.calls[0]["system"]
    assert "<subagents_request>" not in system
