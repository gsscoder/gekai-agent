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
from typing import Any, ClassVar

import pytest

from agent.events import AgentEvent, BudgetExhaustedEvent, DiffEvent, MaxIterationsEvent
from agent.harness import core as harness_core
from agent.harness.core import Harness
from agent.llm.providers.base import ProviderAdapter
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
