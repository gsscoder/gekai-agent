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
takes the "solo (no estimator wired)" branch straight to the single-agent
path — exactly the pre-plan-27 flat behavior these tests were written
against. The mutate/sequencer path is covered separately in
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
from unittest.mock import AsyncMock

import pytest

from agent.events import AgentEvent, BudgetExhaustedEvent, DiffEvent, DirectivePumpEvent, LogEvent, MaxIterationsEvent, TextChunkEvent
from agent.harness import core as harness_core
from agent.harness.core import Harness
from agent.llm.events import AgentStopped, EventBus
from agent.llm.providers.base import ProviderAdapter
from agent.llm.resolve import ResolvedTier
from agent.llm.tiers import TierName, TierPolicy
from agent.llm.tools import Tool
from agent.llm.types import CompletionResponse, StreamDone, StreamEvent, TextBlock, TextDelta, ToolUseBlock
from agent.pipeline.estimate import ScopeEstimate
from agent.pipeline.plan import Task, TaskGraph
from agent.pipeline.sequencer import Sequencer
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
    instances: ClassVar[list["_ScriptedAdapter"]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._responses = list(type(self).responses)
        self.calls: list[dict[str, Any]] = []
        type(self).instances.append(self)

    async def complete(self, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    async def stream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        yield StreamDone(response=self._responses[index])


class _ChunkedScriptedAdapter(ProviderAdapter):
    """Like `_ScriptedAdapter`, but each call can also replay a list of
    `TextDelta` chunks before its `StreamDone` — mirrors `OpenAIAdapter`'s
    real streaming shape, so a test can exercise the harness's
    `TextChunkEvent` bridging (plan 34 Phase 2) end to end, including a
    direct-dispatch turn that streams text and calls a tool mid-response.

    `chunks[i]` is the list of `TextDelta` chunk strings replayed before the
    `StreamDone` on the i-th call this instance makes; an index with no
    entry (or an empty list) plays no chunks, same as `_ScriptedAdapter`.
    """

    responses: list[CompletionResponse] = []
    chunks: list[list[str]] = []
    instances: ClassVar[list["_ChunkedScriptedAdapter"]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._responses = list(type(self).responses)
        self._chunks = list(type(self).chunks)
        self.calls: list[dict[str, Any]] = []
        type(self).instances.append(self)

    async def complete(self, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError

    async def stream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        for chunk in (self._chunks[index] if index < len(self._chunks) else []):
            yield TextDelta(text=chunk)
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


def _make_harness(estimator: ResolvedTier | None = None) -> Harness:
    """No estimator wired — `Harness.stream(subagent=None)` takes the
    "solo (no estimator)" branch straight to the single-agent path,
    exactly as the pre-plan-27 flat behavior these tests were written
    against (see module docstring).

    Plan 28 Phase 2: `Harness` now takes a resolver closure + a `TierPolicy`
    per scaled touchpoint instead of one frozen `ResolvedTier` each — the
    resolver here always returns the same `tier`, regardless of which
    `TierName` it's asked for, so every touchpoint still resolves to the
    exact same config these tests were written against.
    """
    tier = ResolvedTier(model="test-model", api_key="key", api_base="http://localhost", extra_params={})
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE))
    return Harness(
        resolve=lambda _tier, _touchpoint: tier,
        sequencer_policy=TierPolicy(default=TierName.CORE, allowed=(TierName.SUPP, TierName.CORE)),
        root_dispatch_policy=policy,
        subagent_dispatch_policy=policy,
        estimator=estimator,
    )


def _only(events: list, cls: type) -> list:
    return [e for e in events if isinstance(e, cls)]


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
    harness = _make_harness()
    collected = run(_drain(harness, session, "do something"))

    budget_events = _only(collected, BudgetExhaustedEvent)
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
    harness = _make_harness()
    collected = run(_drain(harness, session, "do something"))

    budget_events = _only(collected, BudgetExhaustedEvent)
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
    harness = _make_harness()
    collected = run(_drain(harness, session, "write hello.py"))

    diff_events = _only(collected, DiffEvent)
    assert len(diff_events) == 1
    assert diff_events[0].path == "hello.py"


def test_no_graph_path_never_passes_tools_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # REQ (plan 31 Phase 3, item 5): the no-graph, single-agent branch of
    # `Harness.stream` has no sequencer/task-graph step to read a `scope`
    # signal from, so it must never narrow root's tool grant -- its
    # `_build_agent(...)` call site stays untouched, always default
    # (`tools_override=None`), root always runs full there.
    responses = [CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")]
    _ScriptedAdapter.responses = responses
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    real_build_agent = harness_core._build_agent
    calls: list[dict[str, Any]] = []

    def _spy_build_agent(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_build_agent(*args, **kwargs)

    monkeypatch.setattr(harness_core, "_build_agent", _spy_build_agent)

    session = _make_session(tmp_path)
    harness = _make_harness()
    run(_drain(harness, session, "sum 10 numbers"))

    assert len(calls) == 1
    assert calls[0].get("tools_override") is None


def test_coding_prompt_emits_directive_pump_event_on_root_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Plan 28 Phase 3: a Python-file-shaped prompt on the no-graph
    (trivial, subagent=None) path pumps the coding domain's escaping
    directives into root and reports it via `DirectivePumpEvent` —
    `test_directive_pump.py` covers the pump function itself; this proves
    the harness wiring at the `Harness.stream` call site."""
    responses = [CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")]
    _ScriptedAdapter.responses = responses
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    harness = _make_harness()
    collected = run(_drain(harness, session, "fix the bug in `src/app/foo.py`"))

    pump_events = _only(collected, DirectivePumpEvent)
    assert len(pump_events) == 1
    assert pump_events[0].domains == ["coding"]


def test_estimator_receives_session_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Plan 33 Phase 1: `Harness.stream` must pass `session.messages` through
    to `Estimator.estimate` as `history` -- without it, a history-dependent
    follow-up ("yes", "do it") misclassifies as `chat`/`solo` and the turn
    silently does nothing (the plan's "hard problem 1")."""
    responses = [CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")]
    _ScriptedAdapter.responses = responses
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    harness = _make_harness(estimator=ResolvedTier(model="supp-model", api_key="k", api_base=None, extra_params={}))
    mock_estimate = AsyncMock(return_value=ScopeEstimate(scope="solo"))
    harness._estimator.estimate = mock_estimate

    session = _make_session(tmp_path)
    session.messages.extend(
        {"role": "user", "content": f"turn {i}"} for i in range(1, 5)
    )
    run(_drain(harness, session, "do it"))

    mock_estimate.assert_awaited_once_with("do it", history=session.messages)


def test_chat_scope_strips_extra_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Plan 33 Phase 2: the Estimator's `chat` rung now drives the
    reasoning-param strip that used to be keyed off `Route.trivial` --
    when the caller passes no explicit `extra_params` override and the
    estimator resolves `scope="chat"`, `Harness.stream` must build the
    no-graph agent with `extra_params={}`, even though the root-dispatch
    tier's own `extra_params` is non-empty."""
    responses = [CompletionResponse(content=[TextBlock(text="hi")], stop_reason="end_turn")]
    _ScriptedAdapter.responses = responses
    _ScriptedAdapter.instances = []
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    tier = ResolvedTier(model="test-model", api_key="key", api_base="http://localhost", extra_params={"reasoning_effort": "high"})
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE))
    harness = Harness(
        resolve=lambda _tier, _touchpoint: tier,
        sequencer_policy=TierPolicy(default=TierName.CORE, allowed=(TierName.SUPP, TierName.CORE)),
        root_dispatch_policy=policy,
        subagent_dispatch_policy=policy,
        estimator=ResolvedTier(model="supp-model", api_key="k", api_base=None, extra_params={}),
    )
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="chat"))

    session = _make_session(tmp_path)
    run(_drain(harness, session, "hi"))

    assert len(_ScriptedAdapter.instances) == 1
    call_kwargs = _ScriptedAdapter.instances[0].calls[0]
    assert "reasoning_effort" not in call_kwargs


def test_solo_scope_keeps_tier_extra_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Companion to test_chat_scope_strips_extra_params: `scope="solo"` must
    still build the no-graph agent with the root-dispatch tier's real
    `extra_params`, unlike `chat`."""
    responses = [CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn")]
    _ScriptedAdapter.responses = responses
    _ScriptedAdapter.instances = []
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    tier = ResolvedTier(model="test-model", api_key="key", api_base="http://localhost", extra_params={"reasoning_effort": "high"})
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE))
    harness = Harness(
        resolve=lambda _tier, _touchpoint: tier,
        sequencer_policy=TierPolicy(default=TierName.CORE, allowed=(TierName.SUPP, TierName.CORE)),
        root_dispatch_policy=policy,
        subagent_dispatch_policy=policy,
        estimator=ResolvedTier(model="supp-model", api_key="k", api_base=None, extra_params={}),
    )
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="solo"))

    session = _make_session(tmp_path)
    run(_drain(harness, session, "read src/app.py"))

    assert len(_ScriptedAdapter.instances) == 1
    call_kwargs = _ScriptedAdapter.instances[0].calls[0]
    assert call_kwargs["reasoning_effort"] == "high"


def test_explicit_extra_params_override_wins_over_chat_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The `extra_params` argument to `stream()` is an explicit override used
    by the subagent path and existing callers -- it must win regardless of
    what the estimator resolves, including `chat`."""
    responses = [CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn")]
    _ScriptedAdapter.responses = responses
    _ScriptedAdapter.instances = []
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    tier = ResolvedTier(model="test-model", api_key="key", api_base="http://localhost", extra_params={"reasoning_effort": "high"})
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE))
    harness = Harness(
        resolve=lambda _tier, _touchpoint: tier,
        sequencer_policy=TierPolicy(default=TierName.CORE, allowed=(TierName.SUPP, TierName.CORE)),
        root_dispatch_policy=policy,
        subagent_dispatch_policy=policy,
        estimator=ResolvedTier(model="supp-model", api_key="k", api_base=None, extra_params={}),
    )
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="chat"))

    session = _make_session(tmp_path)

    async def _drain_with_override() -> list[AgentEvent | str]:
        collected: list[AgentEvent | str] = []
        async for item in harness.stream(session, "hi", extra_params={"foo": "bar"}):
            collected.append(item)
        return collected

    run(_drain_with_override())

    assert len(_ScriptedAdapter.instances) == 1
    call_kwargs = _ScriptedAdapter.instances[0].calls[0]
    assert call_kwargs["foo"] == "bar"


def test_chat_scope_uses_explicit_disable_payload_not_bare_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Plan 34 Phase 3 (Part B): the `chat` rung's stripped extra_params must
    resolve through `resolve_thinking_params(model, enabled=False)`, the same
    explicit-disable mechanism Phase 1 wired everywhere else -- a bare `{}`
    is "unspecified" to a DeepSeek-style model, which then defaults to
    reasoning ON. When root-dispatch resolves to a `thinking_style`-bearing
    model, the built agent's extra_params must carry the disable payload,
    not `{}`."""
    responses = [CompletionResponse(content=[TextBlock(text="hi")], stop_reason="end_turn")]
    _ScriptedAdapter.responses = responses
    _ScriptedAdapter.instances = []
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    tier = ResolvedTier(model="deepseek-v4-flash", api_key="key", api_base="http://localhost", extra_params={"reasoning_effort": "high"})
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE))
    harness = Harness(
        resolve=lambda _tier, _touchpoint: tier,
        sequencer_policy=TierPolicy(default=TierName.CORE, allowed=(TierName.SUPP, TierName.CORE)),
        root_dispatch_policy=policy,
        subagent_dispatch_policy=policy,
        estimator=ResolvedTier(model="supp-model", api_key="k", api_base=None, extra_params={}),
    )
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="chat"))

    session = _make_session(tmp_path)
    run(_drain(harness, session, "hi"))

    assert len(_ScriptedAdapter.instances) == 1
    call_kwargs = _ScriptedAdapter.instances[0].calls[0]
    assert "reasoning_effort" not in call_kwargs
    assert call_kwargs.get("extra_body") == {"thinking": {"type": "disabled"}}


def test_chat_rung_registers_zero_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Plan 34 Phase 3 (Part A): a `chat`-rung dispatch must build root's
    agent with `tools_override=frozenset()` -- no tool schemas at all -- so
    the answer call's system prompt carries no `<tools>` JSON schemas."""
    responses = [CompletionResponse(content=[TextBlock(text="Hi! What can I help you with?")], stop_reason="end_turn")]
    _ScriptedAdapter.responses = responses
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    real_build_agent = harness_core._build_agent
    calls: list[dict[str, Any]] = []

    def _spy_build_agent(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_build_agent(*args, **kwargs)

    monkeypatch.setattr(harness_core, "_build_agent", _spy_build_agent)

    tier = ResolvedTier(model="test-model", api_key="key", api_base="http://localhost", extra_params={})
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE))
    harness = Harness(
        resolve=lambda _tier, _touchpoint: tier,
        sequencer_policy=TierPolicy(default=TierName.CORE, allowed=(TierName.SUPP, TierName.CORE)),
        root_dispatch_policy=policy,
        subagent_dispatch_policy=policy,
        estimator=ResolvedTier(model="supp-model", api_key="k", api_base=None, extra_params={}),
    )
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="chat"))

    session = _make_session(tmp_path)
    run(_drain(harness, session, "hi"))

    assert len(calls) == 1
    assert calls[0].get("tools_override") == frozenset()


def test_solo_rung_still_registers_normal_tool_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Companion to test_chat_rung_registers_zero_tools: `scope="solo"` must
    still dispatch with `tools_override=None` (the full, unfiltered grant),
    unlike `chat`."""
    responses = [CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn")]
    _ScriptedAdapter.responses = responses
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    real_build_agent = harness_core._build_agent
    calls: list[dict[str, Any]] = []

    def _spy_build_agent(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_build_agent(*args, **kwargs)

    monkeypatch.setattr(harness_core, "_build_agent", _spy_build_agent)

    tier = ResolvedTier(model="test-model", api_key="key", api_base="http://localhost", extra_params={})
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE))
    harness = Harness(
        resolve=lambda _tier, _touchpoint: tier,
        sequencer_policy=TierPolicy(default=TierName.CORE, allowed=(TierName.SUPP, TierName.CORE)),
        root_dispatch_policy=policy,
        subagent_dispatch_policy=policy,
        estimator=ResolvedTier(model="supp-model", api_key="k", api_base=None, extra_params={}),
    )
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="solo"))

    session = _make_session(tmp_path)
    run(_drain(harness, session, "read src/app.py"))

    assert len(calls) == 1
    assert calls[0].get("tools_override") is None


def test_mutate_routed_turn_does_not_emit_phantom_directive_pump_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Regression test: before the fix, `Harness.stream` computed the
    domain-directive pump (and yielded `DirectivePumpEvent`) unconditionally,
    before knowing whether the turn takes the no-graph path or routes into
    `_stream_graph()` -- so a mutate-routed turn with a coding-shaped prompt
    emitted a phantom `DirectivePumpEvent` for a `system_base`/dispatch that
    was thrown away and never used. Here the graph's single step delegates to
    a non-"root" subagent (`run_subagent` is monkeypatched to bypass
    `_stream_graph`'s own `dispatch()` closure entirely, which is the only
    place a real pump for the graph path would happen -- and only for
    agent_name == "root" steps), so under the fix no `DirectivePumpEvent`
    should be emitted at all for this turn."""
    harness = _make_harness(estimator=ResolvedTier(model="supp-model", api_key="k", api_base=None, extra_params={}))
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(
        summary="fix the bug",
        steps=[Task(agent="code-expert", instruction="fix it", mission="fix it")],
    )
    monkeypatch.setattr(Sequencer, "sequence", AsyncMock(return_value=graph))

    async def fake_run_subagent(agent, task, **kwargs):
        return f"{agent} done"

    monkeypatch.setattr(harness_core, "run_subagent", fake_run_subagent)

    session = _make_session(tmp_path)
    # Same coding-shaped prompt as the no-graph test above -- under the bug,
    # this prompt alone (independent of the graph's own dispatch) was enough
    # to trigger the phantom early pump/yield.
    collected = run(_drain(harness, session, "fix the bug in `src/app/foo.py`"))

    assert not any(isinstance(e, DirectivePumpEvent) for e in collected)


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
    # the outer agent's own run (`root_run_id`). A stray AgentStopped carrying
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
    harness = _make_harness()
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


def test_no_graph_direct_dispatch_emits_text_chunk_events_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Plan 34 Phase 2: the no-graph (direct root dispatch) path must bridge
    `TextChunkReceived` into `TextChunkEvent`, in delivery order, and the
    final yielded answer string must still be exactly what the assembled
    (non-streamed) response would have produced."""
    chunks = ["Hi", "! ", "What can ", "I help ", "you with?"]
    final_text = "".join(chunks)
    _ChunkedScriptedAdapter.responses = [
        CompletionResponse(content=[TextBlock(text=final_text)], stop_reason="end_turn")
    ]
    _ChunkedScriptedAdapter.chunks = [chunks]
    _ChunkedScriptedAdapter.instances = []
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ChunkedScriptedAdapter)

    session = _make_session(tmp_path)
    harness = _make_harness()
    collected = run(_drain(harness, session, "hi"))

    text_events = _only(collected, TextChunkEvent)
    assert [e.text for e in text_events] == chunks
    assert collected[-1] == final_text


def test_graph_routed_turn_emits_zero_text_chunk_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Hard scope boundary (plan 34 Phase 2): a graph-routed turn shares
    `_bridge_llm_event` with the direct-dispatch path, but must never surface
    a `TextChunkEvent` — even though its dispatched subagent step streams
    `TextDelta` chunks onto the very same bus. Only the direct-dispatch
    caller passes `emit_text_chunks=True`; the graph path's `dispatch()`
    leaves it at the default `False`."""
    harness = _make_harness(estimator=ResolvedTier(model="supp-model", api_key="k", api_base=None, extra_params={}))
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(scope="mutate"))
    graph = TaskGraph(
        summary="fix the bug",
        steps=[Task(agent="code-expert", instruction="fix it", mission="fix it")],
    )
    monkeypatch.setattr(Sequencer, "sequence", AsyncMock(return_value=graph))

    _ChunkedScriptedAdapter.responses = [
        CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    ]
    _ChunkedScriptedAdapter.chunks = [["streamed ", "step ", "text"]]
    _ChunkedScriptedAdapter.instances = []
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ChunkedScriptedAdapter)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "fix the bug in `src/app/foo.py`"))

    assert not any(isinstance(e, TextChunkEvent) for e in collected), (
        f"graph-routed turn leaked a TextChunkEvent: {collected}"
    )


def test_no_graph_direct_dispatch_text_streams_and_tool_call_stay_in_causal_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Hard problem 2 (plan 34): a direct turn that both streams text AND
    calls a tool mid-response must keep `TextChunkEvent`/`LogEvent` in the
    order they actually happened — the harness bridges everything through one
    shared `asyncio.Queue` fed by a single bus subscriber, so causal order
    from the provider is preserved end to end, never scrambled by the
    bridge."""
    _ChunkedScriptedAdapter.responses = [
        _tool_call_response("call-1"),
        CompletionResponse(content=[TextBlock(text="Sure, done.")], stop_reason="end_turn"),
    ]
    _ChunkedScriptedAdapter.chunks = [
        ["thinking about ", "the request"],  # turn 1: streamed text before the tool call
        ["Sure, ", "done."],                 # turn 2: streamed text after the tool result
    ]
    _ChunkedScriptedAdapter.instances = []
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ChunkedScriptedAdapter)

    session = _make_session(tmp_path)
    harness = _make_harness()
    collected = run(_drain(harness, session, "read x.py then tell me"))

    kinds = [type(item).__name__ for item in collected if not isinstance(item, str)]
    text_chunk_indices = [i for i, k in enumerate(kinds) if k == "TextChunkEvent"]
    log_indices = [i for i, k in enumerate(kinds) if k == "LogEvent"]
    assert text_chunk_indices, f"expected TextChunkEvents; got {kinds}"
    assert log_indices, f"expected a LogEvent for the tool call; got {kinds}"
    # Turn 1's stream ("thinking about the request") must fully precede the
    # tool-call LogEvent, and turn 2's stream ("Sure, done.") must fully
    # follow it -- exactly the provider's own causal order, not scrambled by
    # the bridge.
    assert max(text_chunk_indices[:2]) < min(log_indices)
    assert min(text_chunk_indices[2:]) > max(log_indices)

    text_events = _only(collected, TextChunkEvent)
    assert [e.text for e in text_events] == [
        "thinking about ", "the request", "Sure, ", "done.",
    ]
    assert collected[-1] == "Sure, done."
