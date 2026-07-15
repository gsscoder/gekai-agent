"""Coverage for `Harness.stream`'s plan 27 improvement 4 wiring (renamed
plan 28): chit-chat/trivial stay on the single-agent path; mutate and a
`/agent-x` seed both route through the planner + interpreter
(`_stream_graph`) — the "one mutation path" (decision 4).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent.events import DoneEvent, TaskGraphHaltedEvent, TaskGraphStartedEvent
from agent.harness import core as harness_core
from agent.harness.core import Harness
from agent.llm.providers.base import ProviderAdapter
from agent.llm.resolve import ResolvedTier
from agent.llm.types import CompletionResponse, StreamDone, TextBlock
from agent.pipeline.estimate import ScopeEstimate
from agent.pipeline.plan import Task, TaskGraph
from agent.session import Session
from agent.settings import Permissions


def run(coro):
    return asyncio.run(coro)


def _make_session(tmp_path: Path) -> Session:
    return Session(working_dir=tmp_path, permissions=Permissions(read=True, write=True, exec=True))


def _make_harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    main_tier = ResolvedTier(model="test-model", api_key="key", api_base="http://localhost", extra_params={})
    estimator_tier = ResolvedTier(model="supp-model", api_key="k", api_base=None, extra_params={})
    with patch("agent.harness.core.OpenAIAdapter"):
        harness = Harness(
            sequencer=main_tier, main_dispatch=main_tier, subagent_dispatch=main_tier,
            estimator=estimator_tier,
        )
    return harness


async def _drain(harness: Harness, session: Session, prompt: str, **kwargs) -> list:
    collected = []
    async for item in harness.stream(session, prompt, **kwargs):
        collected.append(item)
    return collected


class _ScriptedAdapter(ProviderAdapter):
    """Zero-arg-instantiable fake provider, mirroring tests/test_harness_core.py's
    pattern, so the single-agent path actually emits AgentStopped on the bus
    (a hand-rolled fake Agent that skips the bus never terminates the queue)."""

    responses: list[CompletionResponse] = []

    def __init__(self, *args, **kwargs) -> None:
        self._responses = list(type(self).responses)
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        raise NotImplementedError

    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        yield StreamDone(response=self._responses[len(self.calls) - 1])


def test_trivial_estimate_skips_planner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = _make_harness(monkeypatch)
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(mutate=False))
    harness._planner.plan = AsyncMock(side_effect=AssertionError("planner must not run on trivial"))
    _ScriptedAdapter.responses = [
        CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(harness_core, "OpenAIAdapter", _ScriptedAdapter)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "sum 10 numbers"))

    assert not any(isinstance(e, TaskGraphStartedEvent) for e in collected)
    harness._planner.plan.assert_not_awaited()


def test_mutate_estimate_routes_through_planner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = _make_harness(monkeypatch)
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(mutate=True))
    graph = TaskGraph(
        summary="build a library with tests",
        steps=[
            Task(agent="code-expert", instruction="write the library", mission="write the library"),
            Task(agent="test-expert", instruction="write tests", mission="write tests"),
        ],
    )
    harness._planner.plan = AsyncMock(return_value=graph)

    async def fake_run_subagent(agent, task, **kwargs):
        return f"{agent} done"

    monkeypatch.setattr(harness_core, "run_subagent", fake_run_subagent)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "build a library with tests"))

    started = [e for e in collected if isinstance(e, TaskGraphStartedEvent)]
    assert len(started) == 1
    assert started[0].step_count == 2
    assert started[0].agents == ["code-expert", "test-expert"]
    assert any(isinstance(e, DoneEvent) for e in collected)
    assert collected[-1] == "build a library with tests"


def test_seed_routes_through_planner_even_without_mutate_estimate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    harness = _make_harness(monkeypatch)
    harness._estimator.estimate = AsyncMock(side_effect=AssertionError("estimator must not run when seeded"))
    graph = TaskGraph(summary="add coverage", steps=[Task(agent="test-expert", instruction="add coverage", mission="add coverage")])
    harness._planner.plan = AsyncMock(return_value=graph)

    async def fake_run_subagent(agent, task, **kwargs):
        return "done"

    monkeypatch.setattr(harness_core, "run_subagent", fake_run_subagent)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "add coverage", seed="test-expert"))

    harness._planner.plan.assert_awaited_once()
    _, kwargs = harness._planner.plan.await_args
    assert kwargs.get("seed") == "test-expert"
    assert any(isinstance(e, TaskGraphStartedEvent) for e in collected)


def test_graph_halt_yields_halted_event_and_recap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = _make_harness(monkeypatch)
    harness._estimator.estimate = AsyncMock(return_value=ScopeEstimate(mutate=True))
    graph = TaskGraph(summary="build something", steps=[Task(agent="code-expert", instruction="write it", mission="write it")])
    harness._planner.plan = AsyncMock(return_value=graph)

    async def failing_run_subagent(agent, task, **kwargs):
        return ""  # empty dispatch output -> TaskGraphHalted

    monkeypatch.setattr(harness_core, "run_subagent", failing_run_subagent)

    session = _make_session(tmp_path)
    collected = run(_drain(harness, session, "build something"))

    halted = [e for e in collected if isinstance(e, TaskGraphHaltedEvent)]
    assert len(halted) == 1
    assert halted[0].step_index == 0
    assert "prior steps" in collected[-1] or "HALTED" in collected[-1]
