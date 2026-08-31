"""Coverage for the `TurnResult.budget_exhausted` -> TUI rendering wire in
`agent/tui/app.py`'s `GekaiApp._stream`.

`agent/llm/agent.py`'s `max_iterations` handling now salvages a final answer
and sets `AgentStopped.budget_exhausted`, which flows through
`BudgetExhaustedEvent` -> `harness/core.py` -> `harness/turn.py`'s
`TurnResult.budget_exhausted` (already covered by that layer's own tests).
This file covers the last, previously-missing hop: `_stream` mounting a
brief `MessageKind.WARNING` notice next to the salvaged answer so the user
knows the response may be incomplete.

Construction mirrors `tests/test_nested_delegation_rendering.py`: bypass
`GekaiAgent.__init__`, force `tiers_configured()` true, and monkeypatch
`harness_turn.run_step` to return a synthetic `TurnResult` — but unlike that
file, `_stream` itself is NOT stubbed, since it's the code under test here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.containers import ScrollableContainer

from agent.agent import GekaiAgent
from agent.commands.registry import CommandRegistry
from agent.harness import turn as harness_turn
from agent.harness.turn import TurnResult
from agent.settings import Permissions
from agent.tui.app import GekaiApp
from agent.tui.widgets import MessageKind, MessageWidget

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _stub_init_session(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_init_session(self: GekaiApp) -> None:
        self._session = self._agent.start_session()

    monkeypatch.setattr(GekaiApp, "_init_session", _fake_init_session)


@pytest.fixture(autouse=True)
def _tiers_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tui.app.tiers_configured", lambda: True)


def _stub_tui_agent(working_dir: Path) -> GekaiAgent:
    stub = object.__new__(GekaiAgent)
    stub.working_dir = working_dir
    stub.permissions = Permissions(read=True, write=True, exec=True)
    stub.verbose_telemetry = True
    stub.model = "fake-model"
    stub.effort = None
    stub._tier_error = None
    stub.events = SimpleNamespace(emit=lambda *a, **kw: None, new_turn=lambda: "t1")
    return stub


def _make_app(tmp_path: Path) -> GekaiApp:
    return GekaiApp(
        agent=_stub_tui_agent(tmp_path),
        registry=CommandRegistry(),
        working_dir=tmp_path,
        version="test",
        branch=None,
    )


def _messages(conversation: ScrollableContainer) -> list[MessageWidget]:
    return [w for w in conversation.children if isinstance(w, MessageWidget)]


async def test_stream_mounts_warning_when_budget_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fake unconditionally returns `budget_exhausted=True` with a
    # non-empty answer, so the one-shot auto-continue (`_stream` in
    # `agent/tui/app.py`) fires a second `_run_step` call and renders a
    # second assistant+warning pair for it — see
    # `test_stream_auto_continues_once_when_continuation_still_exhausted`
    # below for the call-count cap itself.
    async def _fake_run_step(agent, session, raw, seed, *, on_event=None, **kwargs):
        return TurnResult(outcome="ok", answer="salvaged partial answer", budget_exhausted=True)

    monkeypatch.setattr(harness_turn, "run_step", _fake_run_step)

    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()

        await app._stream("do the thing")
        await pilot.pause()

        conversation = app.query_one("#conversation", ScrollableContainer)
        messages = _messages(conversation)

        assistant_msgs = [w for w in messages if w.has_class("assistant")]
        assert len(assistant_msgs) == 2
        assert all(w.text == "salvaged partial answer" for w in assistant_msgs)

        warning_msgs = [w for w in messages if w.has_class("warning")]
        assert len(warning_msgs) == 2
        assert all("budget" in w.text for w in warning_msgs)


async def test_stream_does_not_mount_warning_when_budget_not_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_run_step(agent, session, raw, seed, *, on_event=None, **kwargs):
        return TurnResult(outcome="ok", answer="a complete answer", budget_exhausted=False)

    monkeypatch.setattr(harness_turn, "run_step", _fake_run_step)

    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()

        await app._stream("do the thing")
        await pilot.pause()

        conversation = app.query_one("#conversation", ScrollableContainer)
        messages = _messages(conversation)

        warning_msgs = [w for w in messages if w.has_class("warning")]
        assert warning_msgs == []


async def test_stream_auto_continues_once_when_budget_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def _fake_run_step(agent, session, raw, seed, *, on_event=None, **kwargs):
        calls.append(raw)
        if len(calls) == 1:
            return TurnResult(outcome="ok", answer="did X, remaining: Y", budget_exhausted=True)
        return TurnResult(outcome="ok", answer="finished Y", budget_exhausted=False)

    monkeypatch.setattr(harness_turn, "run_step", _fake_run_step)

    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()

        await app._stream("do the thing")
        await pilot.pause()

        assert len(calls) == 2
        assert "did X, remaining: Y" in calls[1]

        conversation = app.query_one("#conversation", ScrollableContainer)
        messages = _messages(conversation)

        assistant_msgs = [w for w in messages if w.has_class("assistant")]
        assert len(assistant_msgs) == 2
        assert assistant_msgs[0].text == "did X, remaining: Y"
        assert assistant_msgs[1].text == "finished Y"

        warning_msgs = [w for w in messages if w.has_class("warning")]
        assert len(warning_msgs) == 1


async def test_stream_auto_continues_once_when_continuation_still_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def _fake_run_step(agent, session, raw, seed, *, on_event=None, **kwargs):
        calls.append(raw)
        return TurnResult(outcome="ok", answer="still going", budget_exhausted=True)

    monkeypatch.setattr(harness_turn, "run_step", _fake_run_step)

    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()

        await app._stream("do the thing")
        await pilot.pause()

        assert len(calls) == 2
