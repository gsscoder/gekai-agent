"""Coverage for the TUI's slash-alias submit path (agent/tui/app.py's
`_submit_prompt`, around line 1190): submitting `/refactor <instructions>`
must mount the full `/refactor <instructions>` text as the echoed USER
message, not just the bare instruction (bug B from the live-session report —
every other slash path already mounts the full `stripped` string).

Constructing a testable `GekaiApp` follows tests/test_tiers_grid_flow.py's
module docstring exactly: `GekaiAgent.__init__` is bypassed (`object.__new__`
+ only the attributes `start_session()` reads) and `GekaiApp._init_session`
is stubbed to skip the first-run permissions prompt/history replay, neither
of which this test needs. `GekaiApp._stream` itself is stubbed to a no-op
async generator so `run_worker` has nothing real to do — this test only
checks the mounted echo, not turn execution.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.containers import ScrollableContainer
from textual.widgets import TextArea

from agent.agent import GekaiAgent
from agent.commands.registry import CommandRegistry
from agent.events import DoneEvent, SubAgentStartEvent
from agent.harness import turn as harness_turn
from agent.harness.turn import TurnResult
from agent.settings import Permissions
from agent.subagents import NAMESPACE_COLORS
from agent.tui.app import GekaiApp
from agent.tui.widgets import MessageWidget

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _stub_init_session(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_init_session(self: GekaiApp) -> None:
        self._session = self._agent.start_session()

    monkeypatch.setattr(GekaiApp, "_init_session", _fake_init_session)


@pytest.fixture(autouse=True)
def _stub_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_stream(self: GekaiApp, user_input: str, forced_seed: str | None = None) -> None:
        return

    monkeypatch.setattr(GekaiApp, "_stream", _fake_stream)


@pytest.fixture(autouse=True)
def _tiers_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not what this file is testing — the `_submit_prompt` interaction
    guard blocks everything but `/models`/`/tier`/`/exit` while tiers are
    unconfigured, which would otherwise swallow the `/refactor` submits
    below regardless of the real machine's on-disk tier state."""
    monkeypatch.setattr("agent.tui.app.tiers_configured", lambda: True)


def _stub_tui_agent(working_dir: Path) -> GekaiAgent:
    stub = object.__new__(GekaiAgent)
    stub.working_dir = working_dir
    stub.permissions = Permissions(read=True, write=True, exec=True)
    stub.verbose_telemetry = True
    stub.model = "fake-model"
    stub.effort = None
    stub._tier_error = None
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


async def test_slash_alias_prefix_survives_into_mounted_user_message(tmp_path: Path) -> None:
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", TextArea)
        conversation = app.query_one("#conversation", ScrollableContainer)

        prompt.text = "/refactor fix the bug"
        await app._submit_prompt()
        await pilot.pause()

        user_messages = [w for w in _messages(conversation) if w.has_class("user")]
        assert len(user_messages) == 1
        assert user_messages[0].text == "/refactor fix the bug"


async def test_slash_alias_dispatch_renders_the_subagent_badge_not_triaging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: `code-refactorer`'s `namespace="coding"` badge must render
    for a seed-dispatched turn — before the fix, `SubAgentStartEvent`'s
    handler never passed `namespace`/`bg_color` to `SubAgentRenderer.start()`,
    so it always fell through to root's generic "Triaging..." header
    (`agent/tui/app.py`'s `SubAgentStartEvent` branch, distinct from the
    `DelegationStartEvent` branch just below it which already did this
    correctly for task-graph steps)."""

    async def _fake_run_step(agent, session, raw, seed, *, on_event=None, **kwargs):
        if on_event is not None:
            await on_event(SubAgentStartEvent(name="code-refactorer", description="d", color="#000000"))
            await on_event(DoneEvent(thinking_chars=0, files_touched=[]))
        return TurnResult(outcome="ok", answer="done")

    monkeypatch.setattr(harness_turn, "run_step", _fake_run_step)

    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)

        result = await app._run_step(
            "fix the bug", "code-refactorer",
            turn_id="t1", session_id="s1", conversation=conversation, stage=["harness"],
        )
        await pilot.pause()

        assert result.outcome == "ok"
        headers = [w for w in _messages(conversation) if w.has_class("header")]
        assert len(headers) == 1
        markup = headers[0].text
        assert "code-refactorer" in markup
        assert NAMESPACE_COLORS["coding"] in markup
        assert "Triaging" not in markup
