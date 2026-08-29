"""Regression coverage for `GekaiApp._scroll_hint_clicked` (agent/tui/app.py).

Bug: `_scroll_hint_clicked` used to call `self._toggle_esc_pending_clear()`
(and `self._clear_status()`) as a copy/paste leftover from
`action_cancel_stream`'s own ESC-key fallback tail. Since
`_toggle_esc_pending_clear` clears the prompt when `_esc_pending` is already
armed (see its docstring-less body: `if self._esc_pending: prompt.clear();
...`), clicking the "scroll to bottom" hint after a single ESC press would
silently wipe whatever the user had typed. Clicking the hint should only
scroll to the end and refocus the prompt — it must never touch `_esc_pending`
or the prompt text.

Constructing a testable `GekaiApp` follows tests/test_param_hint.py's
scaffold: `GekaiAgent.__init__` is bypassed (`object.__new__` + only the
attributes `start_session()` reads) and `GekaiApp._init_session`/`_stream`
are stubbed so this test only exercises the click-handler wiring, not real
turn execution or session restore.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Static, TextArea

from agent.agent import GekaiAgent
from agent.commands.registry import CommandRegistry
from agent.settings import Permissions
from agent.tui.app import GekaiApp

pytestmark = pytest.mark.asyncio


class _FakeEvents:
    def emit(self, *args: object, **kwargs: object) -> None:
        return None


def _stub_tui_agent(working_dir: Path) -> GekaiAgent:
    stub = object.__new__(GekaiAgent)
    stub.working_dir = working_dir
    stub.permissions = Permissions(read=True, write=True, exec=True)
    stub.debug = False
    stub.model = "fake-model"
    stub.effort = None
    stub._tier_error = None
    stub.events = _FakeEvents()
    return stub


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


def _make_app(tmp_path: Path) -> GekaiApp:
    return GekaiApp(
        agent=_stub_tui_agent(tmp_path),
        registry=CommandRegistry(),
        working_dir=tmp_path,
        version="test",
        branch=None,
    )


# REQ-001: clicking the scroll-hint after ESC has armed "press ESC again to
# clear" must NOT clear the prompt text (regression for the
# `_toggle_esc_pending_clear()` copy/paste bug in `_scroll_hint_clicked`).
async def test_scroll_hint_click_does_not_clear_prompt_after_esc_armed(
    tmp_path: Path,
) -> None:
    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "some in-progress prompt text"
        await pilot.pause()

        # Arm `_esc_pending` exactly like a real ESC press would, via
        # `action_cancel_stream`'s fallback tail (no panels/worker/palette
        # active, so it falls through to `_toggle_esc_pending_clear()`).
        await app.action_cancel_stream()
        assert app._esc_pending is True
        hint = app.query_one("#hint-area", Static)
        assert hint.display is True

        app.query_one("#scroll-hint-wrap").display = True
        await pilot.pause()
        await pilot.click("#scroll-hint")
        await pilot.pause()

        assert prompt.text == "some in-progress prompt text"
        # The ESC-armed state and its hint must also be left untouched —
        # the scroll-hint click has no business touching either.
        assert app._esc_pending is True
        assert hint.display is True
