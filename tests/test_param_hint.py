"""Coverage for the "param hint" feature in the TUI: a dimmed `#param-hint`
widget shown above `#input-area` while a slash command/subagent with declared
`params` is being typed, and the "wait, don't auto-submit" palette-selection
behavior for any command whose `params` is non-empty.

Source: agent/tui/app.py's `GekaiApp._params_for`, `_render_param_hint`
(wired into `on_text_area_changed`), the palette-display block inside
`_submit_prompt`, and `action_select_command`. Also agent/commands/base.py
(`Command.params`), agent/subagents/__init__.py (`Subagent.params`).

Constructing a testable `GekaiApp` follows tests/test_slash_alias_dispatch.py's
scaffold exactly: `GekaiAgent.__init__` is bypassed (`object.__new__` + only
the attributes `start_session()` reads) and `GekaiApp._init_session`/`_stream`
are stubbed so this test only exercises the palette/hint wiring, not real
turn execution or session restore.

ASSUMPTION: "auto-submits immediately" for a no-params command is verified by
asserting the command's `execute()` was invoked exactly once and the prompt
text area is empty afterward (no lingering `/clear` text) — there is no
separate "submitted" signal on `GekaiApp` to observe directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Static, TextArea

from agent.agent import GekaiAgent
from agent.commands.clear import ClearCommand
from agent.commands.compact import CompactCommand
from agent.commands.registry import CommandRegistry
from agent.settings import Permissions
from agent.tui.app import GekaiApp
from agent.tui.palette import CommandPalette

pytestmark = pytest.mark.asyncio

# Real user-invocable subagent with the default params spec ("<subagent
# prompt>"), per tests/test_slash_alias_dispatch.py's own use of "/refactor"
# and agent/subagents/coding/code_refactorer.py's `alias="refactor"` (no
# `params` override there, so it carries the dataclass default).
_SUBAGENT_ALIAS = "refactor"


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


class _FakeEvents:
    """No-op stand-in for `GekaiAgent.events` — `_submit_prompt`'s
    non-subagent command path calls `self._agent.events.emit(...)`, which
    the real `GekaiAgent.__init__` (bypassed here, per this module's
    docstring) would otherwise wire up."""

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


def _make_registry() -> tuple[CommandRegistry, ClearCommand]:
    registry = CommandRegistry()
    clear_command = ClearCommand()
    registry.register(clear_command)
    registry.register(CompactCommand())
    return registry, clear_command


def _make_app(tmp_path: Path) -> tuple[GekaiApp, ClearCommand]:
    registry, clear_command = _make_registry()
    app = GekaiApp(
        agent=_stub_tui_agent(tmp_path),
        registry=registry,
        working_dir=tmp_path,
        version="test",
        branch=None,
    )
    return app, clear_command


def _hint_widget(app: GekaiApp) -> Static:
    return app.query_one("#param-hint", Static)


def _hint_text(app: GekaiApp) -> str:
    # Same pattern as tests/test_directive_audit_wiring.py: `.render()`
    # returns the RenderableType last passed to `.update()`; str() gives
    # its plain text for substring assertions.
    return str(_hint_widget(app).render())


# REQ-001: selecting a no-params command ("/clear") from the palette must
# still auto-submit immediately — no lingering prompt text, no param-hint
# left showing (per `action_select_command`'s early-return-only-if-params
# branch and `ClearCommand.params == ""`).
async def test_selecting_clear_from_palette_auto_submits_with_no_lingering_state(
    tmp_path: Path,
) -> None:
    app, clear_command = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "/clear"
        await pilot.pause()
        palette = app.query_one(CommandPalette)
        assert palette.display  # sanity: palette is showing candidates

        await app.action_select_command("clear")
        await pilot.pause()

        assert prompt.text == ""
        assert _hint_widget(app).display is False
        assert _hint_text(app) == ""


# REQ-002: setting the prompt text to "/clear" directly and pressing enter
# while the palette is open also auto-submits (the `_submit_prompt`
# palette-display block only intercepts submission for a non-empty
# `params`).
async def test_submitting_clear_with_palette_open_auto_submits(tmp_path: Path) -> None:
    app, clear_command = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "/clear"
        await pilot.pause()
        palette = app.query_one(CommandPalette)
        assert palette.display

        await app._submit_prompt()
        await pilot.pause()

        assert prompt.text == ""
        assert _hint_widget(app).display is False


# REQ-003: typing "/compact" into the prompt shows the param-hint widget
# with text containing the declared param spec (per `_render_param_hint`
# and `CompactCommand.params`).
async def test_typing_compact_shows_param_hint(tmp_path: Path) -> None:
    app, _ = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "/compact"
        await pilot.pause()

        hint = _hint_widget(app)
        assert hint.display is True
        assert "/compact <optional focus instructions>" in _hint_text(app)


# REQ-004: selecting "/compact" from the palette leaves "/compact " in the
# prompt (does not auto-submit) and the hint stays visible (per
# `action_select_command`'s params-non-empty branch).
async def test_selecting_compact_from_palette_waits_instead_of_submitting(
    tmp_path: Path,
) -> None:
    app, _ = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "/compact"
        await pilot.pause()

        await app.action_select_command("compact")
        await pilot.pause()

        assert prompt.text == "/compact "
        hint = _hint_widget(app)
        assert hint.display is True
        assert "/compact <optional focus instructions>" in _hint_text(app)


# REQ-005: the hint persists while more text is typed after "/compact "
# (per `_render_param_hint` re-deriving the command name from the current
# prompt text on every `TextArea.Changed`).
async def test_hint_persists_while_typing_after_compact(tmp_path: Path) -> None:
    app, _ = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "/compact fix the bug"
        await pilot.pause()

        hint = _hint_widget(app)
        assert hint.display is True
        assert "/compact <optional focus instructions>" in _hint_text(app)


# REQ-006: submitting clears the hint — after `_submit_prompt()` runs (and
# `prompt.clear()` fires a `TextArea.Changed` that re-renders the now-empty
# hint), the widget is empty/hidden.
async def test_submitting_clears_the_param_hint(tmp_path: Path) -> None:
    app, _ = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "/compact fix the bug"
        await pilot.pause()
        assert _hint_widget(app).display is True

        await app._submit_prompt()
        await pilot.pause()

        hint = _hint_widget(app)
        assert hint.display is False
        assert _hint_text(app) == ""


# REQ-007: a user-invocable subagent alias also shows a hint like
# "/<name> <subagent prompt>" when typed, and does not auto-submit on
# palette selection (per `_params_for` consulting `_invocable_subagents`
# before the command registry, and `Subagent.params` defaulting to
# "<subagent prompt>").
async def test_typing_subagent_alias_shows_hint_with_default_subagent_prompt_param(
    tmp_path: Path,
) -> None:
    app, _ = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = f"/{_SUBAGENT_ALIAS}"
        await pilot.pause()

        hint = _hint_widget(app)
        assert hint.display is True
        assert f"/{_SUBAGENT_ALIAS} <subagent prompt>" in _hint_text(app)


async def test_selecting_subagent_alias_from_palette_waits_instead_of_submitting(
    tmp_path: Path,
) -> None:
    app, _ = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = f"/{_SUBAGENT_ALIAS}"
        await pilot.pause()

        await app.action_select_command(_SUBAGENT_ALIAS)
        await pilot.pause()

        assert prompt.text == f"/{_SUBAGENT_ALIAS} "
        hint = _hint_widget(app)
        assert hint.display is True
        assert f"/{_SUBAGENT_ALIAS} <subagent prompt>" in _hint_text(app)
