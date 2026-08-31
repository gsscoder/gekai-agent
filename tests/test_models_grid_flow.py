"""Integration coverage for the `/models` grid flow (agent/tui/app.py's
`_open_models_panel`/`_render_models_panel`/`_handle_models_enter`/
`_commit_models_edit`/`_cancel_models_edit`), driven through a real mounted
`GekaiApp` via Textual's `Pilot`.

This closes the gap `tests/test_models_edit.py` and
`tests/test_models_panel.py` explicitly leave open (see both files' module
docstrings): those cover the pure helpers and `ModelsPanel`'s cursor state
machine in isolation, but nothing exercises the actual wiring — does pasting
a key through the real `GekaiApp` action cascade and committing actually
write the keyring? Does cancel truly write nothing? Is a staged clear really
withheld until commit?

## Constructing a testable `GekaiApp`

A real `GekaiAgent` (see agent/agent.py) needs a resolvable tier
catalog/bindings/keyring to construct without raising, and `GekaiApp`'s real
`_init_session` (agent/tui/app.py) drives a first-run permissions prompt and
history/timeline replay — irrelevant to `/models`. Two shortcuts are taken
instead, both scoped to this file only via `monkeypatch` (nothing here
touches app.py/agent.py):

1. `GekaiAgent` is stubbed the same way `tests/test_agent.py::_stub_agent`
   does — `object.__new__(GekaiAgent)`, bypassing `__init__`, setting only
   the attributes actually read by what we exercise.
2. `GekaiApp._init_session` itself is monkeypatched to a trivial stub that
   just creates the session (`#conversation` is already mounted by
   `compose()`, so no extra mounting is needed).

## Isolating disk state

`agent.settings.Path.home` is monkeypatched per test (mirrors
tests/test_settings_tiers.py). `load_model_catalog` is a live read of
`DEFAULT_MODEL_CATALOG` (never persisted to disk), so `_seed_catalog`
monkeypatches that module global directly to `(MODEL_A, MODEL_B)` instead.
The keyring is faked with a plain `dict[str, str]`, patched onto
`agent.tui.app.credentials` (mirrors tests/test_resolve.py's pattern,
applied to the module app.py actually imports its `credentials` reference
from — see agent/tui/app.py's `from agent import credentials`).

## Driving the UI

Navigation is driven by calling `GekaiApp.action_navigate_*` directly —
robust, and it still exercises the real dispatch in app.py. Key entry is
free-text edit mode (`agent/tui/app.py::_handle_models_enter`'s "key"
branch): Enter on an unselected key cell starts editing (buffer always
starts empty), "p" reads the OS clipboard directly and commits the value
outright (no reliance on "ctrl+v"/the terminal's bracketed-paste support,
and no manual typing — pasting is the only way in; this is the locked
design) — `GekaiApp._read_clipboard_text` is monkeypatched per-call to
return a fixed fake secret — all driven via real `pilot.press(...)`
keystrokes through the real `on_key`/`action_confirm_or_submit` dispatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.containers import ScrollableContainer
from textual.pilot import Pilot
from textual.widgets import TextArea

from agent import credentials, settings
from agent.agent import GekaiAgent
from agent.commands.registry import CommandRegistry
from agent.llm.tiers import ModelCatalogEntry
from agent.settings import Permissions
from agent.tui.app import GekaiApp, _mask_key
from agent.tui.widgets import MessageWidget, ModelsPanel

pytestmark = pytest.mark.asyncio

MODEL_A = ModelCatalogEntry(
    name="model-a", base_url="https://a.example.com", efforts=("low", "medium", "high"), thinking=False,
)
MODEL_B = ModelCatalogEntry(
    name="model-b", base_url="https://b.example.com", efforts=("low", "medium", "high"), thinking=True,
)
KEY_A = credentials.credential_key(MODEL_A.provider, MODEL_A.name)
KEY_B = credentials.credential_key(MODEL_B.provider, MODEL_B.name)
COMMIT_ROW = 2  # two catalog models, so the commit row sits at index 2


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings.Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _stub_init_session(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_init_session(self: GekaiApp) -> None:
        self._session = self._agent.start_session()

    monkeypatch.setattr(GekaiApp, "_init_session", _fake_init_session)


def _seed_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "DEFAULT_MODEL_CATALOG", (MODEL_A, MODEL_B))


def _patch_credentials(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    fake_keyring: dict[str, str] = {}
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: name in fake_keyring)
    monkeypatch.setattr("agent.tui.app.credentials.get_api_key", lambda name: fake_keyring[name])
    monkeypatch.setattr(
        "agent.tui.app.credentials.set_api_key",
        lambda name, key: fake_keyring.__setitem__(name, key),
    )
    monkeypatch.setattr(
        "agent.tui.app.credentials.delete_api_key",
        lambda name: fake_keyring.pop(name, None),
    )
    return fake_keyring


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


async def _open_panel(app: GekaiApp, pilot: Pilot) -> tuple[ScrollableContainer, ModelsPanel]:
    await pilot.pause()
    conversation = app.query_one("#conversation", ScrollableContainer)
    await app._open_models_panel(conversation)
    await pilot.pause()
    return conversation, app.query_one(ModelsPanel)


def _goto(app: GekaiApp, panel: ModelsPanel, row: int, column: str) -> None:
    """Drives the grid cursor to (row, column) via the real
    `GekaiApp.action_navigate_*` methods — exercises the actual app.py
    dispatch (not just ModelsPanel's own move_* methods, already covered in
    isolation by tests/test_models_panel.py)."""
    while panel.selected_cell[0] < row:
        app.action_navigate_down()
    while panel.selected_cell[0] > row:
        app.action_navigate_up()
    while panel.selected_cell[1] != column:
        columns = ("ok", "cancel") if panel.selected_cell[0] == COMMIT_ROW else ("key",)
        if columns.index(column) > columns.index(panel.selected_cell[1]):
            app.action_navigate_right()
        else:
            app.action_navigate_left()


async def _set_key_via_paste(pilot: Pilot, monkeypatch: pytest.MonkeyPatch, fake_key: str) -> None:
    """Cursor must already be on a model's "key" cell. Enters edit mode,
    stubs `_read_clipboard_text` to return `fake_key`, pastes it via a real
    "p" keystroke — commits on the spot, no separate confirm step."""
    await pilot.press("enter")  # start editing (buffer starts empty)
    await pilot.pause()
    monkeypatch.setattr(GekaiApp, "_read_clipboard_text", staticmethod(lambda: fake_key))
    await pilot.press("p")  # pastes and commits immediately
    await pilot.pause()


async def _clear_key(pilot: Pilot) -> None:
    """Cursor must already be on a model's "key" cell. Enters edit mode and
    immediately confirms with an empty buffer — the locked design's explicit
    clear."""
    await pilot.press("enter")
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


# ---------------------------------------------------------------------------
# Scenario 1: happy path
# ---------------------------------------------------------------------------


async def test_happy_path_writes_every_pasted_key_to_the_keyring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    fake_keyring = _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        conversation, panel = await _open_panel(app, pilot)

        # One row per catalog model, provider column filled from the entry.
        assert [(r.provider, r.model) for r in panel._rows] == [("openai", "model-a"), ("openai", "model-b")]
        assert [r.status for r in panel._rows] == ["no key", "no key"]

        _goto(app, panel, 0, "key")
        await _set_key_via_paste(pilot, monkeypatch, "sk-aaaaaaaaaa")
        _goto(app, panel, 1, "key")
        await _set_key_via_paste(pilot, monkeypatch, "sk-bbbbbbbbbb")

        assert fake_keyring == {}  # staged only — nothing written before [ok]

        _goto(app, panel, COMMIT_ROW, "ok")
        await app.action_confirm_or_submit()
        await pilot.pause()

        assert fake_keyring == {KEY_A: "sk-aaaaaaaaaa", KEY_B: "sk-bbbbbbbbbb"}
        assert panel.display is False
        assert app._models_edit is None
        assert any("stored" in w.text for w in _messages(conversation) if w.has_class("command_result"))


# ---------------------------------------------------------------------------
# Scenario 2: keys are scoped to the row they were pasted in
# ---------------------------------------------------------------------------


async def test_key_is_scoped_to_the_row_it_was_pasted_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        conversation, panel = await _open_panel(app, pilot)

        _goto(app, panel, 0, "key")
        await _set_key_via_paste(pilot, monkeypatch, "sk-first-aaaaaaaa")
        _goto(app, panel, 1, "key")
        await _set_key_via_paste(pilot, monkeypatch, "sk-second-bbbbbbb")

        assert app._models_edit is not None
        assert app._models_edit.key_input == {KEY_A: "sk-first-aaaaaaaa", KEY_B: "sk-second-bbbbbbb"}
        assert panel._rows[0].key == _mask_key("sk-first-aaaaaaaa")
        assert panel._rows[1].key == _mask_key("sk-second-bbbbbbb")


# ---------------------------------------------------------------------------
# Scenario 3: cancel
# ---------------------------------------------------------------------------


async def test_cancel_discards_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog(monkeypatch)
    fake_keyring = _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        conversation, panel = await _open_panel(app, pilot)

        _goto(app, panel, 0, "key")
        await _set_key_via_paste(pilot, monkeypatch, "sk-aaaaaaaaaa")

        _goto(app, panel, COMMIT_ROW, "cancel")
        await app.action_confirm_or_submit()
        await pilot.pause()

        assert fake_keyring == {}
        assert panel.display is False
        assert app._models_edit is None
        assert any(w.has_class("command_result") and w.text == "cancelled" for w in _messages(conversation))


async def test_escape_cancels_the_same_way_the_cancel_cell_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    fake_keyring = _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        conversation, panel = await _open_panel(app, pilot)

        _goto(app, panel, 0, "key")
        await _set_key_via_paste(pilot, monkeypatch, "sk-aaaaaaaaaa")

        await app.action_cancel_stream()
        await pilot.pause()

        assert fake_keyring == {}
        assert panel.display is False
        assert app._models_edit is None
        assert any(w.has_class("command_result") and w.text == "cancelled" for w in _messages(conversation))


# ---------------------------------------------------------------------------
# Scenario 4: committing with nothing staged
# ---------------------------------------------------------------------------


async def test_ok_with_no_changes_reports_kept_actual_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    fake_keyring = _patch_credentials(monkeypatch)
    fake_keyring[KEY_A] = "sk-preexisting"
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        conversation, panel = await _open_panel(app, pilot)

        _goto(app, panel, COMMIT_ROW, "ok")
        await app.action_confirm_or_submit()
        await pilot.pause()

        assert fake_keyring == {KEY_A: "sk-preexisting"}
        assert panel.display is False
        assert app._models_edit is None
        assert any(
            w.has_class("command_result") and w.text == "kept actual keys" for w in _messages(conversation)
        )


# ---------------------------------------------------------------------------
# Scenario 5: an explicit clear is staged, not applied immediately
# ---------------------------------------------------------------------------


async def test_clearing_key_is_staged_until_commit_and_can_be_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    fake_keyring = _patch_credentials(monkeypatch)
    fake_keyring[KEY_A] = "sk-preexisting"  # simulates a credential from an earlier /models session
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        conversation, panel = await _open_panel(app, pilot)

        _goto(app, panel, 0, "key")
        assert panel._rows[0].key == _mask_key("sk-preexisting")
        assert panel._rows[0].status == "✓ keyed"

        await _clear_key(pilot)

        assert panel._rows[0].key == "no key"
        assert panel._rows[0].status == "no key"
        assert app._models_edit is not None
        assert app._models_edit.key_input == {KEY_A: ""}
        assert fake_keyring == {KEY_A: "sk-preexisting"}  # nothing deleted yet — staged, not committed

        # Re-set via edit mode — the fresh value wins over the stale clear.
        _goto(app, panel, 0, "key")
        await _set_key_via_paste(pilot, monkeypatch, "sk-newnewnew")
        assert panel._rows[0].key == _mask_key("sk-newnewnew")

        _goto(app, panel, COMMIT_ROW, "ok")
        await app.action_confirm_or_submit()
        await pilot.pause()

        assert fake_keyring == {KEY_A: "sk-newnewnew"}
        assert panel.display is False


async def test_committing_a_clear_actually_deletes_the_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    fake_keyring = _patch_credentials(monkeypatch)
    fake_keyring[KEY_A] = "sk-preexisting"
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        conversation, panel = await _open_panel(app, pilot)

        _goto(app, panel, 0, "key")
        await _clear_key(pilot)
        _goto(app, panel, COMMIT_ROW, "ok")
        await app.action_confirm_or_submit()
        await pilot.pause()

        assert fake_keyring == {}
        assert any("cleared" in w.text for w in _messages(conversation) if w.has_class("command_result"))


# ---------------------------------------------------------------------------
# Scenario 6: the prompt is locked while the panel is open
# ---------------------------------------------------------------------------


async def test_prompt_is_locked_while_models_panel_is_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", TextArea)
        conversation = app.query_one("#conversation", ScrollableContainer)

        assert prompt.read_only is False
        assert prompt.show_cursor is True

        await app._open_models_panel(conversation)
        app._sync_prompt_lock()  # normally driven by _poll_prompt_lock's 0.15s tick
        await pilot.pause()

        assert prompt.read_only is True
        assert prompt.show_cursor is False
        assert prompt.has_focus is False

        # A printable keystroke must not land in the prompt while a panel is
        # open — `.focus()`/`.insert()` are programmatic calls that ignore
        # `read_only`, so the auto-focus-and-type fallback in `on_key` would
        # otherwise still work.
        await pilot.press("x")
        await pilot.pause()
        assert prompt.text == ""


# ---------------------------------------------------------------------------
# Scenario 7: paste semantics
# ---------------------------------------------------------------------------


async def test_paste_commits_immediately_and_survives_navigating_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        conversation, panel = await _open_panel(app, pilot)

        _goto(app, panel, 0, "key")
        await _set_key_via_paste(pilot, monkeypatch, "sk-aaaaaaaaaa")
        app.action_navigate_down()  # away and back
        app.action_navigate_up()

        assert app._models_edit is not None
        assert app._models_edit.key_input == {KEY_A: "sk-aaaaaaaaaa"}
        assert panel._rows[0].key == _mask_key("sk-aaaaaaaaaa")


async def test_navigating_away_before_pasting_discards_the_pending_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        conversation, panel = await _open_panel(app, pilot)

        _goto(app, panel, 0, "key")
        await pilot.press("enter")  # start editing
        await pilot.pause()
        assert app._models_edit is not None
        assert app._models_edit.key_editing is True

        app.action_navigate_down()
        await pilot.pause()

        assert app._models_edit.key_editing is False
        assert app._models_edit.key_input == {}


async def test_manual_typing_does_nothing_only_p_pastes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Locked design: pasting via "p" is the *only* way to fill the key
    buffer — typed characters (even a real key typed by hand) must be
    silently ignored while editing."""
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        conversation, panel = await _open_panel(app, pilot)

        _goto(app, panel, 0, "key")
        await pilot.press("enter")  # start editing
        await pilot.pause()
        for key in ("s", "k", "-", "1", "2", "3"):
            await pilot.press(key)
            await pilot.pause()

        assert app._models_edit is not None
        assert app._models_edit.key_editing is True
        assert app._models_edit.key_edit_buffer == ""

        monkeypatch.setattr(GekaiApp, "_read_clipboard_text", staticmethod(lambda: "sk-real"))
        await pilot.press("p")
        await pilot.pause()
        # paste commits immediately — no lingering buffer to back out of
        assert app._models_edit.key_editing is False
        assert app._models_edit.key_edit_buffer == ""
        assert app._models_edit.key_input == {KEY_A: "sk-real"}


async def test_paste_keeps_only_first_line_of_a_multiline_clipboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key is never legitimately multi-line — rather than rejecting the
    paste outright, only the first line is kept (the common case is a
    trailing newline or an accidental whole-file copy, not a real mistake
    worth blocking on)."""
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        conversation, panel = await _open_panel(app, pilot)

        _goto(app, panel, 0, "key")
        await pilot.press("enter")  # start editing
        await pilot.pause()
        monkeypatch.setattr(GekaiApp, "_read_clipboard_text", staticmethod(lambda: "sk-real\nsome trailing junk"))
        await pilot.press("p")
        await pilot.pause()

        assert app._models_edit is not None
        assert app._models_edit.key_input == {KEY_A: "sk-real"}


async def test_paste_of_empty_clipboard_shows_a_hint_and_leaves_buffer_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        conversation, panel = await _open_panel(app, pilot)

        _goto(app, panel, 0, "key")
        await pilot.press("enter")  # start editing
        await pilot.pause()
        monkeypatch.setattr(GekaiApp, "_read_clipboard_text", staticmethod(lambda: None))
        await pilot.press("p")
        await pilot.pause()

        assert app._models_edit is not None
        assert app._models_edit.key_edit_buffer == ""
        assert app._esc_pending is False  # a hint was shown, not the "ESC again" prompt
