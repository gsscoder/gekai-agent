"""Integration coverage for the `/tiers` grid flow (agent/tui/app.py's
`_open_tiers_panel`/`_render_tiers_panel`/`_handle_tiers_enter`/
`_commit_tiers_edit`/`_cancel_tiers_edit`), driven through a real mounted
`GekaiApp` via Textual's `Pilot`.

This closes the gap `tests/test_tiers_edit.py` and `tests/test_tiers_panel.py`
explicitly leave open (see both files' module docstrings): those cover the
pure helpers and `TiersPanel`'s cursor state machine in isolation, but
nothing exercises the actual wiring — does cycling cells through the real
`GekaiApp` action cascade and committing actually write correct on-disk tier
bindings + keyring credentials? Does cancel truly write nothing? Does an
incomplete commit really leave the panel open with nothing saved? Does the
validate-all-before-saving-any commit fix actually hold atomically?

## Constructing a testable `GekaiApp`

A real `GekaiAgent` (see agent/agent.py) needs a resolvable tier
catalog/bindings/keyring to construct without raising, and `GekaiApp`'s real
`_init_session` (agent/tui/app.py) indexes the whole workspace on mount
(`ws_manager.run("onboard", ...)`, `workspace_db.ensure(...)`) — slow and
irrelevant to `/tiers`. Two shortcuts are taken instead, both scoped to this
file only via `monkeypatch` (nothing here touches app.py/agent.py):

1. `GekaiAgent` is stubbed the same way `tests/test_agent.py::_stub_agent`
   does — `object.__new__(GekaiAgent)`, bypassing `__init__`, setting only
   the attributes actually read by what we exercise (`working_dir`,
   `permissions`, `debug` — all `start_session()` needs).
2. `GekaiApp._init_session` itself is monkeypatched to a trivial stub that
   just creates the session (`#conversation` is already mounted by
   `compose()`, so no extra mounting is needed) — skipping workspace
   indexing, the first-run permissions prompt, and history/timeline replay
   entirely, since none of that is reachable from or relevant to `/tiers`.

## Isolating disk state

`agent.settings.Path.home` is monkeypatched per test (mirrors
tests/test_settings_tiers.py) so `load_model_catalog`/`load_tier_bindings`/
`save_tier_binding` hit a throwaway `tmp_path/.gekai/settings.json`. The
keyring is faked with a plain `dict[str, str]`, patched onto
`agent.tui.app.credentials` (mirrors tests/test_resolve.py's pattern,
applied to the module app.py actually imports its `credentials` reference
from — see agent/tui/app.py's `from agent import credentials`).

## Driving the UI

Navigation (`move_up`/`down`/`left`/`right` across the grid) is driven by
calling `GekaiApp.action_navigate_*` directly — robust, and it still
exercises the real dispatch in app.py (which cell type maps to which panel
call). One deliberate exception: `test_happy_path_commits_all_three_tiers_to_disk`
drives FAST's model->effort column move and the effort cycle itself via real
`pilot.press("right")` / `pilot.press("enter")` keystrokes, proving the
`Binding("left"/"right", ..., priority=True)` wiring actually fires end to
end (this is new wiring, not covered by a keystroke-level regression test
anywhere else in this repo).

Key entry is free-text edit mode (`agent/tui/app.py::_handle_tiers_enter`'s
"key" column branch): Enter on an unselected key cell starts editing (buffer
always starts empty), "p" reads the OS clipboard directly and replaces the
buffer outright (no reliance on "ctrl+v"/the terminal's bracketed-paste
support, and no manual typing — pasting is the only way in; this is the
locked design) — `GekaiApp._read_clipboard_text` is monkeypatched per-call to
return a fixed fake secret — and a second Enter confirms — all driven via
real `pilot.press(...)` keystrokes through the real `on_key`/
`action_confirm_or_submit` dispatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.containers import ScrollableContainer
from textual.pilot import Pilot
from textual.widgets import TextArea

from agent import settings
from agent.agent import GekaiAgent
from agent.commands.registry import CommandRegistry
from agent.llm.tiers import ModelCatalogEntry, TierBinding, TierName
from agent.settings import Permissions
from agent.tui.app import GekaiApp, _mask_key
from agent.tui.widgets import MessageWidget, TiersPanel

pytestmark = pytest.mark.asyncio

MODEL_A = ModelCatalogEntry(
    name="model-a", base_url="https://a.example.com", efforts=("low", "medium", "high"), thinking=False,
)
MODEL_B = ModelCatalogEntry(
    name="model-b", base_url="https://b.example.com", efforts=("low", "medium", "high"), thinking=True,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings.Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _stub_init_session(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real `_init_session` indexes the workspace and drives a first-run
    # permissions prompt — neither is reachable from or relevant to /tiers.
    # `#conversation` is already mounted by `compose()`, so all this needs
    # to do is stand up a session.
    async def _fake_init_session(self: GekaiApp) -> None:
        self._session = self._agent.start_session()

    monkeypatch.setattr(GekaiApp, "_init_session", _fake_init_session)


def _seed_catalog() -> None:
    settings.save_model_catalog_entry(MODEL_A)
    settings.save_model_catalog_entry(MODEL_B)


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
    # Bypasses GekaiAgent.__init__ entirely (same pattern as
    # tests/test_agent.py::_stub_agent) — sets only what start_session()
    # reads, since GekaiApp itself never touches the agent beyond that once
    # `_init_session` is stubbed.
    stub = object.__new__(GekaiAgent)
    stub.working_dir = working_dir
    stub.permissions = Permissions(read=True, write=True, exec=True)
    stub.debug = False
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


def _goto(app: GekaiApp, panel: TiersPanel, row: int, column: str) -> None:
    """Drives the grid cursor to (row, column) via the real
    `GekaiApp.action_navigate_*` methods — exercises the actual app.py
    dispatch (not just TiersPanel's own move_* methods, already covered in
    isolation by tests/test_tiers_panel.py)."""
    while panel.selected_cell[0] < row:
        app.action_navigate_down()
    while panel.selected_cell[0] > row:
        app.action_navigate_up()
    while panel.selected_cell[1] != column:
        columns = ("model", "effort", "thinking", "key") if panel.selected_cell[0] < 3 else ("ok", "cancel")
        idx_cur = columns.index(panel.selected_cell[1])
        idx_target = columns.index(column)
        if idx_target > idx_cur:
            app.action_navigate_right()
        else:
            app.action_navigate_left()


async def _set_key_via_paste(
    app: GekaiApp, pilot: Pilot, monkeypatch: pytest.MonkeyPatch, fake_key: str,
) -> None:
    """Cursor must already be on the tier's "key" cell. Enters edit mode,
    stubs `_read_clipboard_text` to return `fake_key`, pastes it via a real
    "p" keystroke, then confirms with "enter" — exercising the actual
    `on_key`/`_handle_tiers_enter` dispatch, not just calling handlers
    directly."""
    await pilot.press("enter")  # start editing (buffer starts empty)
    await pilot.pause()
    monkeypatch.setattr(GekaiApp, "_read_clipboard_text", staticmethod(lambda: fake_key))
    await pilot.press("p")
    await pilot.pause()
    await pilot.press("enter")  # confirm
    await pilot.pause()


async def _clear_key(app: GekaiApp, pilot: Pilot) -> None:
    """Cursor must already be on the tier's "key" cell. Enters edit mode and
    immediately confirms with an empty buffer — the locked design's explicit
    clear."""
    await pilot.press("enter")
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


async def _configure_all_tiers_fully(
    app: GekaiApp, panel: TiersPanel, pilot: Pilot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FAST -> model-a (key sk-fast), SUPP -> model-b (key sk-supp),
    CORE -> model-b + thinking on (key shared with SUPP's model-b entry, no
    second clipboard read needed). Used by both the happy-path and cancel
    tests."""
    _goto(app, panel, 0, "model")
    await app.action_confirm_or_submit()  # FAST -> model-a
    await pilot.pause()
    _goto(app, panel, 0, "key")
    await _set_key_via_paste(app, pilot, monkeypatch, "sk-fast")

    _goto(app, panel, 1, "model")
    await app.action_confirm_or_submit()  # -> model-a
    await app.action_confirm_or_submit()  # -> model-b
    await pilot.pause()
    _goto(app, panel, 1, "key")
    await _set_key_via_paste(app, pilot, monkeypatch, "sk-supp")

    _goto(app, panel, 2, "model")
    await app.action_confirm_or_submit()  # -> model-a
    await app.action_confirm_or_submit()  # -> model-b
    await pilot.pause()
    _goto(app, panel, 2, "thinking")
    await app.action_confirm_or_submit()  # toggle thinking on (model-b supports it, tier is CORE)
    await pilot.pause()


# ---------------------------------------------------------------------------
# Scenario 1: happy path
# ---------------------------------------------------------------------------


async def test_happy_path_commits_all_three_tiers_to_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog()
    fake_keyring = _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        await app._open_tiers_panel(conversation)
        await pilot.pause()
        panel = app.query_one(TiersPanel)

        _goto(app, panel, 0, "model")
        await app.action_confirm_or_submit()  # FAST -> model-a
        await pilot.pause()

        # Real key-event coverage of the priority=True left/right + enter
        # bindings (Binding("left"/"right", ..., priority=True) in app.py's
        # BINDINGS) — everything else in this file drives via action_* calls.
        assert panel.selected_cell == (0, "model")
        await pilot.press("right")
        await pilot.pause()
        assert panel.selected_cell == (0, "effort")
        await pilot.press("enter")
        await pilot.pause()

        _goto(app, panel, 0, "key")
        await _set_key_via_paste(app, pilot, monkeypatch, "sk-fast")

        _goto(app, panel, 1, "model")
        await app.action_confirm_or_submit()  # -> model-a
        await app.action_confirm_or_submit()  # -> model-b
        await pilot.pause()
        _goto(app, panel, 1, "key")
        await _set_key_via_paste(app, pilot, monkeypatch, "sk-supp")

        _goto(app, panel, 2, "model")
        await app.action_confirm_or_submit()  # -> model-a
        await app.action_confirm_or_submit()  # -> model-b
        await pilot.pause()
        _goto(app, panel, 2, "thinking")
        await app.action_confirm_or_submit()  # CORE + model-b supports thinking -> toggles on
        await pilot.pause()
        # CORE's model is model-b, whose key is already staged from SUPP —
        # no key-cell interaction needed here at all.

        _goto(app, panel, 3, "ok")
        await app.action_confirm_or_submit()
        await pilot.pause()

        bindings = settings.load_tier_bindings()
        assert bindings[TierName.FAST] == TierBinding(model="model-a", default_effort="medium", thinking=False)
        assert bindings[TierName.SUPP] == TierBinding(model="model-b", default_effort="low", thinking=False)
        assert bindings[TierName.CORE] == TierBinding(model="model-b", default_effort="low", thinking=True)
        assert fake_keyring == {"model-a": "sk-fast", "model-b": "sk-supp"}
        assert panel.display is False
        assert app._tiers_edit is None
        assert any(
            w.has_class("command_result") and "FAST: model-a" in w.text for w in _messages(conversation)
        )


# ---------------------------------------------------------------------------
# Scenario 2: shared key across tiers
# ---------------------------------------------------------------------------


async def test_shared_key_across_tiers_needs_only_one_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog()
    _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        await app._open_tiers_panel(conversation)
        await pilot.pause()
        panel = app.query_one(TiersPanel)

        _goto(app, panel, 0, "model")
        await app.action_confirm_or_submit()  # FAST -> model-a
        await pilot.pause()
        _goto(app, panel, 0, "key")
        await _set_key_via_paste(app, pilot, monkeypatch, "sk-shared")

        _goto(app, panel, 1, "model")
        await app.action_confirm_or_submit()  # SUPP -> model-a (same model as FAST)
        await pilot.pause()

        # The re-render triggered purely by the model cycle already shows the
        # shared credential's mask — no key-cell interaction on SUPP's row
        # happened yet at all.
        assert panel._rows[1].key == _mask_key("sk-shared")
        assert panel._rows[1].status == "✓ ready"

        # Starting an edit on SUPP's already-filled key cell and confirming
        # with an empty buffer would clear it — but simply entering and then
        # backing out (Escape) must leave the staged state untouched.
        _goto(app, panel, 1, "key")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        assert app._tiers_edit is not None
        assert app._tiers_edit.key_input == {"model-a": "sk-shared"}
        assert panel.display is True  # Escape only discarded the in-progress edit, not the whole panel


# ---------------------------------------------------------------------------
# Scenario 3: commit blocked when incomplete
# ---------------------------------------------------------------------------


async def test_commit_blocked_when_tiers_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog()
    _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        await app._open_tiers_panel(conversation)
        await pilot.pause()
        panel = app.query_one(TiersPanel)

        # Only FAST configured; SUPP/CORE left unset.
        _goto(app, panel, 0, "model")
        await app.action_confirm_or_submit()
        await pilot.pause()
        _goto(app, panel, 0, "key")
        await _set_key_via_paste(app, pilot, monkeypatch, "sk-fast")

        _goto(app, panel, 3, "ok")
        await app.action_confirm_or_submit()
        await pilot.pause()

        assert settings.load_tier_bindings() == {}
        assert panel.display is True
        assert app._tiers_edit is not None
        assert any(
            w.has_class("warning") and "incomplete" in w.text for w in _messages(conversation)
        )


# ---------------------------------------------------------------------------
# Scenario 4: cancel discards everything
# ---------------------------------------------------------------------------


async def test_cancel_discards_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog()
    fake_keyring = _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        await app._open_tiers_panel(conversation)
        await pilot.pause()
        panel = app.query_one(TiersPanel)

        await _configure_all_tiers_fully(app, panel, pilot, monkeypatch)

        _goto(app, panel, 3, "cancel")
        await app.action_confirm_or_submit()
        await pilot.pause()

        assert settings.load_tier_bindings() == {}
        assert fake_keyring == {}
        assert panel.display is False
        assert app._tiers_edit is None
        assert any(
            w.has_class("command_result") and w.text == "cancelled" for w in _messages(conversation)
        )


# ---------------------------------------------------------------------------
# Scenario 4b: [ok] with nothing changed reports "kept actual tiers"
# ---------------------------------------------------------------------------


async def test_ok_with_no_changes_reports_kept_actual_tiers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog()
    fake_keyring = _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        await app._open_tiers_panel(conversation)
        await pilot.pause()
        panel = app.query_one(TiersPanel)

        await _configure_all_tiers_fully(app, panel, pilot, monkeypatch)
        _goto(app, panel, 3, "ok")
        await app.action_confirm_or_submit()
        await pilot.pause()

        save_calls_before = settings.load_tier_bindings()
        keyring_before = dict(fake_keyring)

        # Re-open /tiers on the exact configuration just saved, touch
        # nothing, and hit [ok] again.
        await app._open_tiers_panel(conversation)
        await pilot.pause()
        panel = app.query_one(TiersPanel)
        _goto(app, panel, 3, "ok")
        await app.action_confirm_or_submit()
        await pilot.pause()

        assert settings.load_tier_bindings() == save_calls_before
        assert fake_keyring == keyring_before
        assert panel.display is False
        assert app._tiers_edit is None
        assert any(
            w.has_class("command_result") and w.text == "kept actual tiers" for w in _messages(conversation)
        )


# ---------------------------------------------------------------------------
# Scenario 5: atomicity of the validate-all-before-saving-any commit fix
# ---------------------------------------------------------------------------


async def test_commit_is_atomic_when_a_later_tier_fails_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the exact bug fixed by hand in `_commit_tiers_edit`:
    previously validation and saving happened tier-by-tier in one loop, so a
    binding that failed validation partway through left earlier tiers' saves
    already on disk. FAST and SUPP are built valid; CORE (the 3rd/last tier
    in TierName's iteration order) is deliberately invalid — thinking=True on
    model-a, which declares thinking=False (agent/llm/tiers.py::validate_binding
    rejects this: "does not support thinking"). `_tiers_edit` is seeded
    directly, bypassing the UI cycling (which never lets you construct an
    invalid state in the first place — the model/effort/thinking cells only
    ever offer legal combinations)."""
    _seed_catalog()
    _patch_credentials(monkeypatch)

    save_calls: list[tuple[TierName, TierBinding]] = []
    monkeypatch.setattr(
        "agent.tui.app.save_tier_binding",
        lambda tier, binding: save_calls.append((tier, binding)),
    )

    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        await app._open_tiers_panel(conversation)
        await pilot.pause()

        edit = app._tiers_edit
        assert edit is not None
        for tier in (TierName.FAST, TierName.SUPP):
            edit.model[tier] = "model-a"
            edit.effort[tier] = "low"
            edit.thinking[tier] = False
        edit.key_input["model-a"] = "sk-shared"
        edit.model[TierName.CORE] = "model-a"
        edit.effort[TierName.CORE] = "low"
        edit.thinking[TierName.CORE] = True  # invalid: model-a.thinking is False

        await app._commit_tiers_edit(conversation)
        await pilot.pause()

        # The regression this guards against: FAST/SUPP were valid and, under
        # the old tier-by-tier loop, would already have been saved by the
        # time CORE's validation raised.
        assert save_calls == []
        assert settings.load_tier_bindings() == {}
        assert app._tiers_edit is not None  # commit aborted, edit state kept (mirrors scenario 3)
        assert any(
            w.has_class("error") and "does not support thinking" in w.text for w in _messages(conversation)
        )


# ---------------------------------------------------------------------------
# Scenario 6: clearing a key is staged until [ok], and can be re-set after
# ---------------------------------------------------------------------------


async def test_clearing_key_is_staged_until_commit_and_can_be_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog()
    fake_keyring = _patch_credentials(monkeypatch)
    fake_keyring["model-a"] = "sk-preexisting"  # simulates a credential from an earlier /tiers session
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        await app._open_tiers_panel(conversation)
        await pilot.pause()
        panel = app.query_one(TiersPanel)

        _goto(app, panel, 0, "model")
        await app.action_confirm_or_submit()  # FAST -> model-a
        await pilot.pause()
        _goto(app, panel, 0, "key")
        assert panel._rows[0].key == _mask_key("sk-preexisting")  # pre-existing keyring credential

        await _clear_key(app, pilot)

        assert panel._rows[0].key == "no key"
        assert panel._rows[0].status == "no key"
        assert app._tiers_edit is not None
        assert app._tiers_edit.key_input == {"model-a": ""}
        assert fake_keyring == {"model-a": "sk-preexisting"}  # nothing deleted yet — staged, not committed

        # [ok] is blocked (FAST now has no key anywhere) — the deletion still
        # must not have happened, since nothing committed.
        _goto(app, panel, 3, "ok")
        await app.action_confirm_or_submit()
        await pilot.pause()
        assert fake_keyring == {"model-a": "sk-preexisting"}
        assert panel.display is True

        # Re-set via edit mode — the fresh value wins over the stale clear.
        _goto(app, panel, 0, "key")
        await _set_key_via_paste(app, pilot, monkeypatch, "sk-new")
        assert panel._rows[0].key == _mask_key("sk-new")

        _goto(app, panel, 1, "model")
        await app.action_confirm_or_submit()  # SUPP -> model-a
        await pilot.pause()
        _goto(app, panel, 2, "model")
        await app.action_confirm_or_submit()  # CORE -> model-a
        await pilot.pause()

        _goto(app, panel, 3, "ok")
        await app.action_confirm_or_submit()
        await pilot.pause()

        assert fake_keyring == {"model-a": "sk-new"}
        assert panel.display is False


# ---------------------------------------------------------------------------
# Scenario 7: the prompt is locked (read-only, no cursor) while a panel is open
# ---------------------------------------------------------------------------


async def test_prompt_is_locked_while_tiers_panel_is_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog()
    _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", TextArea)
        conversation = app.query_one("#conversation", ScrollableContainer)

        assert prompt.read_only is False
        assert prompt.show_cursor is True

        await app._open_tiers_panel(conversation)
        app._sync_prompt_lock()  # normally driven by _poll_prompt_lock's 0.15s tick
        await pilot.pause()

        assert prompt.read_only is True
        assert prompt.show_cursor is False
        assert prompt.has_focus is False

        # A printable keystroke must not land in the prompt while a panel is
        # open — this is the actual regression: `.focus()`/`.insert()` are
        # programmatic calls that ignore `read_only`, so the old auto-focus-
        # and-type fallback in `on_key` would otherwise still work.
        await pilot.press("x")
        await pilot.pause()
        assert prompt.text == ""

        panel = app.query_one(TiersPanel)
        _goto(app, panel, 3, "cancel")
        await app.action_confirm_or_submit()
        app._sync_prompt_lock()
        await pilot.pause()

        assert prompt.read_only is False
        assert prompt.show_cursor is True


# ---------------------------------------------------------------------------
# Scenario 8: navigating away from an in-progress key edit discards the buffer
# ---------------------------------------------------------------------------


async def test_navigating_away_discards_in_progress_key_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog()
    _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        await app._open_tiers_panel(conversation)
        await pilot.pause()
        panel = app.query_one(TiersPanel)

        _goto(app, panel, 0, "model")
        await app.action_confirm_or_submit()  # FAST -> model-a
        await pilot.pause()
        _goto(app, panel, 0, "key")

        await pilot.press("enter")  # start editing
        await pilot.pause()
        monkeypatch.setattr(GekaiApp, "_read_clipboard_text", staticmethod(lambda: "sk-pasted"))
        await pilot.press("p")
        await pilot.pause()
        assert app._tiers_edit is not None
        assert app._tiers_edit.key_editing is True
        assert app._tiers_edit.key_edit_buffer == "sk-pasted"

        _goto(app, panel, 0, "model")  # navigating away — buffer must be discarded, nothing staged
        await pilot.pause()

        assert app._tiers_edit.key_editing is False
        assert app._tiers_edit.key_edit_buffer == ""
        assert app._tiers_edit.key_input == {}
        assert panel._rows[0].key == "no key"


async def test_manual_typing_does_nothing_only_p_pastes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Locked design: pasting via "p" is the *only* way to fill the key
    buffer — typed characters (even a real key typed by hand) must be
    silently ignored while editing."""
    _seed_catalog()
    _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        await app._open_tiers_panel(conversation)
        await pilot.pause()
        panel = app.query_one(TiersPanel)

        _goto(app, panel, 0, "model")
        await app.action_confirm_or_submit()  # FAST -> model-a
        await pilot.pause()
        _goto(app, panel, 0, "key")

        await pilot.press("enter")  # start editing
        await pilot.pause()
        for key in ("s", "k", "-", "1", "2", "3"):
            await pilot.press(key)
            await pilot.pause()

        assert app._tiers_edit is not None
        assert app._tiers_edit.key_editing is True
        assert app._tiers_edit.key_edit_buffer == ""

        monkeypatch.setattr(GekaiApp, "_read_clipboard_text", staticmethod(lambda: "sk-real"))
        await pilot.press("p")
        await pilot.pause()
        assert app._tiers_edit.key_edit_buffer == "sk-real"

        # Backspace discards the pasted buffer back to empty (not "delete
        # one character").
        await pilot.press("backspace")
        await pilot.pause()
        assert app._tiers_edit.key_edit_buffer == ""


async def test_paste_keeps_only_first_line_of_a_multiline_clipboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key is never legitimately multi-line — rather than rejecting the
    paste outright, only the first line is kept (the common case is a
    trailing newline or an accidental whole-file copy, not a real mistake
    worth blocking on)."""
    _seed_catalog()
    _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        await app._open_tiers_panel(conversation)
        await pilot.pause()
        panel = app.query_one(TiersPanel)

        _goto(app, panel, 0, "model")
        await app.action_confirm_or_submit()  # FAST -> model-a
        await pilot.pause()
        _goto(app, panel, 0, "key")

        await pilot.press("enter")  # start editing
        await pilot.pause()
        monkeypatch.setattr(GekaiApp, "_read_clipboard_text", staticmethod(lambda: "sk-real\nsome trailing junk"))
        await pilot.press("p")
        await pilot.pause()

        assert app._tiers_edit is not None
        assert app._tiers_edit.key_edit_buffer == "sk-real"


async def test_paste_of_empty_clipboard_shows_a_hint_and_leaves_buffer_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog()
    _patch_credentials(monkeypatch)
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        await app._open_tiers_panel(conversation)
        await pilot.pause()
        panel = app.query_one(TiersPanel)

        _goto(app, panel, 0, "model")
        await app.action_confirm_or_submit()  # FAST -> model-a
        await pilot.pause()
        _goto(app, panel, 0, "key")

        await pilot.press("enter")  # start editing
        await pilot.pause()
        monkeypatch.setattr(GekaiApp, "_read_clipboard_text", staticmethod(lambda: None))
        await pilot.press("p")
        await pilot.pause()

        assert app._tiers_edit is not None
        assert app._tiers_edit.key_edit_buffer == ""
        assert app._esc_pending is False  # a hint was shown, not the "ESC again" prompt
