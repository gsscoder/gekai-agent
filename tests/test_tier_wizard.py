"""Integration coverage for the `/tier <TIER>` wizard
(agent/tui/app.py::_run_tier_wizard`), driven through a real mounted
`GekaiApp` via Textual's `Pilot`.

Unlike the old `/tiers` grid, this isn't a persistent panel — it's a chain of
`_ask_choice`/`ChoiceBar` screens (the same primitive the first-run
permissions prompt uses: agent/tui/app.py::_ask_choice`), so most tests here
drive it directly (`app._run_tier_wizard(tier, conversation)` as a
background task, mirroring how tests/test_models_grid_flow.py calls
`_open_models_panel` directly) rather than by typing into the prompt. Two
tests at the bottom cover the `/tier <TIER>` dispatch wiring itself (parsing,
unknown-tier rejection) through real keystrokes, closing the gap the direct
calls leave open.

## Constructing a testable `GekaiApp` / isolating disk state

Same shortcuts as tests/test_models_grid_flow.py — see that file's
docstring for the rationale: a bare `object.__new__(GekaiAgent)` stub,
`_init_session` monkeypatched to skip the first-run flow, `Path.home` and
`DEFAULT_MODEL_CATALOG` monkeypatched for disk isolation, and a plain
`dict`/`set` fake keyring patched onto `agent.tui.app.credentials`.

## Driving a `ChoiceBar` screen

Each step is driven by asserting `ChoiceBar._question`/`._options` (what the
wizard is actually asking), calling `app.action_navigate_right()` until
`ChoiceBar.selected_key` lands on the wanted option, then
`app.action_confirm_or_submit()` to confirm it (or `app.action_cancel_stream()`
for Escape) — the same real dispatch a keystroke would hit, per
`ChoiceBar.move_right`/`GekaiApp.action_confirm_or_submit`'s `_pending_choice`
handling.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.containers import ScrollableContainer
from textual.widgets import TextArea

from agent import credentials, settings
from agent.agent import GekaiAgent
from agent.commands.exit import ExitCommand
from agent.commands.registry import CommandRegistry
from agent.logging import EventLogger
from agent.llm.tiers import ModelCatalogEntry, TierBinding, TierName
from agent.settings import Permissions
from agent.tui.app import GekaiApp
from agent.tui.widgets import ChoiceBar, MessageWidget, ModelsPanel

pytestmark = pytest.mark.asyncio

MODEL_A = ModelCatalogEntry(
    name="model-a", base_url="https://a.example.com", efforts=("low", "medium"), thinking=False,
)
MODEL_B = ModelCatalogEntry(
    name="model-b", base_url="https://b.example.com", efforts=("high", "xhigh"), thinking=True,
)
KEY_A = credentials.credential_key(MODEL_A.provider, MODEL_A.name)
KEY_B = credentials.credential_key(MODEL_B.provider, MODEL_B.name)


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


def _patch_credentials(monkeypatch: pytest.MonkeyPatch, keyed: set[str]) -> None:
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: name in keyed)
    monkeypatch.setattr("agent.tui.app.credentials.get_api_key", lambda name: f"key-for-{name}")


def _stub_tui_agent(working_dir: Path) -> GekaiAgent:
    stub = object.__new__(GekaiAgent)
    stub.working_dir = working_dir
    stub.permissions = Permissions(read=True, write=True, exec=True)
    stub.debug = False
    stub.model = "fake-model"
    stub.effort = None
    stub._tier_error = None
    stub.events = EventLogger()
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


def _pick(app: GekaiApp, key: str) -> None:
    """Cycles the visible `ChoiceBar` to the option keyed `key` via the real
    `action_navigate_right` dispatch, then leaves it selected (not yet
    confirmed) — mirrors arrowing to a choice before pressing Enter."""
    bar = app.query_one(ChoiceBar)
    for _ in range(len(bar._options)):
        if bar.selected_key == key:
            return
        app.action_navigate_right()
    raise AssertionError(f"option {key!r} not found among {bar._options}")


async def _confirm(app: GekaiApp, pilot) -> None:
    await app.action_confirm_or_submit()
    await pilot.pause()


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


async def test_non_core_tier_never_shows_a_thinking_step_even_for_a_thinking_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch, {KEY_A, KEY_B})
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        task = asyncio.create_task(app._run_tier_wizard(TierName.FAST, conversation))
        await pilot.pause()

        bar = app.query_one(ChoiceBar)
        assert "FAST" in bar._question and "model" in bar._question
        _pick(app, "model-b")  # thinking-capable model, but tier is FAST
        await _confirm(app, pilot)

        bar = app.query_one(ChoiceBar)
        assert "effort" in bar._question
        _pick(app, "xhigh")
        await _confirm(app, pilot)

        # No thinking screen — straight to confirm.
        bar = app.query_one(ChoiceBar)
        assert "save?" in bar._question
        assert "thinking=False" in bar._question
        _pick(app, "confirm")
        await _confirm(app, pilot)

        await task
        assert settings.load_tier_bindings()[TierName.FAST] == TierBinding(
            model="model-b", default_effort="xhigh", thinking=False
        )
        assert bar.display is False
        assert any(
            w.has_class("command_result") and "model-b" in w.text for w in _messages(conversation)
        )


async def test_core_plus_thinking_model_shows_the_thinking_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch, {KEY_A, KEY_B})
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        task = asyncio.create_task(app._run_tier_wizard(TierName.CORE, conversation))
        await pilot.pause()

        _pick(app, "model-b")
        await _confirm(app, pilot)
        _pick(app, "high")
        await _confirm(app, pilot)

        bar = app.query_one(ChoiceBar)
        assert "thinking" in bar._question
        _pick(app, "y")
        await _confirm(app, pilot)

        bar = app.query_one(ChoiceBar)
        assert "thinking=True" in bar._question
        _pick(app, "confirm")
        await _confirm(app, pilot)

        await task
        assert settings.load_tier_bindings()[TierName.CORE] == TierBinding(
            model="model-b", default_effort="high", thinking=True
        )


async def test_core_with_a_non_thinking_model_skips_the_thinking_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch, {KEY_A, KEY_B})
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        task = asyncio.create_task(app._run_tier_wizard(TierName.CORE, conversation))
        await pilot.pause()

        _pick(app, "model-a")  # thinking=False
        await _confirm(app, pilot)
        _pick(app, "low")
        await _confirm(app, pilot)

        bar = app.query_one(ChoiceBar)
        assert "save?" in bar._question  # straight to confirm, no thinking screen
        _pick(app, "confirm")
        await _confirm(app, pilot)

        await task
        assert settings.load_tier_bindings()[TierName.CORE].thinking is False


async def test_only_keyed_models_are_offered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch, {KEY_A})  # model-b has no stored key
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        task = asyncio.create_task(app._run_tier_wizard(TierName.SUPP, conversation))
        await pilot.pause()

        bar = app.query_one(ChoiceBar)
        assert [key for key, _ in bar._options] == ["model-a"]

        await app.action_cancel_stream()
        await pilot.pause()
        await task


# ---------------------------------------------------------------------------
# Back navigation
# ---------------------------------------------------------------------------


async def test_back_from_effort_returns_to_model_and_the_new_pick_sticks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch, {KEY_A, KEY_B})
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        task = asyncio.create_task(app._run_tier_wizard(TierName.SUPP, conversation))
        await pilot.pause()

        _pick(app, "model-a")
        await _confirm(app, pilot)
        bar = app.query_one(ChoiceBar)
        assert "model-a" in bar._question

        _pick(app, "__back__")
        await _confirm(app, pilot)
        bar = app.query_one(ChoiceBar)
        assert "pick a model" in bar._question

        _pick(app, "model-b")  # different model this time
        await _confirm(app, pilot)
        bar = app.query_one(ChoiceBar)
        assert "model-b" in bar._question
        _pick(app, "high")
        await _confirm(app, pilot)
        bar = app.query_one(ChoiceBar)
        _pick(app, "confirm")
        await _confirm(app, pilot)

        await task
        assert settings.load_tier_bindings()[TierName.SUPP].model == "model-b"


async def test_back_from_confirm_lands_on_thinking_when_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch, {KEY_A, KEY_B})
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        task = asyncio.create_task(app._run_tier_wizard(TierName.CORE, conversation))
        await pilot.pause()

        _pick(app, "model-b")
        await _confirm(app, pilot)
        _pick(app, "high")
        await _confirm(app, pilot)
        _pick(app, "n")  # thinking off
        await _confirm(app, pilot)

        bar = app.query_one(ChoiceBar)
        assert "thinking=False" in bar._question
        _pick(app, "__back__")
        await _confirm(app, pilot)

        bar = app.query_one(ChoiceBar)
        assert "thinking" in bar._question  # back to the thinking screen, not effort
        _pick(app, "y")
        await _confirm(app, pilot)
        _pick(app, "confirm")
        await _confirm(app, pilot)

        await task
        assert settings.load_tier_bindings()[TierName.CORE].thinking is True


async def test_back_from_confirm_lands_on_effort_when_thinking_is_not_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch, {KEY_A, KEY_B})
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        task = asyncio.create_task(app._run_tier_wizard(TierName.SUPP, conversation))
        await pilot.pause()

        _pick(app, "model-a")
        await _confirm(app, pilot)
        _pick(app, "low")
        await _confirm(app, pilot)

        bar = app.query_one(ChoiceBar)
        assert "save?" in bar._question
        _pick(app, "__back__")
        await _confirm(app, pilot)

        bar = app.query_one(ChoiceBar)
        assert "effort" in bar._question  # SUPP/non-thinking model: back skips the thinking screen
        _pick(app, "medium")
        await _confirm(app, pilot)
        _pick(app, "confirm")
        await _confirm(app, pilot)

        await task
        assert settings.load_tier_bindings()[TierName.SUPP].default_effort == "medium"


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


async def test_cancel_at_confirm_saves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch, {KEY_A})
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        task = asyncio.create_task(app._run_tier_wizard(TierName.FAST, conversation))
        await pilot.pause()

        _pick(app, "model-a")
        await _confirm(app, pilot)
        _pick(app, "low")
        await _confirm(app, pilot)
        _pick(app, "cancel")
        await _confirm(app, pilot)

        await task
        assert settings.load_tier_bindings() == {}
        assert any(w.has_class("command_result") and w.text == "cancelled" for w in _messages(conversation))


async def test_escape_mid_wizard_cancels_the_whole_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch, {KEY_A})
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        task = asyncio.create_task(app._run_tier_wizard(TierName.FAST, conversation))
        await pilot.pause()

        assert app.query_one(ChoiceBar).display is True
        await app.action_cancel_stream()
        await pilot.pause()

        await task
        assert app.query_one(ChoiceBar).display is False
        assert settings.load_tier_bindings() == {}
        assert any(w.has_class("command_result") and w.text == "cancelled" for w in _messages(conversation))


# ---------------------------------------------------------------------------
# No keyed models
# ---------------------------------------------------------------------------


async def test_no_keyed_models_errors_without_ever_opening_a_choice_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch, set())
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        await app._run_tier_wizard(TierName.FAST, conversation)  # returns immediately, no future to drive

        assert app.query_one(ChoiceBar).display is False
        assert settings.load_tier_bindings() == {}
        assert any(
            w.has_class("error") and "/models" in w.text for w in _messages(conversation)
        )


# ---------------------------------------------------------------------------
# `/tier <TIER>` dispatch wiring, through real prompt keystrokes
# ---------------------------------------------------------------------------


async def test_pressing_enter_on_slash_tier_launches_the_wizard_without_deadlocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the real bug: `_run_tier_wizard` blocks on an
    `_ask_choice` future that only resolves from a *later* keypress. Awaited
    inline inside the Enter binding's own action handler (which Textual's
    message pump awaits directly before dispatching the next key), that
    later keypress could never be processed — the whole app hung, cursor
    included, until the process was killed. Fixed by running the wizard as
    a worker (`self.run_worker(...)`, see agent/tui/app.py's `/tier`
    branch) so the pump stays free. Every `pilot.press` below is wrapped in
    a hard timeout so a reintroduced deadlock fails this test instead of
    hanging the run."""
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch, {KEY_A, KEY_B})  # 2 models so "right" has somewhere to go
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "/tier fast"  # case-insensitive tier name
        await asyncio.wait_for(pilot.press("enter"), timeout=5)
        await pilot.pause()

        bar = app.query_one(ChoiceBar)
        assert bar.display is True
        assert "FAST" in bar._question

        # The actual reported symptom: the cursor (arrow keys) must still
        # move — proves the pump is dispatching new key events, not stuck
        # inside the handler that launched the wizard.
        await asyncio.wait_for(pilot.press("right"), timeout=5)
        await pilot.pause()
        assert bar.selected_key != bar._options[0][0]

        await asyncio.wait_for(pilot.press("escape"), timeout=5)
        await pilot.pause()
        assert app.query_one(ChoiceBar).display is False


async def test_tiers_unconfigured_blocks_prompt_except_models_tier_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_submit_prompt`'s interaction guard: while `tiers_configured()` is
    False, plain chat and unrelated commands (`/clear`) must be blocked with
    a warning and never reach `_stream`/command dispatch, while `/models`,
    `/tier`, and `/exit` stay reachable so the user can actually configure
    tiers (or quit)."""
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch, {KEY_A, KEY_B})
    app = _make_app(tmp_path)
    app._command_registry.register(ExitCommand())

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        prompt = app.query_one("#prompt", TextArea)
        assert settings.tiers_configured() is False

        prompt.text = "hello there"
        await asyncio.wait_for(pilot.press("enter"), timeout=5)
        await pilot.pause()
        assert app._worker is None
        assert not any(w.has_class("user") for w in _messages(conversation))
        assert any(
            w.has_class("warning") and "not configured" in w.text for w in _messages(conversation)
        )

        prompt.text = "/clear"
        await asyncio.wait_for(pilot.press("enter"), timeout=5)
        await pilot.pause()
        assert app._worker is None
        assert not any(w.has_class("command_result") for w in _messages(conversation))

        prompt.text = "/tier fast"
        await asyncio.wait_for(pilot.press("enter"), timeout=5)
        await pilot.pause()
        bar = app.query_one(ChoiceBar)
        assert bar.display is True
        assert "FAST" in bar._question
        await asyncio.wait_for(pilot.press("escape"), timeout=5)
        await pilot.pause()

        prompt.text = "/models"
        await asyncio.wait_for(pilot.press("enter"), timeout=5)
        await pilot.pause()
        assert app.query_one(ModelsPanel).display is True
        await app.action_cancel_stream()
        await pilot.pause()
        assert app.query_one(ModelsPanel).display is False

        prompt.text = "/exit"
        await asyncio.wait_for(pilot.press("enter"), timeout=5)
        await pilot.pause()
        assert app._exit_reason == "command"


async def test_completing_the_last_tier_reruns_the_directive_audit_like_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completing the last of the 3 tiers must catch up the GEKAI.md
    read+audit that never actually ran while tiers were incomplete (see
    `GekaiAgent.start_directive_audit`'s no-op-without-a-working-touchpoint
    note) — the same way `/clear` already does. Confirming an *intermediate*
    tier (leaving one still unconfigured) must not trigger this."""
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch, {KEY_A, KEY_B})
    settings.save_tier_binding(TierName.FAST, TierBinding(model="model-a", default_effort="low", thinking=False))
    settings.save_tier_binding(TierName.SUPP, TierBinding(model="model-a", default_effort="low", thinking=False))
    app = _make_app(tmp_path)

    calls = 0

    def _spy(self: GekaiAgent, session, on_verdict=None) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(GekaiAgent, "start_directive_audit", _spy)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert settings.tiers_configured() is False
        conversation = app.query_one("#conversation", ScrollableContainer)
        task = asyncio.create_task(app._run_tier_wizard(TierName.CORE, conversation))
        await pilot.pause()

        _pick(app, "model-a")  # thinking=False, no thinking step
        await _confirm(app, pilot)
        assert calls == 0
        _pick(app, "low")
        await _confirm(app, pilot)
        assert calls == 0

        bar = app.query_one(ChoiceBar)
        assert "save?" in bar._question
        _pick(app, "confirm")
        await _confirm(app, pilot)

        await task
        assert calls == 1
        assert settings.tiers_configured() is True


async def test_typing_slash_tier_with_an_unknown_name_shows_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_catalog(monkeypatch)
    _patch_credentials(monkeypatch, {KEY_A})
    app = _make_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "/tier ultra"
        await app.action_confirm_or_submit()
        await pilot.pause()

        assert app.query_one(ChoiceBar).display is False
        assert any(
            w.has_class("error") and "unknown tier" in w.text for w in _messages(conversation)
        )
