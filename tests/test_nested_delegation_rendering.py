"""Coverage for nested-delegation rendering in `agent/tui/app.py`'s
`_run_step` (Phases 1-3 of `plan-nested-delegation-rendering.md`).

A task-graph step's subagent (P) can itself delegate to another subagent (C)
via the `delegate` tool (`agent/tools/delegate.py`), so both P and C emit
`DelegationStartEvent`/`DelegationDoneEvent` on the same event bus that
`_on_event` sees as a flat sequence: `start(P) -> start(C) -> done(C) ->
done(P)`. Before the stack fix, a single nullable `delegation_renderer` slot
made this: (1) leak P's spinner task, (2) misattribute P's post-child events
to the outer/root renderer, (3) never close P's block (no final dot, no
`Done` summary).

Construction follows `tests/test_slash_alias_dispatch.py` exactly: bypass
`GekaiAgent.__init__`, stub `_init_session`/`_stream`, force
`tiers_configured()` true, and drive `_run_step` directly with
`harness_turn.run_step` monkeypatched to replay a synthetic event sequence
through the real `on_event` callback.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.containers import ScrollableContainer
from textual.widgets import Static

import agent.tui.app as tui_app
from agent.agent import GekaiAgent
from agent.commands.registry import CommandRegistry
from agent.events import DelegationDoneEvent, DelegationStartEvent, DoneEvent, LogEvent, StatusUpdateEvent, SubAgentStartEvent
from agent.harness import turn as harness_turn
from agent.harness.turn import TurnResult
from agent.settings import Permissions
from agent.tui.app import GekaiApp, SubAgentRenderer
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
    monkeypatch.setattr("agent.tui.app.tiers_configured", lambda: True)


def _stub_tui_agent(working_dir: Path) -> GekaiAgent:
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


def _track_renderer_instances(monkeypatch: pytest.MonkeyPatch) -> list[SubAgentRenderer]:
    """Records every `SubAgentRenderer` created during a run, in construction
    order — the only way to reach back into `_run_step`'s local closures."""
    created: list[SubAgentRenderer] = []
    orig_init = SubAgentRenderer.__init__

    def _tracking_init(self: SubAgentRenderer, conversation: ScrollableContainer, depth: int = 0) -> None:
        orig_init(self, conversation, depth)
        created.append(self)

    monkeypatch.setattr(SubAgentRenderer, "__init__", _tracking_init)
    return created


def _track_log_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[SubAgentRenderer, str]]:
    """Records which renderer instance handled each `log()` call, so a
    LogEvent's attribution after a child delegation closes can be checked
    without reaching into `_on_event`'s closure state directly."""
    calls: list[tuple[SubAgentRenderer, str]] = []
    orig_log = SubAgentRenderer.log

    async def _tracking_log(self: SubAgentRenderer, message: str, tool_name: str = "") -> None:
        calls.append((self, message))
        await orig_log(self, message, tool_name=tool_name)

    monkeypatch.setattr(SubAgentRenderer, "log", _tracking_log)
    return calls


async def test_nested_delegation_closes_both_blocks_and_attributes_correctly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _track_renderer_instances(monkeypatch)
    log_calls = _track_log_calls(monkeypatch)

    async def _fake_run_step(agent, session, raw, seed, *, on_event=None, **kwargs):
        assert on_event is not None
        await on_event(SubAgentStartEvent(name="root", description="d", color="#000000"))
        await on_event(DelegationStartEvent(agent_name="code-expert", task="parent task"))
        await on_event(DelegationStartEvent(agent_name="code-refactorer", task="child task"))
        await on_event(LogEvent(message="Read foo.py", tool_name="read_file"))
        await on_event(DelegationDoneEvent(agent_name="code-refactorer"))
        await on_event(LogEvent(message="Edit bar.py", tool_name="edit_file"))
        await on_event(DelegationDoneEvent(agent_name="code-expert"))
        await on_event(DoneEvent(thinking_chars=0, files_touched=[]))
        return TurnResult(outcome="ok", answer="done")

    monkeypatch.setattr(harness_turn, "run_step", _fake_run_step)

    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)

        result = await app._run_step(
            "do the thing", None,
            turn_id="t1", session_id="s1", conversation=conversation, stage=["harness"],
        )
        await pilot.pause()

        assert result.outcome == "ok"

        # root, P (code-expert), C (code-refactorer) — three renderers, in
        # construction order.
        assert len(created) == 3
        root_renderer, p_renderer, c_renderer = created
        assert p_renderer._depth == 1
        assert c_renderer._depth == 2

        # Both P and C reached a final Done state — no leaked spinner tasks.
        assert p_renderer._spinner_task is None
        assert c_renderer._spinner_task is None

        # Three header widgets mounted: root's generic header + P's + C's badge.
        headers = [w for w in _messages(conversation) if w.has_class("header")]
        assert len(headers) == 3

        # Two "Done" summary lines — one for each badge-style (P, C) renderer.
        # Only C (depth 2, genuinely nested under P's badge) draws the "⎿"
        # connector; P (depth 1, first-level activation under root) does not.
        done_lines = [
            w for w in conversation.children
            if isinstance(w, Static) and "Done (" in str(w.content)
        ]
        assert len(done_lines) == 2
        connector_done_lines = [w for w in done_lines if "⎿ Done" in str(w.content)]
        assert len(connector_done_lines) == 1

        # The LogEvent that arrived between done(C) and done(P) was
        # attributed to P, not the outer/root renderer.
        post_child_log = next(msg for renderer, msg in log_calls if msg == "Edit bar.py")
        attributed_to = next(renderer for renderer, msg in log_calls if msg == "Edit bar.py")
        assert attributed_to is p_renderer
        assert attributed_to is not root_renderer
        assert post_child_log == "Edit bar.py"


async def test_single_level_delegation_matches_current_main_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A depth-1-only sequence (single delegation, no nested delegation)
    produces output identical to pre-stack `main` behavior."""
    created = _track_renderer_instances(monkeypatch)

    async def _fake_run_step(agent, session, raw, seed, *, on_event=None, **kwargs):
        assert on_event is not None
        await on_event(SubAgentStartEvent(name="root", description="d", color="#000000"))
        await on_event(DelegationStartEvent(agent_name="code-expert", task="only task"))
        await on_event(LogEvent(message="Read foo.py", tool_name="read_file"))
        await on_event(DelegationDoneEvent(agent_name="code-expert"))
        await on_event(DoneEvent(thinking_chars=0, files_touched=[]))
        return TurnResult(outcome="ok", answer="done")

    monkeypatch.setattr(harness_turn, "run_step", _fake_run_step)

    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)

        result = await app._run_step(
            "do the thing", None,
            turn_id="t1", session_id="s1", conversation=conversation, stage=["harness"],
        )
        await pilot.pause()

        assert result.outcome == "ok"
        assert len(created) == 2
        root_renderer, p_renderer = created
        assert p_renderer._depth == 1
        assert p_renderer._spinner_task is None

        headers = [w for w in _messages(conversation) if w.has_class("header")]
        assert len(headers) == 2
        # P is depth 1 (first-level activation) — no "⎿" connector.
        done_lines = [
            w for w in conversation.children
            if isinstance(w, Static) and "Done (" in str(w.content)
        ]
        assert len(done_lines) == 1
        assert "⎿ Done" not in str(done_lines[0].content)


async def test_depth_indent_applied_only_below_depth_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 2: a depth-0 renderer's mounted widgets carry no added indent; a
    nested renderer's log/thinking/progress/Done widgets all carry the same
    left indent, consistently."""
    created = _track_renderer_instances(monkeypatch)

    async def _fake_run_step(agent, session, raw, seed, *, on_event=None, **kwargs):
        assert on_event is not None
        await on_event(SubAgentStartEvent(name="root", description="d", color="#000000"))
        await on_event(DelegationStartEvent(agent_name="code-expert", task="parent task"))
        await on_event(LogEvent(message="Read foo.py", tool_name="read_file"))
        await on_event(StatusUpdateEvent(total=10, progress=1))
        await on_event(DelegationDoneEvent(agent_name="code-expert"))
        await on_event(DoneEvent(thinking_chars=0, files_touched=[]))
        return TurnResult(outcome="ok", answer="done")

    monkeypatch.setattr(harness_turn, "run_step", _fake_run_step)

    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)

        await app._run_step(
            "do the thing", None,
            turn_id="t1", session_id="s1", conversation=conversation, stage=["harness"],
        )
        await pilot.pause()

        root_renderer, p_renderer = created
        assert root_renderer._depth == 0
        assert p_renderer._depth == 1

        # root's header widgets carry no margin.
        root_header = next(w for w in _messages(conversation) if w.has_class("header"))
        assert tuple(root_header.styles.margin) == (0, 0, 0, 0)

        # P's header, its Done line, are all indented consistently.
        headers = [w for w in _messages(conversation) if w.has_class("header")]
        p_header = headers[1]
        expected_margin = (0, 0, 0, tui_app._INDENT_PER_DEPTH)
        assert tuple(p_header.styles.margin) == expected_margin

        done_line = next(
            w for w in conversation.children
            if isinstance(w, Static) and "Done (" in str(w.content)
        )
        assert tuple(done_line.styles.margin) == expected_margin


async def test_first_level_activation_renders_no_connector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single delegation (root -> P, P at depth 1) is a first-level
    activation, not nested under a peer badge — its header dot column
    carries no `nested` CSS class and no "⎿" connector, same as root's own
    header. Badge text/namespace-color/ui_label content is unaffected."""

    async def _fake_run_step(agent, session, raw, seed, *, on_event=None, **kwargs):
        assert on_event is not None
        await on_event(SubAgentStartEvent(name="root", description="d", color="#000000"))
        await on_event(DelegationStartEvent(agent_name="code-expert", task="parent task", mission="fix things"))
        await on_event(DelegationDoneEvent(agent_name="code-expert"))
        await on_event(DoneEvent(thinking_chars=0, files_touched=[]))
        return TurnResult(outcome="ok", answer="done")

    monkeypatch.setattr(harness_turn, "run_step", _fake_run_step)

    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)

        await app._run_step(
            "do the thing", None,
            turn_id="t1", session_id="s1", conversation=conversation, stage=["harness"],
        )
        await pilot.pause()

        headers = [w for w in _messages(conversation) if w.has_class("header")]
        assert len(headers) == 2
        root_header, p_header = headers

        root_dot = root_header.query_one(".header-dot", Static)
        p_dot = p_header.query_one(".header-dot", Static)

        assert not root_dot.has_class("nested")
        assert not p_dot.has_class("nested")
        assert "⎿" not in str(root_dot.content)
        assert "⎿" not in str(p_dot.content)

        # Badge content unaffected.
        assert "code-expert" in p_header.text
        assert "fix things" in p_header.text


async def test_true_nested_delegation_renders_connector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subagent delegating to another subagent (C at depth 2, nested under
    P's own badge line) draws the `nested` CSS class and the "⎿" connector;
    P (depth 1) does not."""

    async def _fake_run_step(agent, session, raw, seed, *, on_event=None, **kwargs):
        assert on_event is not None
        await on_event(SubAgentStartEvent(name="root", description="d", color="#000000"))
        await on_event(DelegationStartEvent(agent_name="code-expert", task="parent task"))
        await on_event(DelegationStartEvent(agent_name="code-refactorer", task="child task"))
        await on_event(DelegationDoneEvent(agent_name="code-refactorer"))
        await on_event(DelegationDoneEvent(agent_name="code-expert"))
        await on_event(DoneEvent(thinking_chars=0, files_touched=[]))
        return TurnResult(outcome="ok", answer="done")

    monkeypatch.setattr(harness_turn, "run_step", _fake_run_step)

    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        conversation = app.query_one("#conversation", ScrollableContainer)

        await app._run_step(
            "do the thing", None,
            turn_id="t1", session_id="s1", conversation=conversation, stage=["harness"],
        )
        await pilot.pause()

        headers = [w for w in _messages(conversation) if w.has_class("header")]
        assert len(headers) == 3
        root_header, p_header, c_header = headers

        root_dot = root_header.query_one(".header-dot", Static)
        p_dot = p_header.query_one(".header-dot", Static)
        c_dot = c_header.query_one(".header-dot", Static)

        assert not root_dot.has_class("nested")
        assert not p_dot.has_class("nested")
        assert c_dot.has_class("nested")
        assert "⎿" not in str(root_dot.content)
        assert "⎿" not in str(p_dot.content)
        assert "⎿" in str(c_dot.content)
