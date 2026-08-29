"""Plan 35 v3: the directive audit wired to session start, plus the
`#directive-notice` TUI surface.

Two layers, matching how `agent/agent.py::start_directive_audit` and
`agent/tui/app.py::_apply_directive_verdict` split the work:

- `GekaiAgent.start_directive_audit` (no TUI): cache hit skips the LLM call
  entirely, a cache miss calls then caches, a swallowed failure never raises
  and never leaves an "in flight" state stuck. No real network/credentials
  are ever used — `agent.agent.resolve_touchpoint`/`agent.agent.Auditor` are
  monkeypatched per test.
- `GekaiApp._apply_directive_verdict` / the session-start wiring (real
  `GekaiApp` via Textual's `Pilot`, mirrors `tests/test_tiers_grid_flow.py`):
  the notice states each drive the one-slot notice to the right state, and
  the notice resets to hidden at session start and on `/clear`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.color import Color
from textual.widgets import Static

from agent import agent as agent_module
from agent import logging as agent_logging
from agent.agent import GekaiAgent
from agent.commands.registry import CommandRegistry
from agent.directive_audit import (
    AuditVerdict,
    file_sha,
    load_cached_verdict,
    save_cached_verdict,
)
from agent.llm.resolve import ResolvedTier, TierResolutionError
from agent.settings import Permissions, save_directive_audit_enabled
from agent.tui.app import GekaiApp


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_logging.Path, "home", classmethod(lambda cls: tmp_path))


def _make_agent(working_dir: Path) -> GekaiAgent:
    return GekaiAgent(working_dir=working_dir, permissions=Permissions(read=True, write=True, exec=True))


def _write_gekai_md(working_dir: Path, text: str) -> None:
    (working_dir / "GEKAI.md").write_text(text, encoding="utf-8")


def _fake_resolved() -> ResolvedTier:
    return ResolvedTier(model="fake-model", api_key="fake-key", api_base="http://fake.example.com", extra_params={})


def _patch_resolve_touchpoint(monkeypatch: pytest.MonkeyPatch, *, error: bool = False) -> list[str]:
    calls: list[str] = []

    def _fake(name: str, catalog: object, bindings: object) -> ResolvedTier:
        calls.append(name)
        if error:
            raise TierResolutionError("tier configuration is incomplete")
        return _fake_resolved()

    monkeypatch.setattr(agent_module, "resolve_touchpoint", _fake)
    return calls


def _patch_auditor(monkeypatch: pytest.MonkeyPatch, verdict: AuditVerdict) -> list[str]:
    """Stands in for `agent.agent.Auditor` — records every file text it was
    asked to audit and returns a fixed verdict, never touching the network."""
    calls: list[str] = []

    class _StubAuditor:
        def __init__(self, *, model: str, api_key: str, api_base: str | None, extra_params: dict) -> None:
            pass

        async def audit(self, file_text: str) -> AuditVerdict:
            calls.append(file_text)
            return verdict

    monkeypatch.setattr(agent_module, "Auditor", _StubAuditor)
    return calls


async def _drain(agent: GekaiAgent) -> None:
    """Awaits every currently-tracked background task to completion."""
    tasks = list(agent._background_tasks)
    if tasks:
        await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# GekaiAgent.start_directive_audit — cache, call, swallow
# ---------------------------------------------------------------------------


def test_cache_hit_skips_the_call_entirely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_gekai_md(tmp_path, "always answer in haiku")
    agent = _make_agent(tmp_path)

    def _explode(*a: object, **kw: object) -> None:
        raise AssertionError("Auditor must not be constructed on a cache hit")

    monkeypatch.setattr(agent_module, "Auditor", _explode)
    monkeypatch.setattr(agent_module, "resolve_touchpoint", _explode)

    session = agent.start_session()
    cached = AuditVerdict(has_directives=True, raw="YES")
    save_cached_verdict(tmp_path, "GEKAI.md", session.gekai_md.sha, cached)

    seen: list[tuple[str, AuditVerdict | None]] = []
    agent.start_directive_audit(session, lambda path, v: seen.append((path, v)))

    assert agent._background_tasks == set()  # no task fired — it's all synchronous
    assert seen == [("GEKAI.md", cached)]


def test_cache_miss_calls_then_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_gekai_md(tmp_path, "always answer in haiku")
    agent = _make_agent(tmp_path)
    session = agent.start_session()

    verdict = AuditVerdict(has_directives=False, raw="NO")
    resolve_calls = _patch_resolve_touchpoint(monkeypatch)
    auditor_calls = _patch_auditor(monkeypatch, verdict)

    seen: list[tuple[str, AuditVerdict | None]] = []

    async def _run() -> None:
        agent.start_directive_audit(session, lambda path, v: seen.append((path, v)))
        assert seen == [("GEKAI.md", None)]  # in-flight signal fired synchronously
        await _drain(agent)

    asyncio.run(_run())

    assert resolve_calls == ["directive-audit"]
    assert auditor_calls == ["always answer in haiku"]
    assert seen == [("GEKAI.md", None), ("GEKAI.md", verdict)]

    reread = load_cached_verdict(tmp_path, "GEKAI.md", session.gekai_md.sha)
    assert reread == verdict


def test_disabled_setting_skips_entirely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_gekai_md(tmp_path, "always answer in haiku")
    save_directive_audit_enabled(tmp_path, False)
    agent = _make_agent(tmp_path)
    session = agent.start_session()

    def _explode(*a: object, **kw: object) -> None:
        raise AssertionError("nothing should be called while disabled")

    monkeypatch.setattr(agent_module, "Auditor", _explode)
    monkeypatch.setattr(agent_module, "resolve_touchpoint", _explode)

    seen: list[tuple[str, AuditVerdict | None]] = []
    agent.start_directive_audit(session, lambda path, v: seen.append((path, v)))

    assert seen == []
    assert agent._background_tasks == set()


def test_no_gekai_md_is_a_no_op(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    session = agent.start_session()
    assert session.gekai_md is None

    seen: list[tuple[str, AuditVerdict | None]] = []
    agent.start_directive_audit(session, lambda path, v: seen.append((path, v)))

    assert seen == []
    assert agent._background_tasks == set()


def test_swallowed_failure_leaves_no_error_visible_and_no_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tier config incomplete (or any other failure) degrades to the safe
    `AuditVerdict()` (has_directives=False) — no crash and no stuck
    "in flight" line."""
    _write_gekai_md(tmp_path, "always answer in haiku")
    agent = _make_agent(tmp_path)
    session = agent.start_session()

    _patch_resolve_touchpoint(monkeypatch, error=True)

    seen: list[tuple[str, AuditVerdict | None]] = []

    async def _run() -> None:
        agent.start_directive_audit(session, lambda path, v: seen.append((path, v)))
        await _drain(agent)  # must not raise

    asyncio.run(_run())

    assert seen == [("GEKAI.md", None), ("GEKAI.md", AuditVerdict())]
    # nothing persisted to the cache — there is no real verdict to remember
    assert load_cached_verdict(tmp_path, "GEKAI.md", session.gekai_md.sha) is None


def test_auditor_exception_is_also_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_gekai_md(tmp_path, "always answer in haiku")
    agent = _make_agent(tmp_path)
    session = agent.start_session()

    _patch_resolve_touchpoint(monkeypatch)

    class _RaisingAuditor:
        def __init__(self, **kw: object) -> None:
            pass

        async def audit(self, file_text: str) -> AuditVerdict:
            raise RuntimeError("boom")

    monkeypatch.setattr(agent_module, "Auditor", _RaisingAuditor)

    seen: list[tuple[str, AuditVerdict | None]] = []

    async def _run() -> None:
        agent.start_directive_audit(session, lambda path, v: seen.append((path, v)))
        await _drain(agent)  # must not raise

    asyncio.run(_run())

    assert seen == [("GEKAI.md", None), ("GEKAI.md", AuditVerdict())]


def test_completion_emits_directive_audit_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_gekai_md(tmp_path, "always answer in haiku")
    agent = _make_agent(tmp_path)
    session = agent.start_session()

    _patch_resolve_touchpoint(monkeypatch)
    _patch_auditor(monkeypatch, AuditVerdict(has_directives=True, raw="YES"))

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(agent.events, "emit", lambda evt, **fields: events.append((evt, fields)))

    async def _run() -> None:
        agent.start_directive_audit(session, None)
        await _drain(agent)

    asyncio.run(_run())

    matches = [f for evt, f in events if evt == "directive_audit"]
    assert len(matches) == 1
    assert matches[0]["path"] == "GEKAI.md"
    assert matches[0]["has_directives"] is True
    assert matches[0]["cached"] is False


# ---------------------------------------------------------------------------
# GekaiApp._apply_directive_verdict — the notice states, one slot
# ---------------------------------------------------------------------------


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


@pytest.fixture
def _stub_init_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not autouse: only the `_apply_directive_verdict`-focused tests below
    want `_init_session` skipped (its real first-run permissions prompt and
    history/timeline replay are irrelevant there). The session-start/`/clear`
    wiring tests further down need the real `_init_session` — that's the
    exact code path under test."""
    async def _fake_init_session(self: GekaiApp) -> None:
        self._session = self._agent.start_session()

    monkeypatch.setattr(GekaiApp, "_init_session", _fake_init_session)


@pytest.mark.asyncio
async def test_notice_widget_matches_copy_notice_css_shape(tmp_path: Path, _stub_init_session: None) -> None:
    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        notice = app.query_one("#directive-notice", Static)
        copy_notice = app.query_one("#copy-notice", Static)
        assert notice.display is False  # hidden by default, like #copy-notice
        assert copy_notice.display is False

        assert notice.styles.height == copy_notice.styles.height
        assert notice.styles.text_align == copy_notice.styles.text_align
        assert notice.styles.padding == copy_notice.styles.padding


@pytest.mark.asyncio
async def test_gekai_md_in_flight_state(tmp_path: Path, _stub_init_session: None) -> None:
    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._apply_directive_verdict("GEKAI.md", None)
        notice = app.query_one("#directive-notice", Static)
        assert notice.display is True
        assert str(notice.render()) == "⋯ checking GEKAI.md"
        assert notice.styles.color == Color.parse("grey")


@pytest.mark.asyncio
async def test_gekai_md_no_shows_green_loaded(tmp_path: Path, _stub_init_session: None) -> None:
    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        notice = app.query_one("#directive-notice", Static)

        app._apply_directive_verdict("GEKAI.md", None)  # first flip it visible (in flight)
        assert notice.display is True

        app._apply_directive_verdict("GEKAI.md", AuditVerdict(has_directives=False, raw="NO"))
        assert notice.display is True
        assert str(notice.render()) == "✓ GEKAI.md loaded"
        assert notice.styles.color == Color.parse("green")


@pytest.mark.asyncio
async def test_gekai_md_yes_is_yellow(tmp_path: Path, _stub_init_session: None) -> None:
    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._apply_directive_verdict("GEKAI.md", AuditVerdict(has_directives=True, raw="YES"))
        notice = app.query_one("#directive-notice", Static)
        assert notice.display is True
        assert str(notice.render()) == "⚠  GEKAI.md contains agent directives"
        assert notice.styles.color == Color.parse("yellow")


@pytest.mark.asyncio
async def test_foreign_file_yes_is_yellow(tmp_path: Path, _stub_init_session: None) -> None:
    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._apply_directive_verdict("AGENTS.md", AuditVerdict(has_directives=True, raw="YES"))
        notice = app.query_one("#directive-notice", Static)
        assert notice.display is True
        assert str(notice.render()) == "⚠  AGENTS.md contains agent directives"
        assert notice.styles.color == Color.parse("yellow")


@pytest.mark.asyncio
async def test_foreign_file_no_shows_nothing(tmp_path: Path, _stub_init_session: None) -> None:
    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        notice = app.query_one("#directive-notice", Static)

        app._apply_directive_verdict("AGENTS.md", None)  # first flip it visible (in flight)
        assert notice.display is True

        app._apply_directive_verdict("AGENTS.md", AuditVerdict(has_directives=False, raw="NO"))
        assert notice.display is False
        assert str(notice.render()) == ""


@pytest.mark.asyncio
async def test_last_verdict_wins_no_per_file_store(tmp_path: Path, _stub_init_session: None) -> None:
    """One slot: a second verdict for a different file overwrites the first
    outright — there is no dict keyed by path anywhere in the TUI state."""
    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        notice = app.query_one("#directive-notice", Static)

        app._apply_directive_verdict("GEKAI.md", AuditVerdict(has_directives=True, raw="YES"))
        assert str(notice.render()) == "⚠  GEKAI.md contains agent directives"

        app._apply_directive_verdict("AGENTS.md", AuditVerdict(has_directives=True, raw="YES"))
        assert str(notice.render()) == "⚠  AGENTS.md contains agent directives"

        # And no attribute anywhere on the app holds a per-file mapping.
        assert not any(
            isinstance(getattr(app, name, None), dict) and "directive" in name
            for name in vars(app)
        )


# ---------------------------------------------------------------------------
# session-start / /clear wiring — real (unstubbed) GekaiAgent + a real
# GEKAI.md on disk, driven through a real mounted GekaiApp via Pilot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_start_fires_the_audit_and_lands_the_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_gekai_md(tmp_path, "always answer in haiku")
    # Construct the real agent BEFORE patching `resolve_touchpoint` — its
    # constructor resolves the 4 unrelated touchpoints (estimator, sequencer,
    # ...) against the real (empty, isolated-home) tier config and must be
    # left free to hit its own real "not configured" path, same as every
    # other test that builds an unconfigured `GekaiAgent`.
    real_agent = _make_agent(tmp_path)
    _patch_resolve_touchpoint(monkeypatch)
    _patch_auditor(monkeypatch, AuditVerdict(has_directives=True, raw="YES"))

    app = GekaiApp(
        agent=real_agent, registry=CommandRegistry(), working_dir=tmp_path, version="test", branch=None,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        for _ in range(20):
            await _drain(real_agent)
            await pilot.pause()
            notice = app.query_one("#directive-notice", Static)
            if notice.display and str(notice.render()) != "⋯ checking GEKAI.md":
                break

        notice = app.query_one("#directive-notice", Static)
        assert notice.display is True
        assert str(notice.render()) == "⚠  GEKAI.md contains agent directives"


@pytest.mark.asyncio
async def test_cache_hit_shows_notice_with_no_auditor_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_gekai_md(tmp_path, "always answer in haiku")
    cached = AuditVerdict(has_directives=False, raw="NO")
    save_cached_verdict(tmp_path, "GEKAI.md", file_sha("always answer in haiku"), cached)

    real_agent = _make_agent(tmp_path)  # built before patching — see the sibling test above

    def _explode(*a: object, **kw: object) -> None:
        raise AssertionError("a cache hit must never construct an Auditor")

    monkeypatch.setattr(agent_module, "Auditor", _explode)
    monkeypatch.setattr(agent_module, "resolve_touchpoint", _explode)

    app = GekaiApp(
        agent=real_agent, registry=CommandRegistry(), working_dir=tmp_path, version="test", branch=None,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        notice = app.query_one("#directive-notice", Static)
        assert notice.display is True
        assert str(notice.render()) == "✓ GEKAI.md loaded"


@pytest.mark.asyncio
async def test_restored_session_runs_the_audit_but_suppresses_the_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restore is not a fresh session start from the user's perspective —
    GEKAI.md was already shown loaded before the app closed, so re-flashing
    the notice reads as a reload that never happened. The audit itself must
    still run unconditionally (it's what keeps `session.gekai_md` populated
    for the system-base injection), just with no visible callback."""
    _write_gekai_md(tmp_path, "always answer in haiku")
    cached = AuditVerdict(has_directives=False, raw="NO")
    save_cached_verdict(tmp_path, "GEKAI.md", file_sha("always answer in haiku"), cached)

    real_agent = _make_agent(tmp_path)  # built before patching — see the sibling test above

    def _explode(*a: object, **kw: object) -> None:
        raise AssertionError("a cache hit must never construct an Auditor")

    monkeypatch.setattr(agent_module, "Auditor", _explode)
    monkeypatch.setattr(agent_module, "resolve_touchpoint", _explode)

    app = GekaiApp(
        agent=real_agent, registry=CommandRegistry(), working_dir=tmp_path, version="test", branch=None,
        restored_id="some-earlier-session-id",
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        notice = app.query_one("#directive-notice", Static)
        assert notice.display is False
        assert app._session is not None
        assert app._session.gekai_md is not None


@pytest.mark.asyncio
async def test_notice_resets_to_hidden_at_session_start_and_on_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_agent = _make_agent(tmp_path)  # no GEKAI.md at all — nothing to audit
    app = GekaiApp(
        agent=real_agent, registry=CommandRegistry(), working_dir=tmp_path, version="test", branch=None,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        notice = app.query_one("#directive-notice", Static)
        assert notice.display is False  # nothing fired — no GEKAI.md, hidden at start

        # Simulate a stale notice left over from a previous turn/session,
        # then confirm /clear resets it before the (no-op, no GEKAI.md) audit
        # would land anything.
        app._apply_directive_verdict("GEKAI.md", AuditVerdict(has_directives=True, raw="YES"))
        assert notice.display is True

        await app._clear_session()
        await pilot.pause()

        assert notice.display is False
