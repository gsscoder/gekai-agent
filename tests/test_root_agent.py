from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.harness import core as harness_core
from agent.harness.core import Harness, _recency_turns, _build_agent, _RECENCY_N
from agent.llm.resolve import ResolvedTier
from agent.llm.tiers import TierName, TierPolicy
from agent.llm.types import Message, TextBlock
from agent.pipeline.plan import Task, TaskGraph
from agent.harness.interpreter import StepResult, TaskGraphHalted
from agent.session import Session
from agent.settings import Permissions
from agent.subagents import Subagent
from agent.tools.catalog import ALL_TOOLS, READ_TOOLS


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _recency_turns
# ---------------------------------------------------------------------------

def _msgs(*pairs: tuple[str, str]) -> list[dict]:
    """Build a session.messages list: system + n user/assistant pairs + trailing user."""
    out: list[dict] = [{"role": "system", "content": "workspace context"}]
    for user, assistant in pairs:
        out.append({"role": "user", "content": user})
        out.append({"role": "assistant", "content": assistant})
    out.append({"role": "user", "content": "current request"})
    return out


def test_recency_turns_returns_last_n_pairs():
    messages = _msgs(("u1", "a1"), ("u2", "a2"), ("u3", "a3"))
    result = _recency_turns(messages, 2)
    assert len(result) == 4
    assert [m.role for m in result] == ["user", "assistant", "user", "assistant"]
    assert result[0].content == "u2"
    assert result[3].content == "a3"


def test_recency_turns_excludes_current_user_input():
    messages = _msgs(("u1", "a1"))
    result = _recency_turns(messages, 2)
    # current request ("current request") must NOT appear
    contents = [m.content for m in result]
    assert "current request" not in contents


def test_recency_turns_no_system_messages():
    messages = _msgs(("u1", "a1"), ("u2", "a2"))
    result = _recency_turns(messages, 2)
    assert all(m.role != "system" for m in result)


def test_recency_turns_fewer_turns_than_n():
    messages = _msgs(("u1", "a1"))
    result = _recency_turns(messages, 2)
    assert len(result) == 2
    assert result[0].content == "u1"
    assert result[1].content == "a1"


def test_recency_turns_empty_history():
    # only system + current user input, no prior turns
    messages = [
        {"role": "system", "content": "ws"},
        {"role": "user", "content": "current"},
    ]
    result = _recency_turns(messages, 2)
    assert result == []


def test_recency_n_constant():
    assert _RECENCY_N == 2


# ---------------------------------------------------------------------------
# direct vs spawn mode: recency selection
# ---------------------------------------------------------------------------
# `Harness.stream` picks `prior = [] if subagent else _recency_turns(...)`.
# Direct mode (no subagent) carries recency context; spawn mode (a subagent
# is given) starts cold. This mirrors that selection without exercising the
# full Agent/EventBus loop.

def test_direct_mode_carries_recency():
    messages = _msgs(("u1", "a1"), ("u2", "a2"))
    subagent = None
    prior = [] if subagent else _recency_turns(messages, _RECENCY_N)
    assert prior != []
    assert [m.content for m in prior] == ["u1", "a1", "u2", "a2"]


def test_spawn_mode_is_cold():
    messages = _msgs(("u1", "a1"), ("u2", "a2"))
    subagent = object()  # stand-in: any truthy subagent value
    prior = [] if subagent else _recency_turns(messages, _RECENCY_N)
    assert prior == []


# ---------------------------------------------------------------------------
# _build_agent: <tools> block reflects the effective (allowlist + permission
# filtered) registered tool set, not the subagent's declared allowlist
# ---------------------------------------------------------------------------

_FULL_PERMS = Permissions(read=True, write=True, exec=True)


def _build(tmp_path: Path, subagent: Subagent | None, permissions: Permissions = _FULL_PERMS) -> tuple:
    base = subagent.build_system_base() if subagent else "base prompt"
    agent = _build_agent(
        "dummy-model", "dummy-key", None, {},
        tmp_path, permissions, None, base, None,
        subagent=subagent,
    )
    assert agent.system is not None
    registered = {t.name for t in agent.tools._tools.values()}
    return agent.system, registered


def test_tools_block_matches_subagent_allowlist(tmp_path: Path):
    sub = Subagent(name="t", namespace="coding", description="d", tools=list(READ_TOOLS))
    system, registered = _build(tmp_path, sub)
    assert registered == set(READ_TOOLS)
    assert "run_command" not in system
    assert "edit_file" not in system
    assert "you MUST use tools to read actual files" in system


def test_tools_block_narrows_with_permissions(tmp_path: Path):
    # full allowlist but read-only session permissions and no permission_callback
    # -> write/exec tools are filtered out, and the prompt must not reference them
    sub = Subagent(name="t", namespace="coding", description="d", tools=list(ALL_TOOLS))
    system, registered = _build(tmp_path, sub, permissions=Permissions(read=True, write=False, exec=False))
    assert "run_command" not in registered
    assert "edit_file" not in registered
    assert "use run_command for build, test, and git operations" not in system


# ---------------------------------------------------------------------------
# _build_agent: tools_override (plan 31 Phase 3, assignment-time tool
# scoping) — a second, narrower filter applied on top of the existing
# subagent.tools/permission filtering, not a replacement for it.
# ---------------------------------------------------------------------------


def test_build_agent_tools_override_narrows_selected(tmp_path: Path):
    agent = _build_agent(
        "dummy-model", "dummy-key", None, {},
        tmp_path, _FULL_PERMS, None, "base prompt", None,
        tools_override=frozenset(READ_TOOLS),
    )
    registered = {t.name for t in agent.tools._tools.values()}
    assert registered == set(READ_TOOLS)


def test_build_agent_tools_override_none_leaves_selected_unchanged(tmp_path: Path):
    agent_default = _build_agent(
        "dummy-model", "dummy-key", None, {},
        tmp_path, _FULL_PERMS, None, "base prompt", None,
    )
    agent_explicit_none = _build_agent(
        "dummy-model", "dummy-key", None, {},
        tmp_path, _FULL_PERMS, None, "base prompt", None,
        tools_override=None,
    )
    registered_default = {t.name for t in agent_default.tools._tools.values()}
    registered_explicit_none = {t.name for t in agent_explicit_none.tools._tools.values()}
    assert registered_default == registered_explicit_none == set(ALL_TOOLS)


def test_build_agent_tools_override_still_respects_subagent_allowlist(tmp_path: Path):
    # tools_override is a second, narrower filter on top of subagent.tools —
    # it must not widen the grant back past the subagent's own allowlist.
    sub = Subagent(name="t", namespace="coding", description="d", tools=list(READ_TOOLS))
    agent = _build_agent(
        "dummy-model", "dummy-key", None, {},
        tmp_path, _FULL_PERMS, None, sub.build_system_base(), None,
        subagent=sub,
        tools_override=frozenset(ALL_TOOLS),  # wider than the subagent's own allowlist
    )
    registered = {t.name for t in agent.tools._tools.values()}
    assert registered == set(READ_TOOLS)


def test_tools_block_full_set_for_unrestricted_subagent(tmp_path: Path):
    sub = Subagent(name="t", namespace="coding", description="d")  # tools=None -> all
    system, registered = _build(tmp_path, sub)
    assert registered == set(ALL_TOOLS)
    assert "use run_command for build, test, and git operations" in system
    assert "prefer read_file/grep/list_files over shell equivalents" in system


def test_direct_mode_tools_block_uses_full_set(tmp_path: Path):
    system, registered = _build(tmp_path, subagent=None)
    # root has the full workspace tool set; no delegate tool (plan 27
    # decision 11 — removed from root outright, the harness owns cross-agent
    # control flow instead)
    assert set(ALL_TOOLS) <= registered
    assert "delegate" not in registered
    assert system.startswith("base prompt")
    assert "<tools>" in system


# ---------------------------------------------------------------------------
# `Harness._respond`: root absorbs the Responder (plan 32 Phase 3) — the
# synthesis call must carry session recency, not just the current graph's
# own summary/step outputs.
# ---------------------------------------------------------------------------


def _make_respond_harness() -> Harness:
    tier = ResolvedTier(model="test-model", api_key="key", api_base="http://localhost", extra_params={})
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE))
    return Harness(
        resolve=lambda _tier, _touchpoint: tier,
        sequencer_policy=TierPolicy(default=TierName.CORE, allowed=(TierName.SUPP, TierName.CORE)),
        root_dispatch_policy=policy,
        subagent_dispatch_policy=policy,
    )


def _spy_build_agent_capturing_run(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replaces `_build_agent` with a fake whose `.run()` is an `AsyncMock`
    capturing exactly what `_respond` sends the model, instead of a real
    provider call."""
    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(
        return_value=[Message(role="assistant", content=[TextBlock(text="synthesized answer")])]
    )
    monkeypatch.setattr(harness_core, "_build_agent", lambda *a, **kw: fake_agent)
    return fake_agent


def test_respond_success_path_carries_session_recency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = _make_respond_harness()
    fake_agent = _spy_build_agent_capturing_run(monkeypatch)

    session = Session(working_dir=tmp_path, permissions=Permissions(read=True, write=True, exec=True))
    session.messages = [
        {"role": "system", "content": "workspace context"},
        {"role": "user", "content": "the prior turn's request"},
        {"role": "assistant", "content": "the prior turn's answer"},
        {"role": "user", "content": "current request"},
    ]
    graph = TaskGraph(summary="did the thing", steps=[Task(agent="code-expert", instruction="do it", mission="do it")])
    results = [StepResult(step=graph[0], output="code-expert's step output")]

    answer, event = _run(harness._respond(
        "current request", session, graph, results,
        halted=None, permission_callback=None, hidden_grant_callback=None,
    ))

    assert answer == "synthesized answer"
    assert event is not None and event.fell_back is False

    prior_messages = fake_agent.run.await_args.args[0]
    contents = [m.content for m in prior_messages]
    assert "the prior turn's request" in contents
    assert "the prior turn's answer" in contents
    # the graph's own summary/step outputs must also reach the model, not
    # just prior-turn recency
    assert any("did the thing" in c and "code-expert's step output" in c for c in contents)


def test_respond_falls_back_to_recap_on_synthesis_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = _make_respond_harness()

    def _raise(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(harness_core, "_build_agent", _raise)

    session = Session(working_dir=tmp_path, permissions=Permissions(read=True, write=True, exec=True))
    graph = TaskGraph(summary="did the thing", steps=[Task(agent="code-expert", instruction="do it", mission="do it")])
    results = [StepResult(step=graph[0], output="output")]

    answer, event = _run(harness._respond(
        "current request", session, graph, results,
        halted=None, permission_callback=None, hidden_grant_callback=None,
    ))

    assert answer == "did the thing"  # mechanical `_recap` fallback (graph.summary)
    assert event is not None and event.fell_back is True


def test_respond_halted_path_falls_back_to_recap_on_synthesis_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    harness = _make_respond_harness()

    def _raise(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(harness_core, "_build_agent", _raise)

    session = Session(working_dir=tmp_path, permissions=Permissions(read=True, write=True, exec=True))
    graph = TaskGraph(summary="did the thing", steps=[Task(agent="code-expert", instruction="do it", mission="do it")])
    halted = TaskGraphHalted(0, graph[0], "empty dispatch output")

    answer, event = _run(harness._respond(
        "current request", session, graph, [],
        halted=halted, permission_callback=None, hidden_grant_callback=None,
    ))

    assert "HALTED" in answer
    assert event is not None and event.fell_back is True
