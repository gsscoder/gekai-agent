from __future__ import annotations

from pathlib import Path

import pytest

from agent.harness.core import _recency_turns, _build_agent, _RECENCY_N
from agent.llm.types import Message
from agent.settings import Permissions
from agent.subagents import Subagent
from agent.tools.catalog import ALL_TOOLS, READ_TOOLS


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


def test_tools_block_full_set_for_unrestricted_subagent(tmp_path: Path):
    sub = Subagent(name="t", namespace="coding", description="d")  # tools=None -> all
    system, registered = _build(tmp_path, sub)
    assert registered == set(ALL_TOOLS)
    assert "use run_command for build, test, and git operations" in system
    assert "prefer read_file/grep/list_files over shell equivalents" in system


def test_direct_mode_tools_block_uses_full_set(tmp_path: Path):
    system, registered = _build(tmp_path, subagent=None)
    assert registered == set(ALL_TOOLS)
    assert system.startswith("base prompt")
    assert "<tools>" in system
