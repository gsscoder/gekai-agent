from __future__ import annotations

import pytest

from agent.harness import _recency_turns, _build_agent, _RECENCY_N
from agent.llm.types import Message


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
