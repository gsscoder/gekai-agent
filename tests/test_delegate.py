"""Tests for the delegate tool: make_delegate_tool factory, recursion guard,
and unknown-agent error path.

Source: plan 25 Improvement 2 — subagents exposed to main as delegate(agent, task).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.subagents import SUBAGENTS
from agent.tools.delegate import make_delegate_tool
from agent.settings import Permissions


_PERMISSIONS = Permissions(read=True, write=True, exec=True)
_WORKING_DIR = Path(".")


def _make_delegate(**kwargs):
    defaults = dict(
        model="test-model",
        api_key="key",
        api_base="http://localhost",
        extra_params={},
        working_dir=_WORKING_DIR,
        permissions=_PERMISSIONS,
        permission_callback=None,
        bus=None,
        hidden_grant_callback=None,
    )
    defaults.update(kwargs)
    return make_delegate_tool(**defaults)


def run(coro):
    return asyncio.run(coro)


def test_delegate_tool_name_is_delegate() -> None:
    t = _make_delegate()
    assert t.name == "delegate"


def test_delegate_tool_has_agent_and_task_params() -> None:
    t = _make_delegate()
    props = t.input_schema.get("properties", {})
    assert "agent" in props
    assert "task" in props


def test_delegate_tool_required_permission_is_none() -> None:
    t = _make_delegate()
    assert t.required_permission == "none"


def test_delegate_unknown_agent_returns_error() -> None:
    t = _make_delegate()
    result = run(t.call(agent="no-such-agent", task="do something"))
    assert result.startswith("[error]")
    assert "no-such-agent" in result


def test_delegate_description_lists_user_invocable_agents() -> None:
    t = _make_delegate()
    expected_names = sorted(s.name for s in SUBAGENTS if s.user_invocable)
    for name in expected_names:
        assert name in t.description


def test_no_user_invocable_description_contains_donts() -> None:
    banned = ("not for", "that's main")
    for s in SUBAGENTS:
        if not s.user_invocable:
            continue
        lowered = s.description.lower()
        for phrase in banned:
            assert phrase not in lowered, f"{s.name} description contains {phrase!r}: {s.description}"


def test_delegate_tool_is_async() -> None:
    t = _make_delegate()
    assert t.is_async is True


def test_recursion_guard_no_delegate_tool_when_subagent() -> None:
    """_build_agent must not register the delegate tool when subagent is set."""
    from agent.harness.core import _build_agent
    from agent.settings import Permissions

    code_expert = next(s for s in SUBAGENTS if s.name == "code-expert")
    perms = Permissions(read=True, write=False, exec=False)

    with (
        patch("agent.harness.core.OpenAIAdapter"),
        patch("agent.harness.core.make_tools", return_value=[]),
    ):
        agent = _build_agent(
            "m", "k", None, {}, Path("."), perms, None,
            code_expert.build_system_base(), None,
            subagent=code_expert,
        )

    assert "delegate" not in agent.tools


def test_delegate_tool_present_for_main_agent() -> None:
    """_build_agent must register delegate when subagent is None."""
    from agent.harness.core import _build_agent

    perms = Permissions(read=True, write=True, exec=True)

    with (
        patch("agent.harness.core.OpenAIAdapter"),
        patch("agent.harness.core.make_tools", return_value=[]),
    ):
        agent = _build_agent(
            "m", "k", None, {}, Path("."), perms, None, "sys", None,
        )

    assert "delegate" in agent.tools


def test_delegate_call_returns_text_from_history() -> None:
    """Happy-path: delegate.call extracts text from the nested agent's message history."""
    from agent.llm.types import Message, TextBlock

    fake_history = [
        Message(role="assistant", content=[TextBlock(text="done")])
    ]

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=fake_history)

    t = _make_delegate()
    with (
        patch("agent.harness.core._build_agent", return_value=mock_agent),
        patch("agent.harness.core._enrich_system_base", return_value="sys"),
    ):
        result = run(t.call(agent="code-expert", task="do it"))

    assert result == "done"


def test_delegate_emits_delegation_started_then_completed_on_success() -> None:
    """REQ: problem 2's TUI badge relies on `delegate()` emitting
    `DelegationStarted` then `DelegationCompleted` (in that order) on the
    shared bus around a successful nested run."""
    from agent.llm.types import Message, TextBlock

    fake_history = [Message(role="assistant", content=[TextBlock(text="done")])]

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=fake_history)

    mock_bus = MagicMock()
    t = _make_delegate(bus=mock_bus)
    with (
        patch("agent.harness.core._build_agent", return_value=mock_agent),
        patch("agent.harness.core._enrich_system_base", return_value="sys"),
    ):
        result = run(t.call(agent="code-expert", task="do it"))

    assert result == "done"
    assert [call.args[0].__class__.__name__ for call in mock_bus.emit.call_args_list] == [
        "DelegationStarted",
        "DelegationCompleted",
    ]
    started, completed = (call.args[0] for call in mock_bus.emit.call_args_list)
    assert started.agent == "code-expert"
    assert started.task == "do it"
    assert completed.agent == "code-expert"
    # Same nested run_id correlates the pair.
    assert started.run_id == completed.run_id
    assert started.run_id


def test_delegate_emits_delegation_completed_when_nested_run_raises() -> None:
    """REQ: `DelegationCompleted` must still fire (via try/finally) even when
    the nested run raises — otherwise a failed delegation would leave the TUI
    showing an open specialist badge forever."""
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=RuntimeError("boom"))

    mock_bus = MagicMock()
    t = _make_delegate(bus=mock_bus)
    with (
        patch("agent.harness.core._build_agent", return_value=mock_agent),
        patch("agent.harness.core._enrich_system_base", return_value="sys"),
    ):
        result = run(t.call(agent="code-expert", task="do it"))

    assert result.startswith("[error]")
    assert "boom" in result
    assert [call.args[0].__class__.__name__ for call in mock_bus.emit.call_args_list] == [
        "DelegationStarted",
        "DelegationCompleted",
    ]
