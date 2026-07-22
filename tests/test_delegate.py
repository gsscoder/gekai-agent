"""Tests for `run_subagent` (agent/tools/delegate.py): unknown-agent error,
happy path, and the DelegationStarted/Completed bus pair.

Source: plan 27 improvement 5 — the `delegate` tool is removed from main
outright; this nested-agent call is reused by the interpreter's `dispatch`
instead (cold subagent run, run-tagged events on the shared bus).
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agent.settings import Permissions
from agent.tools.delegate import run_subagent

_PERMISSIONS = Permissions(read=True, write=True, exec=True)
_WORKING_DIR = Path(".")


def run(coro):
    return asyncio.run(coro)


@contextlib.contextmanager
def _patched_build_agent(mock_agent):
    with (
        patch("agent.harness.core._build_agent", return_value=mock_agent),
        patch("agent.harness.core._enrich_system_base", return_value="sys"),
    ):
        yield


def _call_run_subagent(agent: str, task: str = "do it", **kwargs):
    defaults = dict(
        model="test-model", api_key="key", api_base="http://localhost",
        extra_params={}, working_dir=_WORKING_DIR, permissions=_PERMISSIONS,
        permission_callback=None, bus=None, hidden_grant_callback=None,
    )
    defaults.update(kwargs)
    return run_subagent(agent, task, **defaults)


def test_unknown_agent_returns_error() -> None:
    result = run(_call_run_subagent("no-such-agent"))
    assert result.startswith("[error]")
    assert "no-such-agent" in result


def test_returns_text_from_history() -> None:
    from agent.llm.types import Message, TextBlock

    fake_history = [Message(role="assistant", content=[TextBlock(text="done")])]
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=fake_history)

    with _patched_build_agent(mock_agent):
        result = run(_call_run_subagent("code-expert"))

    assert result == "done"


def test_emits_delegation_started_then_completed_on_success() -> None:
    from agent.llm.types import Message, TextBlock

    fake_history = [Message(role="assistant", content=[TextBlock(text="done")])]
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=fake_history)
    mock_bus = MagicMock()

    with _patched_build_agent(mock_agent):
        result = run(_call_run_subagent("code-expert", task="do it", bus=mock_bus))

    assert result == "done"
    assert [call.args[0].__class__.__name__ for call in mock_bus.emit.call_args_list] == [
        "DelegationStarted",
        "DelegationCompleted",
    ]
    started, completed = (call.args[0] for call in mock_bus.emit.call_args_list)
    assert started.agent == "code-expert"
    assert started.task == "do it"
    assert completed.agent == "code-expert"
    assert started.run_id == completed.run_id
    assert started.run_id


def test_emits_delegation_completed_when_nested_run_raises() -> None:
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=RuntimeError("boom"))
    mock_bus = MagicMock()

    with _patched_build_agent(mock_agent):
        result = run(_call_run_subagent("code-expert", bus=mock_bus))

    assert result.startswith("[error]")
    assert "boom" in result
    assert [call.args[0].__class__.__name__ for call in mock_bus.emit.call_args_list] == [
        "DelegationStarted",
        "DelegationCompleted",
    ]


def test_no_delegate_tool_registered_for_main_or_subagent() -> None:
    """decision 11: `delegate` is removed from main outright — no hybrid.
    `_build_agent` must never register a `delegate` tool, subagent or not."""
    from agent.harness.core import _build_agent
    from agent.subagents import SUBAGENTS

    code_expert = next(s for s in SUBAGENTS if s.name == "code-expert")

    with (
        patch("agent.harness.core.OpenAIAdapter"),
        patch("agent.harness.core.make_tools", return_value=[]),
    ):
        main_agent = _build_agent(
            "m", "k", None, {}, _WORKING_DIR, _PERMISSIONS, None, "sys", None,
        )
        sub_agent = _build_agent(
            "m", "k", None, {}, _WORKING_DIR, _PERMISSIONS, None,
            code_expert.build_system_base(), None,
            subagent=code_expert,
        )

    assert "delegate" not in main_agent.tools
    assert "delegate" not in sub_agent.tools
