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
from agent.tools.delegate import make_delegate_tool, run_subagent

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


# ---------------------------------------------------------------------------
# `make_delegate_tool` — schema shape (plan-delegate-reintroduction Phase 2)
# ---------------------------------------------------------------------------

def _make_tool(targets: tuple[str, ...] = ("code-expert", "test-expert")):
    return make_delegate_tool(
        targets,
        "test-model", "key", "http://localhost", {}, _WORKING_DIR,
        _PERMISSIONS, None, None, None,
        frozenset({"read_file", "edit_file"}),
    )


def test_visible_schema_properties_are_exactly_agent_and_task() -> None:
    t = _make_tool()
    assert set(t.input_schema["properties"]) == {"agent", "task"}
    assert set(t.input_schema.get("required", [])) == {"agent", "task"}


def test_agent_enum_equals_declared_targets() -> None:
    t = _make_tool(("code-expert", "test-expert"))
    assert t.input_schema["properties"]["agent"]["enum"] == ["code-expert", "test-expert"]


def test_hidden_params_absent_from_schema() -> None:
    t = _make_tool()
    hidden = {
        "model", "api_key", "api_base", "extra_params", "working_dir",
        "permissions", "permission_callback", "bus", "hidden_grant_callback",
        "parent_tools",
    }
    assert hidden.isdisjoint(t.input_schema["properties"])
    assert t.pending_hidden_params == frozenset()


def test_required_permission_none_and_not_concurrency_safe() -> None:
    t = _make_tool()
    assert t.required_permission == "none"
    assert t.is_concurrency_safe is False


def test_description_composed_from_target_descriptions() -> None:
    from agent.subagents import SUBAGENTS

    targets = ("code-expert", "test-expert")
    t = _make_tool(targets)
    roster = {s.name: s for s in SUBAGENTS}
    for name in targets:
        assert roster[name].description in t.description


def test_calling_with_unknown_agent_returns_error_string_not_raise() -> None:
    t = _make_tool()
    result = run(t.call(agent="no-such-agent", task="do it"))
    assert result.startswith("[error]")
    assert "no-such-agent" in result


def test_calling_with_undeclared_but_real_agent_delegates_via_run_subagent() -> None:
    # the enum is a model-facing schema restriction, not a runtime check --
    # `make_delegate_tool` is a thin wrapper that defers entirely to
    # `run_subagent`'s own full-roster validation, never duplicating it.
    from agent.llm.types import Message, TextBlock

    fake_history = [Message(role="assistant", content=[TextBlock(text="done")])]
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=fake_history)
    t = _make_tool(("code-expert",))

    with _patched_build_agent(mock_agent):
        result = run(t.call(agent="test-expert", task="do it"))

    assert result == "done"


# ---------------------------------------------------------------------------
# `_build_agent` wiring — declared-not-ambient registration, root's
# exclusion, the depth-1 cap, and the tighten-only capability intersection
# (plan-delegate-reintroduction Phase 3)
# ---------------------------------------------------------------------------

def _build_real_agent(subagent=None, tools_override=None, can_delegate=True):
    import dataclasses

    from agent.harness.core import _build_agent
    from agent.subagents import SUBAGENTS

    code_expert = next(s for s in SUBAGENTS if s.name == "code-expert")
    resolved_subagent = None
    if subagent == "code-expert-no-delegate":
        resolved_subagent = code_expert
    elif subagent == "code-expert-delegates":
        resolved_subagent = dataclasses.replace(code_expert, delegates_to=("test-expert",))

    return _build_agent(
        "m", "k", None, {}, _WORKING_DIR, _PERMISSIONS, None,
        resolved_subagent.build_system_base() if resolved_subagent else "sys",
        None,
        subagent=resolved_subagent,
        tools_override=tools_override,
        can_delegate=can_delegate,
    )


def test_subagent_with_empty_delegates_to_gets_no_delegate_tool() -> None:
    agent = _build_real_agent(subagent="code-expert-no-delegate")
    assert "delegate" not in agent.tools


def test_subagent_with_targets_gets_exactly_one_delegate_tool() -> None:
    agent = _build_real_agent(subagent="code-expert-delegates")
    assert "delegate" in agent.tools
    names = [d.name for d in agent.tools.definitions()]
    assert names.count("delegate") == 1


def test_root_never_gets_delegate_tool_regardless_of_tool_scope() -> None:
    from agent.tools.catalog import RUNGS

    agent_full = _build_real_agent(subagent=None)
    agent_read_only = _build_real_agent(subagent=None, tools_override=frozenset(RUNGS[0]))
    assert "delegate" not in agent_full.tools
    assert "delegate" not in agent_read_only.tools


def test_can_delegate_false_suppresses_delegate_tool_even_with_targets() -> None:
    agent = _build_real_agent(subagent="code-expert-delegates", can_delegate=False)
    assert "delegate" not in agent.tools


def test_run_subagent_builds_child_with_can_delegate_false() -> None:
    # depth-1 cap: `run_subagent` must always suppress the child's own
    # ability to delegate, regardless of the target's own `delegates_to`.
    from agent.llm.types import Message, TextBlock

    fake_history = [Message(role="assistant", content=[TextBlock(text="done")])]
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=fake_history)

    with (
        patch("agent.harness.core._build_agent", return_value=mock_agent) as mock_build,
        patch("agent.harness.core._enrich_system_base", return_value="sys"),
    ):
        run(_call_run_subagent("code-expert"))

    assert mock_build.call_args.kwargs["can_delegate"] is False


def test_read_scoped_parent_produces_child_with_no_edit_or_fs_tools() -> None:
    # escalation guard: a parent scoped to the read rung must never let a
    # delegated child claw back edit/fs capability, even though the target
    # subagent's own `tools`/`tool_policy` would normally allow it.
    import dataclasses

    from agent.harness.core import _build_agent
    from agent.subagents import SUBAGENTS
    from agent.tools.catalog import EDIT_TOOLS, FS_TOOLS, RUNGS

    code_expert = next(s for s in SUBAGENTS if s.name == "code-expert")
    read_only_parent = dataclasses.replace(code_expert, delegates_to=("test-expert",))
    parent_agent = _build_agent(
        "m", "k", None, {}, _WORKING_DIR, _PERMISSIONS, None,
        read_only_parent.build_system_base(), None,
        subagent=read_only_parent,
        tools_override=frozenset(RUNGS[0]),  # read rung: no edit/fs tools
    )
    assert "delegate" in parent_agent.tools
    parent_tools = frozenset(d.name for d in parent_agent.tools.definitions() if d.name != "delegate")

    child = _build_agent(
        "m", "k", None, {}, _WORKING_DIR, _PERMISSIONS, None,
        "sys", None,
        subagent=next(s for s in SUBAGENTS if s.name == "test-expert"),  # no tools ceiling of its own
        tools_override=parent_tools,
        can_delegate=False,
    )
    child_tool_names = {d.name for d in child.tools.definitions()}
    assert child_tool_names.isdisjoint(EDIT_TOOLS)
    assert child_tool_names.isdisjoint(FS_TOOLS)


def test_run_subagent_intersects_parent_tools_into_tools_override() -> None:
    from agent.llm.types import Message, TextBlock

    fake_history = [Message(role="assistant", content=[TextBlock(text="done")])]
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=fake_history)

    with (
        patch("agent.harness.core._build_agent", return_value=mock_agent) as mock_build,
        patch("agent.harness.core._enrich_system_base", return_value="sys"),
    ):
        run(_call_run_subagent(
            "code-expert",
            tools_override=frozenset({"read_file", "edit_file", "write_file"}),
            parent_tools=frozenset({"read_file", "grep"}),
        ))

    assert mock_build.call_args.kwargs["tools_override"] == frozenset({"read_file"})


def test_run_subagent_uses_parent_tools_alone_when_no_tools_override_given() -> None:
    from agent.llm.types import Message, TextBlock

    fake_history = [Message(role="assistant", content=[TextBlock(text="done")])]
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=fake_history)

    with (
        patch("agent.harness.core._build_agent", return_value=mock_agent) as mock_build,
        patch("agent.harness.core._enrich_system_base", return_value="sys"),
    ):
        run(_call_run_subagent(
            "code-expert",
            parent_tools=frozenset({"read_file", "grep"}),
        ))

    assert mock_build.call_args.kwargs["tools_override"] == frozenset({"read_file", "grep"})
