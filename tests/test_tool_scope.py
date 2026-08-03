from __future__ import annotations

import pytest

from agent.harness.tool_scope import ROOT_TOOL_POLICY, scope
from agent.subagents import ToolPolicy
from agent.tools.catalog import RUNGS


def test_root_tool_policy_ceiling_is_full() -> None:
    assert ROOT_TOOL_POLICY.ceiling == len(RUNGS) - 1


@pytest.mark.parametrize("ceiling", list(range(len(RUNGS))))
def test_policy_accepts_in_range_ceiling(ceiling: int) -> None:
    policy = ToolPolicy(ceiling=ceiling)
    assert policy.ceiling == ceiling


@pytest.mark.parametrize("ceiling", [-1, len(RUNGS), len(RUNGS) + 5])
def test_policy_rejects_out_of_range_ceiling(ceiling: int) -> None:
    with pytest.raises(ValueError, match="out of range"):
        ToolPolicy(ceiling=ceiling)


def test_none_policy_defaults_to_full() -> None:
    tools, reason = scope(None, None)
    assert tools == frozenset(RUNGS[-1])
    assert reason == "default"


def test_none_step_scope_is_neutral_default() -> None:
    policy = ToolPolicy(ceiling=1)
    tools, reason = scope(policy, None)
    assert tools == frozenset(RUNGS[1])
    assert reason == "default"


def test_unrecognized_step_scope_is_neutral_default() -> None:
    policy = ToolPolicy(ceiling=1)
    tools, reason = scope(policy, "bogus")
    assert tools == frozenset(RUNGS[1])
    assert reason == "default"


@pytest.mark.parametrize(
    "step_scope, rung_index",
    [("read", 0), ("edit", 1), ("fs", 2)],
)
def test_scope_maps_step_scope_to_rung_when_within_ceiling(step_scope: str, rung_index: int) -> None:
    policy = ToolPolicy(ceiling=len(RUNGS) - 1)
    tools, reason = scope(policy, step_scope)
    if rung_index >= policy.ceiling:
        assert tools == frozenset(RUNGS[policy.ceiling])
        assert reason == "default"
    else:
        assert tools == frozenset(RUNGS[rung_index])
        assert reason == f"narrowed to {step_scope!r}"


def test_scope_clamps_when_requested_exceeds_ceiling() -> None:
    policy = ToolPolicy(ceiling=0)
    tools, reason = scope(policy, "fs")
    assert tools == frozenset(RUNGS[0])
    assert reason == "default"


def test_scope_clamps_when_requested_equals_ceiling() -> None:
    policy = ToolPolicy(ceiling=1)
    tools, reason = scope(policy, "edit")
    assert tools == frozenset(RUNGS[1])
    assert reason == "default"


def test_scope_narrows_when_requested_below_ceiling() -> None:
    policy = ToolPolicy(ceiling=2)
    tools, reason = scope(policy, "read")
    assert tools == frozenset(RUNGS[0])
    assert reason == "narrowed to 'read'"
