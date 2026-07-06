from __future__ import annotations

import pytest

from agent.pipeline.plan import PlanStep, parse_plan
from agent.subagents import Subagent

_ROSTER = [
    Subagent(name="code-expert", namespace="coding", description="d", auto_assignable=True),
    Subagent(name="test-expert", namespace="testing", description="d", auto_assignable=True),
    Subagent(name="fact-checker", namespace="testing", description="d", auto_assignable=False),
]


def test_well_formed_plan_parses() -> None:
    raw = [
        {"agent": "main", "task": "scaffold repo"},
        {"agent": "code-expert", "task": "write code", "verify": "fact-checker"},
        {"agent": "test-expert", "task": "write tests for {{step_2}}", "repair": "fact-checker"},
    ]
    plan = parse_plan(raw, _ROSTER)
    assert plan == [
        PlanStep(agent="main", task="scaffold repo"),
        PlanStep(agent="code-expert", task="write code", verify="fact-checker"),
        PlanStep(agent="test-expert", task="write tests for {{step_2}}", repair="fact-checker"),
    ]


def test_unknown_agent_rejects_whole_plan() -> None:
    raw = [{"agent": "nonexistent-agent", "task": "do something"}]
    with pytest.raises(ValueError, match="nonexistent-agent"):
        parse_plan(raw, _ROSTER)


def test_empty_task_rejects_whole_plan() -> None:
    raw = [{"agent": "main", "task": ""}]
    with pytest.raises(ValueError, match="task"):
        parse_plan(raw, _ROSTER)


def test_dangling_ref_rejects_whole_plan() -> None:
    raw = [{"agent": "main", "task": "use {{step_5}}"}]
    with pytest.raises(ValueError, match="dangling ref"):
        parse_plan(raw, _ROSTER)


def test_post_planning_only_agent_in_phase1_agent_field_rejects() -> None:
    raw = [{"agent": "fact-checker", "task": "review the change"}]
    with pytest.raises(ValueError, match="fact-checker"):
        parse_plan(raw, _ROSTER)


def test_auto_assignable_agent_in_verify_field_rejects() -> None:
    raw = [{"agent": "main", "task": "scaffold", "verify": "code-expert"}]
    with pytest.raises(ValueError, match="code-expert"):
        parse_plan(raw, _ROSTER)


def test_empty_plan_rejects() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        parse_plan([], _ROSTER)
