from __future__ import annotations

import pytest

from agent.pipeline.plan import Task, parse_task_graph
from agent.subagents import Subagent

_ROSTER = [
    Subagent(name="code-expert", namespace="coding", description="d", auto_assignable=True),
    Subagent(name="test-expert", namespace="testing", description="d", auto_assignable=True),
    Subagent(name="fact-checker", namespace="testing", description="d", auto_assignable=False),
]


def test_well_formed_graph_parses() -> None:
    raw = {
        "summary": "build a library with tests",
        "steps": [
            {"agent": "main", "instruction": "scaffold repo", "mission": "scaffold the repo"},
            {"agent": "code-expert", "instruction": "write code", "mission": "write the code", "verify": "fact-checker"},
            {"agent": "test-expert", "instruction": "write tests for {{step_2}}", "mission": "write tests", "repair": "fact-checker"},
        ],
    }
    graph = parse_task_graph(raw, _ROSTER)
    assert graph.summary == "build a library with tests"
    assert graph == [
        Task(agent="main", instruction="scaffold repo", mission="scaffold the repo"),
        Task(agent="code-expert", instruction="write code", mission="write the code", verify="fact-checker"),
        Task(agent="test-expert", instruction="write tests for {{step_2}}", mission="write tests", repair="fact-checker"),
    ]


def test_unknown_agent_rejects_whole_graph() -> None:
    raw = {"summary": "s", "steps": [{"agent": "nonexistent-agent", "instruction": "do something"}]}
    with pytest.raises(ValueError, match="nonexistent-agent"):
        parse_task_graph(raw, _ROSTER)


def test_empty_instruction_rejects_whole_graph() -> None:
    raw = {"summary": "s", "steps": [{"agent": "main", "instruction": ""}]}
    with pytest.raises(ValueError, match="instruction"):
        parse_task_graph(raw, _ROSTER)


def test_dangling_ref_rejects_whole_graph() -> None:
    raw = {"summary": "s", "steps": [{"agent": "main", "instruction": "use {{step_5}}", "mission": "use it"}]}
    with pytest.raises(ValueError, match="dangling ref"):
        parse_task_graph(raw, _ROSTER)


def test_post_planning_only_agent_in_phase1_agent_field_rejects() -> None:
    raw = {"summary": "s", "steps": [{"agent": "fact-checker", "instruction": "review the change"}]}
    with pytest.raises(ValueError, match="fact-checker"):
        parse_task_graph(raw, _ROSTER)


def test_auto_assignable_agent_in_verify_field_rejects() -> None:
    raw = {"summary": "s", "steps": [{"agent": "main", "instruction": "scaffold", "mission": "scaffold it", "verify": "code-expert"}]}
    with pytest.raises(ValueError, match="code-expert"):
        parse_task_graph(raw, _ROSTER)


def test_empty_graph_rejects() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        parse_task_graph({"summary": "s", "steps": []}, _ROSTER)


def test_missing_summary_rejects() -> None:
    with pytest.raises(ValueError, match="summary"):
        parse_task_graph({"steps": [{"agent": "main", "instruction": "do it"}]}, _ROSTER)


def test_bare_array_input_rejects() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_task_graph([{"agent": "main", "instruction": "do it"}], _ROSTER)
