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


@pytest.mark.parametrize(
    "raw, match",
    [
        (
            {"summary": "s", "steps": [{"agent": "nonexistent-agent", "instruction": "do something"}]},
            "nonexistent-agent",
        ),
        (
            {"summary": "s", "steps": [{"agent": "main", "instruction": ""}]},
            "instruction",
        ),
        (
            {"summary": "s", "steps": [{"agent": "main", "instruction": "use {{step_5}}", "mission": "use it"}]},
            "dangling ref",
        ),
        (
            {"summary": "s", "steps": [{"agent": "fact-checker", "instruction": "review the change"}]},
            "fact-checker",
        ),
        (
            {"summary": "s", "steps": [{"agent": "main", "instruction": "scaffold", "mission": "scaffold it", "verify": "code-expert"}]},
            "code-expert",
        ),
        (
            {"summary": "s", "steps": []},
            "at least one step",
        ),
        (
            {"steps": [{"agent": "main", "instruction": "do it"}]},
            "summary",
        ),
        (
            [{"agent": "main", "instruction": "do it"}],
            "JSON object",
        ),
    ],
)
def test_unknown_agent_rejects_whole_graph(raw, match) -> None:
    with pytest.raises(ValueError, match=match):
        parse_task_graph(raw, _ROSTER)
