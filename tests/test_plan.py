from __future__ import annotations

import pytest

from agent.pipeline.plan import Task, parse_task_graph
from agent.subagents import Subagent

_ROSTER = [
    Subagent(name="code-expert", namespace="coding", description="d", auto_assignable=True),
    Subagent(name="test-expert", namespace="testing", description="d", auto_assignable=True),
    Subagent(name="fact-checker", namespace="testing", description="d", auto_assignable=False),
    Subagent(name="ws-explorer", namespace="generic", description="d", auto_assignable=True, discovery_stage=True),
]


def test_well_formed_graph_parses() -> None:
    raw = {
        "summary": "build a library with tests",
        "steps": [
            {"agent": "code-expert", "instruction": "scaffold repo", "mission": "scaffold the repo"},
            {"agent": "code-expert", "instruction": "write code", "mission": "write the code", "verify": "fact-checker"},
            {"agent": "test-expert", "instruction": "write tests for {{step_2}}", "mission": "write tests"},
        ],
    }
    graph = parse_task_graph(raw, _ROSTER)
    assert graph.summary == "build a library with tests"
    assert graph == [
        Task(agent="code-expert", instruction="scaffold repo", mission="scaffold the repo"),
        Task(agent="code-expert", instruction="write code", mission="write the code", verify="fact-checker"),
        Task(agent="test-expert", instruction="write tests for {{step_2}}", mission="write tests"),
    ]


def test_root_agent_is_rejected_as_step_agent() -> None:
    """Phase 3: root leaves the graph — it is never a valid step `agent`,
    even though it is still an auto-assignable-looking string. Every step
    must go to an auto-assignable specialist."""
    raw = {
        "summary": "s",
        "steps": [{"agent": "root", "instruction": "scaffold repo", "mission": "scaffold the repo"}],
    }
    with pytest.raises(ValueError, match="root"):
        parse_task_graph(raw, _ROSTER)


@pytest.mark.parametrize(
    "raw, match",
    [
        (
            {"summary": "s", "steps": [{"agent": "nonexistent-agent", "instruction": "do something"}]},
            "nonexistent-agent",
        ),
        (
            {"summary": "s", "steps": [{"agent": "code-expert", "instruction": ""}]},
            "instruction",
        ),
        (
            {"summary": "s", "steps": [{"agent": "code-expert", "instruction": "use {{step_5}}", "mission": "use it"}]},
            "dangling ref",
        ),
        (
            {"summary": "s", "steps": [{"agent": "fact-checker", "instruction": "review the change"}]},
            "fact-checker",
        ),
        (
            {"summary": "s", "steps": [{"agent": "code-expert", "instruction": "scaffold", "mission": "scaffold it", "verify": "code-expert"}]},
            "code-expert",
        ),
        (
            {"summary": "s", "steps": []},
            "at least one step",
        ),
        (
            {"steps": [{"agent": "code-expert", "instruction": "do it"}]},
            "summary",
        ),
        (
            [{"agent": "code-expert", "instruction": "do it"}],
            "JSON object",
        ),
    ],
)
def test_unknown_agent_rejects_whole_graph(raw, match) -> None:
    with pytest.raises(ValueError, match=match):
        parse_task_graph(raw, _ROSTER)


@pytest.mark.parametrize("scope", ["read", "edit", "fs"])
def test_scope_field_parses_into_plain_string(scope) -> None:
    raw = {
        "summary": "edit a file with narrow tools",
        "steps": [
            {
                "agent": "code-expert",
                "instruction": "edit config",
                "mission": "edit the config file",
                "scope": scope,
            }
        ],
    }
    graph = parse_task_graph(raw, _ROSTER)
    assert graph == [
        Task(
            agent="code-expert",
            instruction="edit config",
            mission="edit the config file",
            scope=scope,
        ),
    ]


def test_scope_field_omitted_defaults_to_none() -> None:
    raw = {
        "summary": "s",
        "steps": [{"agent": "code-expert", "instruction": "do it", "mission": "do it"}],
    }
    graph = parse_task_graph(raw, _ROSTER)
    assert graph[0].scope is None


@pytest.mark.parametrize(
    "scope, match",
    [
        ("bogus", "scope must be"),
        (123, "scope must be"),
        ([], "scope must be"),
        ({}, "scope must be"),
        ("mechanical", "scope must be"),
    ],
)
def test_malformed_scope_rejects_whole_graph(scope, match) -> None:
    raw = {
        "summary": "s",
        "steps": [{"agent": "code-expert", "instruction": "do it", "mission": "do it", "scope": scope}],
    }
    with pytest.raises(ValueError, match=match):
        parse_task_graph(raw, _ROSTER)


# ---------------------------------------------------------------------------
# discovery-stage steps (e.g. ws-explorer) — a precursor whose report a
# *later* step consumes via {{step_k}}, never a deliverable on its own
# ---------------------------------------------------------------------------

def test_discovery_stage_accepted_as_only_step() -> None:
    """A pure-investigation turn: the discovery step's report reaches the
    user via `_respond`'s outputs_block directly — no later step is needed
    to consume it."""
    raw = {
        "summary": "s",
        "steps": [{"agent": "ws-explorer", "instruction": "explore the codebase", "mission": "explore"}],
    }
    graph = parse_task_graph(raw, _ROSTER)
    assert graph == [Task(agent="ws-explorer", instruction="explore the codebase", mission="explore")]


def test_discovery_stage_accepted_as_last_step() -> None:
    raw = {
        "summary": "s",
        "steps": [
            {"agent": "code-expert", "instruction": "scaffold repo", "mission": "scaffold the repo"},
            {"agent": "ws-explorer", "instruction": "explore the codebase", "mission": "explore"},
        ],
    }
    graph = parse_task_graph(raw, _ROSTER)
    assert graph == [
        Task(agent="code-expert", instruction="scaffold repo", mission="scaffold the repo"),
        Task(agent="ws-explorer", instruction="explore the codebase", mission="explore"),
    ]


def test_two_discovery_stages_rejected() -> None:
    raw = {
        "summary": "s",
        "steps": [
            {"agent": "ws-explorer", "instruction": "explore area A", "mission": "explore A"},
            {"agent": "ws-explorer", "instruction": "explore area B", "mission": "explore B"},
            {"agent": "code-expert", "instruction": "use {{step_1}} and {{step_2}}", "mission": "implement"},
        ],
    }
    with pytest.raises(ValueError, match="discovery-stage steps"):
        parse_task_graph(raw, _ROSTER)


def test_well_formed_discovery_stage_graph_accepted() -> None:
    raw = {
        "summary": "investigate then implement",
        "steps": [
            {"agent": "ws-explorer", "instruction": "explore the codebase", "mission": "explore"},
            {"agent": "code-expert", "instruction": "implement using {{step_1}}", "mission": "implement"},
        ],
    }
    graph = parse_task_graph(raw, _ROSTER)
    assert graph == [
        Task(agent="ws-explorer", instruction="explore the codebase", mission="explore"),
        Task(agent="code-expert", instruction="implement using {{step_1}}", mission="implement"),
    ]
