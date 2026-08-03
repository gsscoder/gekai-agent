"""TaskGraph data schema + validator (plan 27 origin; renamed under plan 28).

v1 is data, not generated code: a flat list of Task nodes carrying
verify/repair attributes, edges implicit-sequential. Conditional branching
is a deliberately deferred capability (plan 28) — until built, nesting/
branching stays the fixed interpreter's job, not encoded here (plan 27
decision 2, preserved).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..subagents import Subagent
from ..text_format import clean_output

_STEP_REF = re.compile(r"\{\{step_(\d+)\}\}")

ROOT_AGENT = "root"


@dataclass(frozen=True)
class Task:
    agent: str  # an auto-assignable subagent name (root is never a step agent)
    instruction: str
    mission: str  # short human-readable phrase (~8-10 words) describing the step's job
    verify: str | None = None  # post-planning-only agent name, or a mechanical check command
    repair: str | None = None  # post-planning-only agent name, or a mechanical check command
    scope: str | None = None  # sequencer's per-step tool-breadth classification: "read"/"edit"/"fs", or None if the sequencer emitted no signal (assignment-time tool scoping); inert until a later phase wires it into dispatch


@dataclass(frozen=True, eq=False)
class TaskGraph:
    """Validated task graph: the model's own gist of the request (`summary`) plus the
    ordered steps. List-like (`len`, iteration, indexing) so existing callers
    that treat a graph as a sequence of Task keep working unchanged."""

    summary: str
    steps: list[Task]

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __getitem__(self, index):
        return self.steps[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TaskGraph):
            return self.summary == other.summary and self.steps == other.steps
        return self.steps == other  # allow comparing against a bare list of steps in tests


def parse_task_graph(raw: dict, roster: list[Subagent]) -> TaskGraph:
    """Schema-check, roster-validate, and phase-eligibility-check a raw task graph.

    `raw` is the model's parsed JSON object: {"summary": str, "steps": [...]}.
    Raises ValueError on any violation — the whole graph is rejected, never a
    silently dropped step (fail loud).
    """
    if not isinstance(raw, dict):
        raise ValueError(f"task graph must be a JSON object with 'summary' and 'steps', got {raw!r}")

    summary = raw.get("summary")
    if not summary or not isinstance(summary, str):
        raise ValueError(f"task graph must contain a non-empty 'summary' string, got {summary!r}")
    summary = clean_output(summary)

    step_list = raw.get("steps")
    if not isinstance(step_list, list) or not step_list:
        raise ValueError("task graph must contain at least one step")

    by_name = {s.name: s for s in roster}
    steps: list[Task] = []
    for i, item in enumerate(step_list):
        agent = item.get("agent")
        instruction = item.get("instruction")
        mission = item.get("mission")
        verify = item.get("verify")
        repair = item.get("repair")
        scope = item.get("scope")

        if not (agent in by_name and by_name[agent].auto_assignable):
            raise ValueError(f"step {i}: agent {agent!r} is not an auto-assignable subagent")
        if not instruction or not isinstance(instruction, str):
            raise ValueError(f"step {i}: instruction must be a non-empty string, got {instruction!r}")
        if not mission or not isinstance(mission, str):
            raise ValueError(f"step {i}: mission must be a non-empty string, got {mission!r}")
        _check_post_planning_field(i, "verify", verify, by_name)
        _check_post_planning_field(i, "repair", repair, by_name)
        _check_scope_field(i, scope)
        _check_refs(i, instruction, len(steps))

        steps.append(
            Task(agent=agent, instruction=instruction, mission=mission, verify=verify, repair=repair, scope=scope)
        )

    return TaskGraph(summary=summary, steps=steps)


def _check_post_planning_field(
    index: int, field: str, value: str | None, by_name: dict[str, Subagent]
) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"step {index}: {field} must be a non-empty string or None, got {value!r}")
    if value in by_name and by_name[value].auto_assignable:
        raise ValueError(
            f"step {index}: {field}={value!r} is an auto-assignable agent, "
            "not a post-planning-only verify/repair agent"
        )


def _check_scope_field(index: int, value: str | None) -> None:
    if value is None:
        return
    if value not in ("read", "edit", "fs"):
        raise ValueError(f"step {index}: scope must be 'read', 'edit', 'fs', or None, got {value!r}")


def _check_refs(index: int, instruction: str, prior_step_count: int) -> None:
    for match in _STEP_REF.finditer(instruction):
        ref = int(match.group(1))
        if ref < 1 or ref > prior_step_count:
            raise ValueError(
                f"step {index}: dangling ref {{{{step_{ref}}}}} — only steps 1..{prior_step_count} exist"
            )


__all__ = ["Task", "TaskGraph", "parse_task_graph", "ROOT_AGENT"]
