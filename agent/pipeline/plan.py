"""Plan data schema + validator (plan 27, improvement 1).

The plan is data, not generated code: a flat list of PlanStep carrying
verify/repair attributes. Nesting/branching is the fixed interpreter's job,
not encoded here (plan 27 decision 2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..subagents import Subagent

_STEP_REF = re.compile(r"\{\{step_(\d+)\}\}")

MAIN_AGENT = "main"


@dataclass(frozen=True)
class PlanStep:
    agent: str  # "main" or an auto-assignable subagent name
    task: str
    verify: str | None = None  # post-planning-only agent name, or a mechanical check command
    repair: str | None = None  # post-planning-only agent name, or a mechanical check command


Plan = list[PlanStep]


def parse_plan(raw: list[dict], roster: list[Subagent]) -> Plan:
    """Schema-check, roster-validate, and phase-eligibility-check a raw plan.

    Raises ValueError on any violation — the whole plan is rejected, never a
    silently dropped step (fail loud).
    """
    if not raw:
        raise ValueError("plan must contain at least one step")

    by_name = {s.name: s for s in roster}
    steps: list[PlanStep] = []
    for i, item in enumerate(raw):
        agent = item.get("agent")
        task = item.get("task")
        verify = item.get("verify")
        repair = item.get("repair")

        if agent != MAIN_AGENT and not (agent in by_name and by_name[agent].auto_assignable):
            raise ValueError(
                f"step {i}: agent {agent!r} is not {MAIN_AGENT!r} or an auto-assignable subagent"
            )
        if not task or not isinstance(task, str):
            raise ValueError(f"step {i}: task must be a non-empty string, got {task!r}")
        _check_post_planning_field(i, "verify", verify, by_name)
        _check_post_planning_field(i, "repair", repair, by_name)
        _check_refs(i, task, len(steps))

        steps.append(PlanStep(agent=agent, task=task, verify=verify, repair=repair))

    return steps


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


def _check_refs(index: int, task: str, prior_step_count: int) -> None:
    for match in _STEP_REF.finditer(task):
        ref = int(match.group(1))
        if ref < 1 or ref > prior_step_count:
            raise ValueError(
                f"step {index}: dangling ref {{{{step_{ref}}}}} — only steps 1..{prior_step_count} exist"
            )


__all__ = ["PlanStep", "Plan", "parse_plan", "MAIN_AGENT"]
