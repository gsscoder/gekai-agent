"""Fixed step-runner interpreter (plan 27, improvement 2).

Written once, engineered control flow: execute -> verify -> repair ->
re-verify -> halt. Knows no agent by name or role — the planner populates
`agent`/`verify`/`repair`; this module only walks the plan. Runs "as main":
a `main` step is direct main-processing (handled by whatever `dispatch`
does for that name), a subagent step is a spawn `dispatch` owns.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..pipeline.plan import Plan, PlanStep

DispatchFn = Callable[[str, str], Awaitable[str]]
VerifyFn = Callable[[PlanStep, str], Awaitable[bool]]
# Structured pass/fail is a placeholder here (hard problem 3, open point 2 —
# the verdict contract is not yet designed); a bool is enough to drive the
# fixed repair/halt policy below.
OnEvent = Callable[[str, int, PlanStep], None]


@dataclass(slots=True)
class StepResult:
    step: PlanStep
    output: str


class PlanHalted(Exception):
    def __init__(self, index: int, step: PlanStep, reason: str) -> None:
        self.index = index
        self.step = step
        self.reason = reason
        super().__init__(f"step {index} ({step.agent}) halted: {reason}")


def resolve_refs(task: str, prior_outputs: list[str]) -> str:
    """Template-substitute {{step_k}} (1-based) with the k-th prior step's output."""
    for k, output in enumerate(prior_outputs, start=1):
        task = task.replace(f"{{{{step_{k}}}}}", output)
    return task


def _repair_task(step: PlanStep, out: str) -> str:
    return f"{step.task}\n\nthe previous attempt failed verification. its output was:\n{out}"


async def run_plan(
    plan: Plan,
    dispatch: DispatchFn,
    verify_agent: VerifyFn | None = None,
    on_event: OnEvent | None = None,
) -> list[StepResult]:
    """Walk `plan` with the fixed execute->verify->repair->re-verify->halt policy.

    Raises PlanHalted on an empty dispatch output or a second verify failure —
    completed steps' results are not rolled back (halt-and-report, no resume).
    """
    prior_outputs: list[str] = []
    results: list[StepResult] = []

    for index, step in enumerate(plan):
        if on_event:
            on_event("start", index, step)
        task = resolve_refs(step.task, prior_outputs)
        out = await dispatch(step.agent, task)
        if not out:
            if on_event:
                on_event("halt", index, step)
            raise PlanHalted(index, step, "empty dispatch output")

        if step.verify and verify_agent is not None:
            if on_event:
                on_event("verify", index, step)
            if not await verify_agent(step, out):
                if on_event:
                    on_event("repair", index, step)
                out = await dispatch(step.repair or step.agent, _repair_task(step, out))
                if on_event:
                    on_event("verify", index, step)
                if not await verify_agent(step, out):
                    if on_event:
                        on_event("halt", index, step)
                    raise PlanHalted(index, step, "failed verification twice")

        if on_event:
            on_event("done", index, step)
        prior_outputs.append(out)
        results.append(StepResult(step=step, output=out))

    return results


__all__ = ["StepResult", "PlanHalted", "resolve_refs", "run_plan"]
