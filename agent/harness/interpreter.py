"""Fixed step-runner interpreter (plan 27, improvement 2; renamed under plan 28).

Written once, engineered control flow: execute -> verify -> repair ->
re-verify -> halt. Knows no agent by name or role — the planner populates
`agent`/`verify`/`repair`; this module only walks the task graph. Runs "as
main": a `main` step is direct main-processing (handled by whatever
`dispatch` does for that name), a subagent step is a spawn `dispatch` owns.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..pipeline.plan import Task, TaskGraph
from .scaling import WorkSignal, node_signal

DispatchFn = Callable[[str, str, str, WorkSignal], Awaitable[str]]
VerifyFn = Callable[[Task, str], Awaitable[bool]]
# Structured pass/fail is a placeholder here (hard problem 3, open point 2 —
# the verdict contract is not yet designed); a bool is enough to drive the
# fixed repair/halt policy below.


@dataclass(slots=True)
class StepResult:
    step: Task
    output: str


class TaskGraphHalted(Exception):
    def __init__(self, index: int, step: Task, reason: str) -> None:
        self.index = index
        self.step = step
        self.reason = reason
        super().__init__(f"step {index} ({step.agent}) halted: {reason}")


def resolve_refs(instruction: str, prior_outputs: list[str]) -> str:
    """Template-substitute {{step_k}} (1-based) with the k-th prior step's output."""
    for k, output in enumerate(prior_outputs, start=1):
        instruction = instruction.replace(f"{{{{step_{k}}}}}", output)
    return instruction


def _repair_instruction(step: Task, out: str) -> str:
    return f"{step.instruction}\n\nthe previous attempt failed verification. its output was:\n{out}"


_SIBLING_TASK_LIMIT = 60


def _truncate(text: str, limit: int = _SIBLING_TASK_LIMIT) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _inject_request_summary(graph: TaskGraph, index: int, instruction: str) -> str:
    """Prepend a mechanical <request_summary> framing block to a step's instruction:
    the overall ask (planner-produced `summary`) plus this step's own
    boundary against sibling steps — so a subagent that runs cold, with no
    awareness of the graph around it, doesn't redo or collide with work
    another step already owns (e.g. writing its own throwaway tests when a
    dedicated test step exists)."""
    total = len(graph)
    lines = [
        "<request_summary>",
        f"user wants: {graph.summary}",
        f"your step ({index + 1}/{total}): {graph[index].instruction}",
    ]
    if total > 1:
        others = ", ".join(
            f"step {i + 1} {s.agent} ({_truncate(s.instruction)})"
            for i, s in enumerate(graph)
            if i != index
        )
        lines.append(f"handled elsewhere — do not do: {others}")
    lines.append("</request_summary>")
    return "\n".join(lines) + "\n\n" + instruction


async def run_task_graph(
    graph: TaskGraph,
    dispatch: DispatchFn,
    verify_agent: VerifyFn | None = None,
) -> list[StepResult]:
    """Walk `graph` with the fixed execute->verify->repair->re-verify->halt policy.

    Raises TaskGraphHalted on an empty dispatch output or a second verify
    failure — completed steps' results are not rolled back (halt-and-report,
    no resume).
    """
    prior_outputs: list[str] = []
    results: list[StepResult] = []

    for index, step in enumerate(graph):
        instruction = resolve_refs(step.instruction, prior_outputs)
        instruction = _inject_request_summary(graph, index, instruction)
        out = await dispatch(step.agent, instruction, step.mission, node_signal(step))
        if not out:
            raise TaskGraphHalted(index, step, "empty dispatch output")

        if step.verify and verify_agent is not None:
            if not await verify_agent(step, out):
                # Repair dispatch composes the retry bump on top of the
                # node's own reasoning-shaped signal — not a bare
                # WorkSignal(retry=1) replacing it (plan 28 Phase 2).
                base = node_signal(step)
                retry_signal = WorkSignal(direction=base.direction, retry=base.retry + 1)
                out = await dispatch(step.repair or step.agent, _repair_instruction(step, out), step.mission, retry_signal)
                if not await verify_agent(step, out):
                    raise TaskGraphHalted(index, step, "failed verification twice")

        prior_outputs.append(out)
        results.append(StepResult(step=step, output=out))

    return results


__all__ = ["StepResult", "TaskGraphHalted", "resolve_refs", "run_task_graph"]
