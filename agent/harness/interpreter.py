"""Fixed step-runner interpreter (plan 27, improvement 2; renamed under plan 28).

Written once, engineered control flow: execute -> verify -> repair ->
re-verify -> halt. Knows no agent by name or role — the sequencer populates
`agent`/`verify`; this module only walks the task graph, always redispatching
a failed step to its own `agent` for repair, never a separate named repair
agent. Every step is a cold, fire-and-forget spawn `dispatch` owns — root is
never a step agent (plan 32 Phase 3); it owns the graph's execution, not a
place in it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..pipeline.plan import Task, TaskGraph
from ..pipeline.verifier import Verdict
from .dispatch import ERROR_PREFIX
from .scaling import WorkSignal, node_signal

DispatchFn = Callable[[str, str, str, WorkSignal, str | None], Awaitable[str]]
VerifyFn = Callable[[Task, str, int, int], Awaitable[Verdict]]
# (step, step_output, attempt, index) -> Verdict. attempt is 0 for the first
# check, 1 for the re-check after one repair dispatch — the caller (core.py)
# uses this to decide whether to scale the verifier call toward CORE. index
# is the step's real position in the graph (0-based) — the caller uses this
# as VerifyEvent.step_index, correlating against TaskGraphStartedEvent.steps;
# it is NOT a count of verify calls made, which would desync from the real
# position whenever an earlier step's verification was gate-skipped or a
# repair triggered a second call for the same step.


@dataclass(slots=True)
class StepResult:
    step: Task
    output: str


class TaskGraphHalted(Exception):
    def __init__(
        self, index: int, step: Task, reason: str, results: list[StepResult] | None = None,
        last_output: str = "",
    ) -> None:
        self.index = index
        self.step = step
        self.reason = reason
        self.results = results if results is not None else []
        self.last_output = last_output
        super().__init__(f"step {index} ({step.agent}) halted: {reason}")


def resolve_refs(instruction: str, prior_outputs: list[str]) -> str:
    """Template-substitute {{step_k}} (1-based) with the k-th prior step's output."""
    for k, output in enumerate(prior_outputs, start=1):
        instruction = instruction.replace(f"{{{{step_{k}}}}}", output)
    return instruction


def _repair_instruction(step: Task, out: str, violations: list[str]) -> str:
    violations_text = "\n".join(f"- {v}" for v in violations) if violations else "(no specific violations reported)"
    return (
        f"{step.instruction}\n\nthe previous attempt failed verification with these specific problems:\n"
        f"{violations_text}\n\nthe previous attempt's output was:\n{out}\n\nfix the problems above."
    )


_SIBLING_TASK_LIMIT = 60


def _truncate(text: str, limit: int = _SIBLING_TASK_LIMIT) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _inject_request_summary(graph: TaskGraph, index: int, instruction: str, user_input: str) -> str:
    """Prepend a mechanical <request_summary> framing block to a step's instruction:
    the overall ask (sequencer-produced `summary`) plus this step's own
    boundary against sibling steps — so a subagent that runs cold, with no
    awareness of the graph around it, doesn't redo or collide with work
    another step already owns (e.g. writing its own throwaway tests when a
    dedicated test step exists)."""
    total = len(graph)
    lines = [
        "<request_summary>",
        f"the user's original request, verbatim: {user_input}",
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
    user_input: str,
    verify_agent: VerifyFn | None = None,
) -> list[StepResult]:
    """Walk `graph` with the fixed execute->verify->repair->re-verify->halt policy.

    `verify_agent`, when given, is called unconditionally for every step —
    gating on whether a step's dispatch actually produced a coding diff worth
    checking is the caller's job (core.py's `verify_agent` closure), not this
    module's. Raises TaskGraphHalted on an empty dispatch output, a dispatch
    output that is a crashed-run error sentinel (`ERROR_PREFIX`), or a second
    verify failure — completed steps' results are not rolled back
    (halt-and-report, no resume) and are carried on the exception's own
    `results` so the caller can still report what finished.
    """
    prior_outputs: list[str] = []
    results: list[StepResult] = []

    for index, step in enumerate(graph):
        instruction = resolve_refs(step.instruction, prior_outputs)
        instruction = _inject_request_summary(graph, index, instruction, user_input)
        out = await dispatch(step.agent, instruction, step.mission, node_signal(step), step.scope)
        if not out:
            raise TaskGraphHalted(index, step, "empty dispatch output", results)
        if out.startswith(ERROR_PREFIX):
            raise TaskGraphHalted(index, step, out, results)

        if verify_agent is not None:
            verdict = await verify_agent(step, out, 0, index)
            if not verdict.ok:
                # Repair dispatch composes the retry bump on top of the
                # node's own reasoning-shaped signal — not a bare
                # WorkSignal(retry=1) replacing it (plan 28 Phase 2).
                base = node_signal(step)
                retry_signal = WorkSignal(direction=base.direction, retry=base.retry + 1)
                out = await dispatch(step.agent, _repair_instruction(step, out, verdict.violations), step.mission, retry_signal, step.scope)
                verdict = await verify_agent(step, out, 1, index)
                if not verdict.ok:
                    reason = "failed verification twice"
                    if verdict.violations:
                        reason += ": " + "; ".join(verdict.violations)
                    raise TaskGraphHalted(index, step, reason, results, last_output=out)

        prior_outputs.append(out)
        results.append(StepResult(step=step, output=out))

    return results


__all__ = ["StepResult", "TaskGraphHalted", "resolve_refs", "run_task_graph"]
