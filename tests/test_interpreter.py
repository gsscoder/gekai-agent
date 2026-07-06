from __future__ import annotations

import asyncio

import pytest

from agent.harness.interpreter import PlanHalted, resolve_refs, run_plan
from agent.pipeline.plan import PlanStep


def run(coro):
    return asyncio.run(coro)


async def _dispatch_ok(agent: str, task: str) -> str:
    return f"{agent} did: {task}"


def test_resolve_refs_substitutes_prior_outputs() -> None:
    assert resolve_refs("build on {{step_1}}", ["tokenizer done"]) == "build on tokenizer done"


def test_resolve_refs_leaves_dangling_ref_untouched() -> None:
    assert resolve_refs("build on {{step_2}}", ["only one"]) == "build on {{step_2}}"


def test_passing_plan_runs_all_steps_with_refs_substituted() -> None:
    plan = [
        PlanStep(agent="code-expert", task="write tokenizer"),
        PlanStep(agent="test-expert", task="test {{step_1}}"),
    ]
    results = run(run_plan(plan, _dispatch_ok))
    assert [r.output for r in results] == [
        "code-expert did: write tokenizer",
        "test-expert did: test code-expert did: write tokenizer",
    ]


def test_step_failing_verify_once_is_repaired_then_passes() -> None:
    plan = [PlanStep(agent="code-expert", task="write it", verify="fact-checker")]
    verify_calls: list[str] = []

    async def verify(step: PlanStep, out: str) -> bool:
        verify_calls.append(out)
        return len(verify_calls) > 1  # fail first call, pass second

    results = run(run_plan(plan, _dispatch_ok, verify_agent=verify))
    assert len(results) == 1
    assert len(verify_calls) == 2


def test_step_failing_verify_twice_halts_and_later_steps_do_not_run() -> None:
    plan = [
        PlanStep(agent="code-expert", task="write it", verify="fact-checker"),
        PlanStep(agent="test-expert", task="test it"),
    ]
    ran: list[str] = []

    async def dispatch(agent: str, task: str) -> str:
        ran.append(agent)
        return f"{agent} output"

    async def always_fail(step: PlanStep, out: str) -> bool:
        return False

    with pytest.raises(PlanHalted) as exc_info:
        run(run_plan(plan, dispatch, verify_agent=always_fail))
    assert exc_info.value.index == 0
    assert ran == ["code-expert", "code-expert"]  # original attempt + one repair, no test-expert


def test_empty_dispatch_output_halts() -> None:
    async def empty_dispatch(agent: str, task: str) -> str:
        return ""

    plan = [PlanStep(agent="code-expert", task="write it")]
    with pytest.raises(PlanHalted, match="empty dispatch output"):
        run(run_plan(plan, empty_dispatch))


def test_repair_uses_named_repair_agent_not_original() -> None:
    plan = [PlanStep(agent="code-expert", task="write it", verify="fact-checker", repair="code-refactorer")]
    dispatched: list[str] = []

    async def dispatch(agent: str, task: str) -> str:
        dispatched.append(agent)
        return f"{agent} output"

    calls = 0

    async def verify(step: PlanStep, out: str) -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    run(run_plan(plan, dispatch, verify_agent=verify))
    assert dispatched == ["code-expert", "code-refactorer"]
