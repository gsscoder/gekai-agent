from __future__ import annotations

import asyncio

import pytest

from agent.harness.interpreter import TaskGraphHalted, resolve_refs, run_task_graph
from agent.harness.scaling import WorkSignal
from agent.pipeline.plan import Task, TaskGraph


def run(coro):
    return asyncio.run(coro)


async def _dispatch_ok(agent: str, instruction: str, mission: str = "", signal: WorkSignal = WorkSignal()) -> str:
    return f"{agent} did: {instruction}"


@pytest.mark.parametrize(
    "text,prior,expected",
    [
        ("build on {{step_1}}", ["tokenizer done"], "build on tokenizer done"),
        ("build on {{step_2}}", ["only one"], "build on {{step_2}}"),
    ],
)
def test_resolve_refs(text: str, prior: list[str], expected: str) -> None:
    assert resolve_refs(text, prior) == expected


def test_passing_graph_runs_all_steps_with_refs_substituted() -> None:
    graph = TaskGraph(
        summary="build a tokenizer with tests",
        steps=[
            Task(agent="code-expert", instruction="write tokenizer", mission="write tokenizer"),
            Task(agent="test-expert", instruction="test {{step_1}}", mission="test tokenizer"),
        ],
    )
    results = run(run_task_graph(graph, _dispatch_ok))
    assert results[0].output.startswith("code-expert did:")
    assert results[0].output.endswith("write tokenizer")
    assert results[1].output.startswith("test-expert did:")
    # the {{step_1}} ref is resolved to step 1's dispatch output before the
    # request_summary framing block is prepended around it
    assert "test code-expert did:" in results[1].output


def test_step_failing_verify_once_is_repaired_then_passes() -> None:
    graph = TaskGraph(summary="s", steps=[Task(agent="code-expert", instruction="write it", mission="write it", verify="fact-checker")])
    verify_calls: list[str] = []

    async def verify(step: Task, out: str) -> bool:
        verify_calls.append(out)
        return len(verify_calls) > 1  # fail first call, pass second

    results = run(run_task_graph(graph, _dispatch_ok, verify_agent=verify))
    assert len(results) == 1
    assert len(verify_calls) == 2


def test_step_failing_verify_twice_halts_and_later_steps_do_not_run() -> None:
    graph = TaskGraph(
        summary="s",
        steps=[
            Task(agent="code-expert", instruction="write it", mission="write it", verify="fact-checker"),
            Task(agent="test-expert", instruction="test it", mission="test it"),
        ],
    )
    ran: list[str] = []

    async def dispatch(agent: str, instruction: str, mission: str = "", signal: WorkSignal = WorkSignal()) -> str:
        ran.append(agent)
        return f"{agent} output"

    async def always_fail(step: Task, out: str) -> bool:
        return False

    with pytest.raises(TaskGraphHalted) as exc_info:
        run(run_task_graph(graph, dispatch, verify_agent=always_fail))
    assert exc_info.value.index == 0
    assert ran == ["code-expert", "code-expert"]  # original attempt + one repair, no test-expert


def test_empty_dispatch_output_halts() -> None:
    async def empty_dispatch(agent: str, instruction: str, mission: str = "", signal: WorkSignal = WorkSignal()) -> str:
        return ""

    graph = TaskGraph(summary="s", steps=[Task(agent="code-expert", instruction="write it", mission="write it")])
    with pytest.raises(TaskGraphHalted, match="empty dispatch output"):
        run(run_task_graph(graph, empty_dispatch))


def test_repair_uses_named_repair_agent_not_original() -> None:
    graph = TaskGraph(
        summary="s",
        steps=[Task(agent="code-expert", instruction="write it", mission="write it", verify="fact-checker", repair="code-refactorer")],
    )
    dispatched: list[str] = []

    async def dispatch(agent: str, instruction: str, mission: str = "", signal: WorkSignal = WorkSignal()) -> str:
        dispatched.append(agent)
        return f"{agent} output"

    calls = 0

    async def verify(step: Task, out: str) -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    run(run_task_graph(graph, dispatch, verify_agent=verify))
    assert dispatched == ["code-expert", "code-refactorer"]
