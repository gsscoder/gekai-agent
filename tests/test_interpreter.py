from __future__ import annotations

import asyncio

import pytest

from agent.harness.interpreter import TaskGraphHalted, resolve_refs, run_task_graph
from agent.harness.scaling import WorkSignal
from agent.pipeline.plan import Task, TaskGraph, parse_task_graph
from agent.subagents import SUBAGENTS
from agent.tools.delegate import ERROR_PREFIX


def run(coro):
    return asyncio.run(coro)


async def _dispatch_ok(agent: str, instruction: str, mission: str = "", signal: WorkSignal = WorkSignal(), scope: str | None = None) -> str:
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

    async def dispatch(agent: str, instruction: str, mission: str = "", signal: WorkSignal = WorkSignal(), scope: str | None = None) -> str:
        ran.append(agent)
        return f"{agent} output"

    async def always_fail(step: Task, out: str) -> bool:
        return False

    with pytest.raises(TaskGraphHalted) as exc_info:
        run(run_task_graph(graph, dispatch, verify_agent=always_fail))
    assert exc_info.value.index == 0
    assert ran == ["code-expert", "code-expert"]  # original attempt + one repair, no test-expert
    assert exc_info.value.last_output == "code-expert output"  # the failed repair attempt's own output


def test_repair_dispatch_receives_step_scope() -> None:
    graph = TaskGraph(
        summary="s",
        steps=[
            Task(agent="code-expert", instruction="write it", mission="write it", verify="fact-checker", scope="edit"),
        ],
    )
    scopes: list[str | None] = []

    async def dispatch(agent: str, instruction: str, mission: str = "", signal: WorkSignal = WorkSignal(), scope: str | None = None) -> str:
        scopes.append(scope)
        return f"{agent} output"

    async def always_fail(step: Task, out: str) -> bool:
        return False

    with pytest.raises(TaskGraphHalted):
        run(run_task_graph(graph, dispatch, verify_agent=always_fail))
    assert scopes == ["edit", "edit"]  # initial dispatch and repair dispatch both carry the step's scope


def test_empty_dispatch_output_halts() -> None:
    async def empty_dispatch(agent: str, instruction: str, mission: str = "", signal: WorkSignal = WorkSignal(), scope: str | None = None) -> str:
        return ""

    graph = TaskGraph(summary="s", steps=[Task(agent="code-expert", instruction="write it", mission="write it")])
    with pytest.raises(TaskGraphHalted, match="empty dispatch output"):
        run(run_task_graph(graph, empty_dispatch))


def test_error_sentinel_dispatch_output_halts() -> None:
    """A crashed/unresolvable dispatch (`run_subagent`'s own `ERROR_PREFIX`-
    prefixed sentinel) must halt the graph, not flow into verify/`{{step_k}}`
    substitution as if it were a legitimate output."""
    async def crashed_dispatch(agent: str, instruction: str, mission: str = "", signal: WorkSignal = WorkSignal(), scope: str | None = None) -> str:
        return f"{ERROR_PREFIX}{agent} failed: boom"

    graph = TaskGraph(
        summary="s",
        steps=[Task(agent="code-expert", instruction="write it", mission="write it", verify="mechanical")],
    )
    verify_calls: list[str] = []

    async def verify(step: Task, out: str) -> bool:
        verify_calls.append(out)
        return True  # would wrongly pass "mechanical" verify if the crash string ever reached here

    with pytest.raises(TaskGraphHalted) as exc_info:
        run(run_task_graph(graph, crashed_dispatch, verify_agent=verify))
    assert exc_info.value.index == 0
    assert verify_calls == []  # halted before ever reaching verify
    assert exc_info.value.results == []


def test_one_step_ws_explorer_graph_parses_and_produces_step_result() -> None:
    """Change 1 end to end: a pure-investigation graph is exactly one
    ws-explorer step. It parses (a lone discovery step is no longer
    rejected) and running it produces a `StepResult` whose output is what
    `_respond`'s outputs_block reads."""
    raw = {
        "summary": "investigate the banking pane",
        "steps": [{"agent": "ws-explorer", "instruction": "explore the banking pane", "mission": "explore"}],
    }
    graph = parse_task_graph(raw, SUBAGENTS)

    async def dispatch(agent: str, instruction: str, mission: str = "", signal: WorkSignal = WorkSignal(), scope: str | None = None) -> str:
        return "found the banking pane in src/panes/banking.py"

    results = run(run_task_graph(graph, dispatch))
    assert len(results) == 1
    assert results[0].output == "found the banking pane in src/panes/banking.py"

    outputs_block = "\n\n".join(f"--- step {i + 1} output ---\n{r.output}" for i, r in enumerate(results))
    assert "found the banking pane in src/panes/banking.py" in outputs_block


def test_repair_uses_named_repair_agent_not_original() -> None:
    graph = TaskGraph(
        summary="s",
        steps=[Task(agent="code-expert", instruction="write it", mission="write it", verify="fact-checker", repair="code-refactorer")],
    )
    dispatched: list[str] = []

    async def dispatch(agent: str, instruction: str, mission: str = "", signal: WorkSignal = WorkSignal(), scope: str | None = None) -> str:
        dispatched.append(agent)
        return f"{agent} output"

    calls = 0

    async def verify(step: Task, out: str) -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    run(run_task_graph(graph, dispatch, verify_agent=verify))
    assert dispatched == ["code-expert", "code-refactorer"]
