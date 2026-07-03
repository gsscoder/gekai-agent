"""Tests for the per-request edit-tool budget wrap: EditBudget and
_wrap_with_edit_budget in agent/harness/core.py.

Source: plan 26 ("force delegation structurally") Improvement 2 — action
budget wrap (mechanism #1). Mirrors tests/test_delegate.py's pattern of
mocking agent.harness.core.make_tools and OpenAIAdapter and calling
_build_agent directly.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from unittest.mock import patch

from agent.harness.core import EditBudget, _build_agent
from agent.llm.tools import tool
from agent.settings import Permissions
from agent.subagents import SUBAGENTS

_PERMISSIONS = Permissions(read=True, write=True, exec=True)
_WORKING_DIR = Path(".")

_EDIT_BUDGET_EXCEEDED_MSG = (
    "[budget exceeded — this task is implementation-sized; "
    "delegate the remainder to a specialist]"
)


def run(coro):
    return asyncio.run(coro)


def _make_edit_tool() -> tuple[object, list[str]]:
    """A fake edit_file tool with a call counter (via `calls`) standing in
    for the real `agent.tools.files.edit_file`, so wrapping can be asserted
    without touching the filesystem.
    """
    calls: list[str] = []

    async def edit_file(path: str = "", content: str = "") -> str:
        calls.append(path)
        return "edited"

    t = tool(edit_file, name="edit_file", required_permission="write")
    return t, calls


def _build_main_agent(budget: EditBudget | None, tools: list):
    with (
        patch("agent.harness.core.OpenAIAdapter"),
        patch("agent.harness.core.make_tools", return_value=tools),
    ):
        return _build_agent(
            "m", "k", None, {}, _WORKING_DIR, _PERMISSIONS, None, "sys", None,
            subagent=None,
            budget=budget,
        )


def test_main_agent_edit_tool_trips_budget_after_n_calls() -> None:
    """First `remaining` calls hit the real fn; the next returns the exact
    budget-exceeded string without invoking the real fn again."""
    t, calls = _make_edit_tool()
    budget = EditBudget(remaining=2)
    agent = _build_main_agent(budget, [t])

    edit = agent.tools.get("edit_file")

    assert run(edit.call(path="a.py", content="x")) == "edited"
    assert calls == ["a.py"]

    assert run(edit.call(path="b.py", content="y")) == "edited"
    assert calls == ["a.py", "b.py"]

    result = run(edit.call(path="c.py", content="z"))
    assert result == _EDIT_BUDGET_EXCEEDED_MSG
    # The real fn must not have run a third time.
    assert calls == ["a.py", "b.py"]


def test_subagent_edit_tool_never_wrapped_even_with_budget() -> None:
    """Specialists run under their own unbounded budget — the cap is main's
    alone; passing a `budget` to a subagent build must have no effect."""
    t, calls = _make_edit_tool()
    code_expert = next(s for s in SUBAGENTS if s.name == "code-expert")
    budget = EditBudget(remaining=1)

    with (
        patch("agent.harness.core.OpenAIAdapter"),
        patch("agent.harness.core.make_tools", return_value=[t]),
    ):
        agent = _build_agent(
            "m", "k", None, {}, _WORKING_DIR, _PERMISSIONS, None,
            code_expert.build_system_base(), None,
            subagent=code_expert,
            budget=budget,
        )

    edit = agent.tools.get("edit_file")
    for i in range(3):
        result = run(edit.call(path=f"{i}.py", content="x"))
        assert result == "edited"

    assert calls == ["0.py", "1.py", "2.py"]
    assert _EDIT_BUDGET_EXCEEDED_MSG not in calls


def test_edit_budget_with_infinite_remaining_never_trips() -> None:
    t, calls = _make_edit_tool()
    budget = EditBudget(remaining=math.inf)
    agent = _build_main_agent(budget, [t])

    edit = agent.tools.get("edit_file")
    for i in range(50):
        result = run(edit.call(path=f"{i}.py", content="x"))
        assert result == "edited"

    assert len(calls) == 50


def test_no_wrap_when_budget_is_none() -> None:
    """budget=None (the default) must leave EDIT_TOOLS unwrapped even on the
    main branch — the harness only wraps when a budget is explicitly passed."""
    t, calls = _make_edit_tool()
    agent = _build_main_agent(None, [t])

    edit = agent.tools.get("edit_file")
    for i in range(10):
        result = run(edit.call(path=f"{i}.py", content="x"))
        assert result == "edited"

    assert len(calls) == 10


def test_edit_budget_try_consume_decrements_and_reports_exhaustion() -> None:
    budget = EditBudget(remaining=1)
    assert budget.try_consume() is True
    assert budget.remaining == 0
    assert budget.try_consume() is False
    assert budget.remaining == 0
