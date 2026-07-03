"""Integration probe: when a stubbed `edit_file` starts returning a
budget-exceeded error result, does the real CORE model switch to `delegate`
for the remainder — or does it keep retrying `edit_file` (bang the wall)?

Source: plan 26 ("force delegation structurally") — "The one real risk"
(line ~62) and "Phased tasks" task 0, "Probe first" (line ~93). Mechanism #1
("enforced action budget") caps main's direct edit-tool calls and has the
tool itself return `[budget exceeded — this task is implementation-sized;
delegate the remainder to a specialist]` once the cap is hit. This is a
tool-result string, not an exception; the plan's named risk is that a weak
model reacts by retrying the same edit instead of reading the error and
calling `delegate`. This probe checks that BEFORE any production wiring
(no changes to agent/harness/core.py, agent/tools/catalog.py,
agent/tools/delegate.py, or agent/settings.py) — entirely self-contained
stub closures, mirroring tests/test_delegate_probe.py (plan 25's analogous
probe) but stubbing `edit_file` instead of only `delegate`.

Requires: GEKAI_CORE_MODEL_NAME, GEKAI_CORE_MODEL_KEY
Optional: GEKAI_CORE_MODEL_URL (custom base URL), GEKAI_THINKING_EFFORT

ASSUMPTIONS:
- A trial budget of 3 direct edits is small enough that the canonical
  multi-file prompt below plausibly needs more than 3 edits to finish, so
  the budget wall is reached during a normal run. This is a probe trial
  value (task 0 explicitly says "no estimate pre-pass ... hardcoded at a
  trial N"), not a tuned production threshold — plan 26 improvement 2
  (`Budget`/`EDIT_TOOLS` wrap) is what eventually derives N from a scope
  estimate.
- The model will attempt at least `_EDIT_BUDGET_N + 1` direct edits (i.e.
  will actually trip the budget) for the canonical prompt rather than
  delegating from the very first tool call; if it delegates immediately,
  the budget is never exercised and this probe's core assertion is
  unreachable (see the first assertion below, which fails loudly in that
  case with a distinct message rather than silently passing).
- asyncio.run() is safe here because no outer event loop exists in a plain
  pytest process (same pattern as tests/test_delegate_probe.py).
"""

from __future__ import annotations

import asyncio
import os

import pytest

from agent.llm.agent import Agent
from agent.llm.model_caps import resolve_thinking_params
from agent.llm.providers.openai import OpenAIAdapter
from agent.llm.tools import ToolRegistry, tool
from agent.persona import SYSTEM_PROMPT
from agent.subagents import SUBAGENTS

# Probe trial value only — not a tuned production threshold. Small on
# purpose so the canonical prompt below (a multi-file library + error
# handling + edge-case suite, well over 3 files' worth of direct edits)
# reliably trips the wall during a normal run.
_EDIT_BUDGET_N = 3

_BUDGET_EXCEEDED_MSG = (
    "[budget exceeded — this task is implementation-sized; "
    "delegate the remainder to a specialist]"
)

# Adapted from the plan's own canonical over-threshold example (plan 26
# line ~33, "the calcexpr probe": multi-file library + error handling +
# edge-case suite) — clearly more than _EDIT_BUDGET_N edits' worth of work
# if done as separate files/passes.
_CANONICAL_PROMPT = (
    "create a calcexpr Python library that parses and evaluates arithmetic "
    "expressions (+, -, *, /, parentheses, operator precedence); split it into "
    "a tokenizer module, a parser module, and an evaluator module; add custom "
    "exception types for syntax errors and division-by-zero; add a test suite "
    "covering edge cases (empty input, unbalanced parentheses, division by "
    "zero, nested parentheses)"
)


@pytest.mark.llm
def test_main_delegates_after_budget_error_instead_of_retrying_edit() -> None:
    model_name = os.environ.get("GEKAI_CORE_MODEL_NAME")
    api_key = os.environ.get("GEKAI_CORE_MODEL_KEY")
    if not model_name or not api_key:
        pytest.skip("GEKAI_CORE_MODEL_NAME and GEKAI_CORE_MODEL_KEY must be set")

    # Shared ordered log across both stub tools, tagged by tool name, so the
    # index of the first budget-exceeded edit_file result can be located and
    # compared against later delegate calls.
    calls: list[tuple[str, str]] = []
    edit_count = 0

    def edit_file(content: str) -> str:
        """Write or edit a file with the given content, as a direct implementation action.

        Args:
            content: the full text content to write.
        """
        nonlocal edit_count
        edit_count += 1
        if edit_count <= _EDIT_BUDGET_N:
            calls.append(("edit_file", "edited"))
            return "edited"
        calls.append(("edit_file", _BUDGET_EXCEEDED_MSG))
        return _BUDGET_EXCEEDED_MSG

    roster = ", ".join(s.name for s in SUBAGENTS if s.user_invocable)

    def _delegate(agent: str, task: str) -> str:
        calls.append(("delegate", agent))
        return f"[stub] {agent} completed: ok"

    # __doc__ is read by tool() via _parse_docstring(fn.__doc__) to build the
    # description sent to the model — set it after the roster is known.
    _delegate.__doc__ = (
        f"Delegates a self-contained task to a specialist subagent. "
        f"Available specialists: {roster}. "
        "Call once per specialist unit; never split one file across multiple calls."
    )

    edit_tool = tool(edit_file, name="edit_file")
    delegate_tool = tool(_delegate, name="delegate")
    registry = ToolRegistry()
    registry.register(edit_tool)
    registry.register(delegate_tool)

    base_url = os.environ.get("GEKAI_CORE_MODEL_URL")
    provider = (
        OpenAIAdapter(api_key=api_key, base_url=base_url)
        if base_url is not None
        else OpenAIAdapter(api_key=api_key)
    )

    agent = Agent(
        provider=provider,
        model=model_name,
        tools=registry,
        system=SYSTEM_PROMPT,
        max_iterations=20,
        extra_params=resolve_thinking_params(model_name, os.environ.get("GEKAI_THINKING_EFFORT")),
    )

    asyncio.run(agent.run(_CANONICAL_PROMPT))

    first_budget_idx = next(
        (i for i, (tool_name, result) in enumerate(calls)
         if tool_name == "edit_file" and result == _BUDGET_EXCEEDED_MSG),
        None,
    )
    assert first_budget_idx is not None, (
        f"the budget was never exceeded — edit_file was called at most "
        f"{edit_count} time(s) (trial N={_EDIT_BUDGET_N}); this probe requires "
        f"a prompt that plausibly needs more than N direct edits, so either "
        f"the model delegated before ever hitting the wall (in which case the "
        f"real go/no-go signal is moot — main never needed the budget) or the "
        f"prompt undershoots N; recorded calls: {calls}"
    )

    delegated_after_budget_error = any(
        tool_name == "delegate" for tool_name, _ in calls[first_budget_idx + 1:]
    )
    assert delegated_after_budget_error, (
        "MODEL BANGED THE BUDGET WALL: after the budget-exceeded edit_file "
        f"result (call #{first_budget_idx}), main kept calling edit_file "
        "instead of switching to `delegate` for the remainder — this is the "
        "risk named in plan 26 'The one real risk' (line ~62): mechanism #1 "
        "degrades into a wall main bangs against, so it should not be locked "
        "in without mechanism #2's schema asymmetry as a complement; "
        f"full call sequence: {calls}"
    )
