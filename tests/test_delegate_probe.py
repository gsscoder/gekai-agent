"""Integration probe: does the real CORE model autonomously call `delegate`
and honour dependency order (code-expert before test-expert)?

Source: plan 25 ("dissolve the planner, agents become tools") — this test is
the go/no-go gate before the irreversible deletion of the planner.

Requires: GEKAI_CORE_MODEL_NAME, GEKAI_CORE_MODEL_KEY
Optional: GEKAI_CORE_MODEL_URL (custom base URL), GEKAI_THINKING_EFFORT

ASSUMPTIONS:
- The model will call `delegate` at least twice for the canonical multi-build
  prompt rather than doing all work inline.
- "code-expert" and "test-expert" are stable subagent names (asserted via
  SUBAGENTS at import time implicitly through the roster string).
- asyncio.run() is safe here because no outer event loop exists in a plain
  pytest process (same pattern as tests/test_llm_agent.py).
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

_DELEGATION_DIRECTIVE = (
    "do coherent work yourself; for a unit that fits a specialist, call `delegate` with a "
    "self-contained task you write; never fragment one artifact (a file, a module) across "
    "delegates; order by dependency (scaffold → logic → tests) regardless of prompt order"
)

_CANONICAL_PROMPT = (
    "create a normalization Python module that cleans text from XML/HTML/MD; "
    "use requirements for deps and add minimal coverage"
)


@pytest.mark.llm
def test_main_delegates_code_then_test_expert_in_order() -> None:
    model_name = os.environ.get("GEKAI_CORE_MODEL_NAME")
    api_key = os.environ.get("GEKAI_CORE_MODEL_KEY")
    if not model_name or not api_key:
        pytest.skip("GEKAI_CORE_MODEL_NAME and GEKAI_CORE_MODEL_KEY must be set")

    calls: list[tuple[str, str]] = []

    roster = ", ".join(s.name for s in SUBAGENTS if s.user_invocable)

    def _delegate(agent: str, task: str) -> str:
        calls.append((agent, task))
        return f"[stub] {agent} completed: ok"

    # __doc__ is read by tool() via _parse_docstring(fn.__doc__) to build the
    # description sent to the model — set it after the roster is known.
    _delegate.__doc__ = (
        f"Delegates a self-contained task to a specialist subagent. "
        f"Available specialists: {roster}. "
        "Call once per specialist unit; never split one file across multiple calls."
    )

    delegate_tool = tool(_delegate, name="delegate")
    registry = ToolRegistry()
    registry.register(delegate_tool)

    system_prompt = SYSTEM_PROMPT + "\n<directives>\n" + _DELEGATION_DIRECTIVE

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
        system=system_prompt,
        max_iterations=10,
        extra_params=resolve_thinking_params(model_name, os.environ.get("GEKAI_THINKING_EFFORT")),
    )

    asyncio.run(agent.run(_CANONICAL_PROMPT))

    assert len(calls) >= 2, (
        f"expected at least 2 delegate calls, got {len(calls)}: {calls}"
    )

    code_expert_first = next(
        (i for i, (name, _) in enumerate(calls) if name == "code-expert"), None
    )
    test_expert_first = next(
        (i for i, (name, _) in enumerate(calls) if name == "test-expert"), None
    )

    assert code_expert_first is not None, (
        f"no delegate call to 'code-expert' found; recorded calls: {calls}"
    )
    assert test_expert_first is not None, (
        f"no delegate call to 'test-expert' found; recorded calls: {calls}"
    )
    assert code_expert_first < test_expert_first, (
        f"'code-expert' must be delegated before 'test-expert' (dependency order: "
        f"scaffold/logic first, tests last), but code-expert was call #{code_expert_first} "
        f"and test-expert was call #{test_expert_first}; full sequence: {calls}"
    )
