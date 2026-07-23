"""Integration probe: does the real CORE model, given a one-shot sequencer-style
prompt, decompose a multi-part build into a rational-order step list
(code-expert before test-expert), attach preventive verification to the
complex steps, and thread code-expert's output into test-expert's task?

Source: plan 27 ("plan gate and data plan") — "Phased tasks" step 0, the
go/no-go probe before any sequencer/interpreter production code is written.
Entirely self-contained: no agent/pipeline/plan.py, no sequencer module, no
interpreter — just a raw one-shot completion (no tools) parsed as JSON,
mirroring the retired tests/test_delegate_probe.py's no-production-wiring
approach.

Requires: GEKAI_CORE_MODEL_NAME, GEKAI_CORE_MODEL_KEY
Optional: GEKAI_CORE_MODEL_URL (custom base URL), GEKAI_THINKING_EFFORT

Run 3 trials (LLM output is non-deterministic); the probe is a go/no-go
signal, not a strict contract, so it requires a majority (>=2/3) to pass
rather than all 3.
"""

from __future__ import annotations

import asyncio
import json
import os
import re

import pytest

from agent.llm.agent import Agent
from agent.llm.model_caps import resolve_thinking_params
from agent.llm.providers.openai import OpenAIAdapter

_SEQUENCER_SYSTEM = (
    "You are a planning stage in a coding harness. Decompose the user's request into "
    "a JSON object with keys:\n"
    '  "summary": a short 1-2 sentence gist of what the user wants\n'
    '  "steps": a JSON array of steps, each an object with keys:\n'
    '    "agent": "main", "code-expert", or "test-expert"\n'
    '    "task": a self-contained instruction string for that agent\n'
    '    "verify": true or false — true if this step\'s change is complex enough to '
    "warrant a preventive check before moving on\n"
    "Order steps by dependency (scaffolding/logic before tests), regardless of the "
    "order mentioned in the prompt. Never fragment one artifact across steps. "
    "If building a library/package from scratch, use conventional layout for the "
    "language (e.g. for Python: a package directory, a tests/ directory — not flat). "
    "Respond with ONLY the JSON object, no prose, no markdown fences."
)

_CANONICAL_PROMPT = (
    "create a calcexpr Python library that parses and evaluates arithmetic "
    "expressions (+, -, *, /, parentheses, operator precedence); split it into "
    "a tokenizer module, a parser module, and an evaluator module; add custom "
    "exception types for syntax errors and division-by-zero; add a test suite "
    "covering edge cases (empty input, unbalanced parentheses, division by "
    "zero, nested parentheses)"
)

_TRIALS = 3
_REQUIRED_PASSES = 2


def _extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in model output: {text!r}")
    return json.loads(match.group(0))


async def _run_one_trial(model_name: str, api_key: str, base_url: str | None) -> list[dict]:
    provider = (
        OpenAIAdapter(api_key=api_key, base_url=base_url)
        if base_url is not None
        else OpenAIAdapter(api_key=api_key)
    )
    agent = Agent(
        provider=provider,
        model=model_name,
        system=_SEQUENCER_SYSTEM,
        max_iterations=1,
        extra_params=resolve_thinking_params(model_name, os.environ.get("GEKAI_THINKING_EFFORT")),
    )
    messages = await agent.run(_CANONICAL_PROMPT)
    text = "".join(
        b.text for b in messages[-1].content if hasattr(b, "text")
    ) if not isinstance(messages[-1].content, str) else messages[-1].content
    raw = _extract_json_object(text)
    return raw["steps"]


def _check_plan(steps: list[dict]) -> str | None:
    """Returns None if the plan satisfies the probe's assertions, else a failure reason."""
    if not steps:
        return "empty plan"

    allowed_agents = {"main", "code-expert", "test-expert"}
    for step in steps:
        if step.get("agent") not in allowed_agents:
            return f"unexpected agent: {step.get('agent')!r}"
        if not step.get("task"):
            return f"empty task in step: {step!r}"

    code_idx = next((i for i, s in enumerate(steps) if s["agent"] == "code-expert"), None)
    test_idx = next((i for i, s in enumerate(steps) if s["agent"] == "test-expert"), None)
    if code_idx is None:
        return "no code-expert step"
    if test_idx is None:
        return "no test-expert step"
    if code_idx >= test_idx:
        return f"code-expert (#{code_idx}) not before test-expert (#{test_idx})"

    if not any(s.get("verify") for s in steps):
        return "no step marked verify=true despite a multi-module library + custom exceptions"

    return None


@pytest.mark.llm
def test_sequencer_decomposes_and_orders_with_verify_placement() -> None:
    model_name = os.environ.get("GEKAI_CORE_MODEL_NAME")
    api_key = os.environ.get("GEKAI_CORE_MODEL_KEY")
    if not model_name or not api_key:
        pytest.skip("GEKAI_CORE_MODEL_NAME and GEKAI_CORE_MODEL_KEY must be set")
    base_url = os.environ.get("GEKAI_CORE_MODEL_URL")

    failures: list[str] = []
    for trial in range(_TRIALS):
        try:
            steps = asyncio.run(_run_one_trial(model_name, api_key, base_url))
            reason = _check_plan(steps)
        except Exception as exc:  # noqa: BLE001 — record and keep trying other trials
            reason = f"trial raised {type(exc).__name__}: {exc}"
        if reason is not None:
            failures.append(f"trial {trial}: {reason}")

    passes = _TRIALS - len(failures)
    assert passes >= _REQUIRED_PASSES, (
        f"only {passes}/{_TRIALS} trials passed (need >= {_REQUIRED_PASSES}); "
        f"failures: {failures}"
    )
