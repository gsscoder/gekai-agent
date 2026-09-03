"""Real-LLM integration probes for `agent/compact.py::summarize`.

These are smoke probes, not unit tests: they call a real model end-to-end to
confirm `summarize()` actually produces a usable summary and that its
`instructions` parameter genuinely steers the model's output — the existing
`tests/test_compact.py` already covers `summarize()`/`context_state()`/
`apply_summary()` behavior against a mocked `AsyncOpenAI` client, so no
mock-level cases are duplicated here.

Requires: GEKAI_SUPPORT_MODEL_NAME, GEKAI_SUPPORT_MODEL_KEY
Optional: GEKAI_SUPPORT_MODEL_URL

Run: pytest --llm-compact tests/test_compact_llm.py
"""

from __future__ import annotations

import asyncio
import os

import pytest

from agent import compact
from agent.llm.model_caps import resolve_thinking_params
from agent.tiers.resolve import ResolvedTier


def _support_tier() -> ResolvedTier | None:
    model_name = os.environ.get("GEKAI_SUPPORT_MODEL_NAME")
    api_key = os.environ.get("GEKAI_SUPPORT_MODEL_KEY")
    if not model_name or not api_key:
        return None
    base_url = os.environ.get("GEKAI_SUPPORT_MODEL_URL")
    return ResolvedTier(
        model=model_name,
        api_key=api_key,
        api_base=base_url or None,
        extra_params=resolve_thinking_params(model_name, "low", enabled=False),
    )


@pytest.mark.llm_compact
def test_summarize_preserves_planted_fact() -> None:
    tier = _support_tier()
    if tier is None:
        pytest.skip("GEKAI_SUPPORT_MODEL_NAME and GEKAI_SUPPORT_MODEL_KEY must be set")

    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Where is the auth bug?"},
        {
            "role": "assistant",
            "content": (
                "Found it: agent/tui/app.py line 1210 — the `read_only` flag on the "
                "prompt TextArea was never cleared when the command palette closed, "
                "so keystrokes after the first were silently dropped."
            ),
        },
    ]

    result = asyncio.run(compact.summarize(messages, tier, instructions=""))

    assert result
    assert "1210" in result.lower() or "read_only" in result.lower()


@pytest.mark.llm_compact
def test_summarize_instructions_steer_focus() -> None:
    tier = _support_tier()
    if tier is None:
        pytest.skip("GEKAI_SUPPORT_MODEL_NAME and GEKAI_SUPPORT_MODEL_KEY must be set")

    messages = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "What's the status of the build and CI?"},
        {
            "role": "assistant",
            "content": (
                "Two things: the database migration script is "
                "`migrations/0042_add_index.sql`, and the flaky CI job is called "
                "`windows-integration-tests`."
            ),
        },
    ]

    baseline = asyncio.run(compact.summarize(messages, tier, instructions=""))
    assert "0042" in baseline.lower()

    steered = asyncio.run(
        compact.summarize(
            messages,
            tier,
            instructions="Focus only on the flaky CI job; ignore the database migration entirely.",
        )
    )
    assert "windows-integration-tests" in steered.lower()
