"""Tests for the Estimator scope classifier (plan 33 Phase 1 — folds Gate's
chat/act axis into the Estimator as a third rung, alongside history support).

The estimator outputs exactly CHAT | SOLO | MUTATE. Any parse failure or
exception falls back to the safe middle rung, `ScopeEstimate(scope="solo")`
— never "chat" (would skip needed codebase access) and never "mutate"
(would spend a planning call unnecessarily).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent.pipeline.estimate import Estimator, ScopeEstimate
from tests.conftest import mock_llm_response, run


def _make_estimator() -> Estimator:
    with patch("agent.pipeline.estimate.AsyncOpenAI"):
        return Estimator(model="test-model", api_key="key", api_base="http://localhost")


def test_estimate_chat() -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(return_value=mock_llm_response("CHAT"))
    result = run(e.estimate("hi there"))
    assert result == ScopeEstimate(scope="chat")


def test_estimate_solo() -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(return_value=mock_llm_response("SOLO"))
    result = run(e.estimate("write a small script"))
    assert result == ScopeEstimate(scope="solo")


def test_estimate_mutate() -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(return_value=mock_llm_response("MUTATE"))
    result = run(e.estimate("build a module and its test suite"))
    assert result == ScopeEstimate(scope="mutate")


def test_estimate_parse_failure_falls_back_to_solo() -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(return_value=mock_llm_response("nonsense-token"))
    result = run(e.estimate("do something"))
    assert result == ScopeEstimate(scope="solo")


def test_estimate_exception_falls_back_to_solo() -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    result = run(e.estimate("do something"))
    assert result == ScopeEstimate(scope="solo")


def test_estimate_default_scope_is_solo() -> None:
    assert ScopeEstimate().scope == "solo"


def test_estimate_history_filtered_and_sliced_to_last_six() -> None:
    """Mirrors `Gate.gate()`'s history handling exactly: only user/assistant
    messages are kept, and only the last 6 of those, positioned between the
    system prompt and the current user_input in the chat completion call."""
    e = _make_estimator()
    mock_create = AsyncMock(return_value=mock_llm_response("SOLO"))
    e._client.chat.completions.create = mock_create

    history = [
        {"role": "system", "content": "should be dropped"},
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "turn 1 reply"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "turn 2 reply"},
        {"role": "user", "content": "turn 3"},
        {"role": "assistant", "content": "turn 3 reply"},
        {"role": "user", "content": "turn 4"},
    ]
    run(e.estimate("do it", history=history))

    messages = mock_create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "do it"}
    assert messages[1:-1] == history[-6:]
    assert all(m["role"] in ("user", "assistant") for m in messages[1:-1])


def test_estimate_no_history_omits_context_messages() -> None:
    e = _make_estimator()
    mock_create = AsyncMock(return_value=mock_llm_response("SOLO"))
    e._client.chat.completions.create = mock_create

    run(e.estimate("hi"))

    messages = mock_create.call_args.kwargs["messages"]
    assert len(messages) == 2
    assert messages[-1] == {"role": "user", "content": "hi"}


def test_estimate_sends_extra_params() -> None:
    # plan 34 phase 1: the estimator must forward whatever extra_params its
    # resolved tier carries (e.g. the explicit thinking-disable payload) —
    # a `thinking: false` binding must not silently degrade to `{}`.
    with patch("agent.pipeline.estimate.AsyncOpenAI"):
        e = Estimator(
            model="test-model",
            api_key="key",
            api_base="http://localhost",
            extra_params={"extra_body": {"thinking": {"type": "disabled"}}},
        )
    mock_create = AsyncMock(return_value=mock_llm_response("CHAT"))
    e._client.chat.completions.create = mock_create

    run(e.estimate("hi"))

    assert mock_create.call_args.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_estimate_no_extra_params_defaults_to_empty() -> None:
    e = _make_estimator()
    mock_create = AsyncMock(return_value=mock_llm_response("CHAT"))
    e._client.chat.completions.create = mock_create

    run(e.estimate("hi"))

    assert "extra_body" not in mock_create.call_args.kwargs


def test_estimate_prompt_flags_multi_layer_requests_as_mutate() -> None:
    """Regression: the estimator under-called a backend+frontend request as SOLO,
    exhausting the solo harness's 25-iteration budget instead of routing to the
    sequencer's per-step budgets (session 2d573c47, turn fb02cf81, 2026-08-23)."""
    from agent.pipeline.estimate import _ESTIMATE_PROMPT
    assert "more than one layer of the stack" in _ESTIMATE_PROMPT
