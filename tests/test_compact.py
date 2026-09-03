"""Tests for the `/compact` core logic (Phase 1 of 6 — no TUI, no
persistence). Covers the context-usage thresholds, session mutation, and the
summarization LLM call in isolation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from agent.compact import apply_summary, context_state, summarize
from agent.tiers.resolve import ResolvedTier
from agent.persona import ROOT_SYSTEM_PROMPT
from agent.session import Session
from tests.conftest import mock_llm_response, run


def _make_tier() -> ResolvedTier:
    return ResolvedTier(model="test-model", api_key="key", api_base="http://localhost", extra_params={})


def test_context_state_below_warn() -> None:
    assert context_state(749, 1000) == "ok"


def test_context_state_at_warn_boundary() -> None:
    assert context_state(750, 1000) == "warn"


def test_context_state_below_auto() -> None:
    assert context_state(799, 1000) == "warn"


def test_context_state_at_auto_boundary() -> None:
    assert context_state(800, 1000) == "auto"


def test_context_state_at_full() -> None:
    assert context_state(1000, 1000) == "auto"


def test_apply_summary_replaces_messages_with_two_entries() -> None:
    session = Session()
    session.messages = [
        {"role": "system", "content": "old system prompt"},
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "reply 2"},
    ]

    apply_summary(session, "the summary text")

    assert len(session.messages) == 2
    assert session.messages[0] == {"role": "system", "content": ROOT_SYSTEM_PROMPT}
    assert session.messages[1]["role"] == "user"
    assert "the summary text" in session.messages[1]["content"]


def test_summarize_sends_instructions_when_given() -> None:
    tier = _make_tier()
    mock_create = AsyncMock(return_value=mock_llm_response("summary output"))
    with patch("agent.oneshot.build_client") as mock_cls:
        mock_cls.return_value.chat.completions.create = mock_create
        result = run(summarize(
            [{"role": "system", "content": ROOT_SYSTEM_PROMPT}, {"role": "user", "content": "hi"}],
            tier,
            instructions="focus on the auth module",
        ))

    assert result == "summary output"
    messages = mock_create.call_args.kwargs["messages"]
    assert "focus on the auth module" in messages[0]["content"]


def test_summarize_omits_instructions_when_not_given() -> None:
    tier = _make_tier()
    mock_create = AsyncMock(return_value=mock_llm_response("summary output"))
    with patch("agent.oneshot.build_client") as mock_cls:
        mock_cls.return_value.chat.completions.create = mock_create
        run(summarize(
            [{"role": "system", "content": ROOT_SYSTEM_PROMPT}, {"role": "user", "content": "hi"}],
            tier,
        ))

    messages = mock_create.call_args.kwargs["messages"]
    assert "Additional Instructions" not in messages[0]["content"]


def test_summarize_drops_original_leading_system_message() -> None:
    tier = _make_tier()
    mock_create = AsyncMock(return_value=mock_llm_response("summary output"))
    with patch("agent.oneshot.build_client") as mock_cls:
        mock_cls.return_value.chat.completions.create = mock_create
        run(summarize(
            [{"role": "system", "content": ROOT_SYSTEM_PROMPT}, {"role": "user", "content": "hi"}],
            tier,
        ))

    messages = mock_create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] != ROOT_SYSTEM_PROMPT
    assert ROOT_SYSTEM_PROMPT not in [m["content"] for m in messages if m["role"] == "system"]
    assert {"role": "user", "content": "hi"} in messages
