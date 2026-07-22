"""Tests for the Gate classifier (plan 27 improvement 4 — REJECTED <name>
dropped; the gate is a pure chit-chat/act binary. Rejection of an unknown
agent is capability-based, at the command layer (`/agent-x`), never a
model-side name check).

The gate outputs exactly TRIVIAL | ACT. An unrecognized token falls back to
ACT (not to a silent read path).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent.pipeline.gate import Gate, Route
from agent.pipeline._directives import PIPELINE_DIRECTIVES
from tests.conftest import mock_llm_response, run


def _make_gate() -> Gate:
    with patch("agent.pipeline.gate.AsyncOpenAI"):
        return Gate(model="test-model", api_key="key", api_base="http://localhost")


def test_gate_system_prompt_contains_pipeline_directives() -> None:
    g = _make_gate()
    assert g._prompt.startswith(PIPELINE_DIRECTIVES)


@pytest.mark.parametrize("response_text", ["TRIVIAL", "  TRIVIAL  "])
def test_gate_trivial(response_text: str) -> None:
    g = _make_gate()
    g._client.chat.completions.create = AsyncMock(return_value=mock_llm_response(response_text))
    route = run(g.gate("hi there"))
    assert route.trivial


@pytest.mark.parametrize("response_text", ["ACT", "QUERY", "nonsense-token"])
def test_gate_act(response_text: str) -> None:
    g = _make_gate()
    g._client.chat.completions.create = AsyncMock(return_value=mock_llm_response(response_text))
    route = run(g.gate("create a module"))
    assert not route.trivial


def test_gate_passes_history_to_model() -> None:
    g = _make_gate()
    create_mock = AsyncMock(return_value=mock_llm_response("ACT"))
    g._client.chat.completions.create = create_mock
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "prev question"},
        {"role": "assistant", "content": "prev answer"},
    ]
    run(g.gate("follow-up question", history=history))
    call_messages = create_mock.call_args.kwargs["messages"]
    # gate system prompt + 1 user/assistant pair + current input = 4 messages
    assert len(call_messages) == 4
    assert call_messages[1]["role"] == "user"
    assert call_messages[1]["content"] == "prev question"


def test_route_default_trivial_is_false() -> None:
    assert Route().trivial is False


def test_route_no_rejected_plan_or_query_fields() -> None:
    # Route must not have rejected/plan/query/plan_requested fields (plan 27 improvement 4
    # drops REJECTED <name>; plan 25 improvement 1 already dropped plan/query)
    r = Route()
    assert not hasattr(r, "rejected")
    assert not hasattr(r, "reason")
    assert not hasattr(r, "plan")
    assert not hasattr(r, "query")
    assert not hasattr(r, "plan_requested")
