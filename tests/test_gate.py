"""Tests for the Gate classifier (plan 25 Improvement 1 — gate replaces router).

The gate outputs exactly TRIVIAL | REJECTED <name> | ACT.
An unrecognized token falls back to ACT (not to a silent read path).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent.pipeline.gate import Gate, Route
from agent.pipeline._directives import PIPELINE_DIRECTIVES


def run(coro):
    return asyncio.run(coro)


def _make_gate() -> Gate:
    with patch("agent.pipeline.gate.AsyncOpenAI"):
        return Gate(model="test-model", api_key="key", api_base="http://localhost")


def _mock_response(text: str):
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice])


def test_gate_system_prompt_contains_pipeline_directives() -> None:
    g = _make_gate()
    assert g._prompt.startswith(PIPELINE_DIRECTIVES)


def test_gate_trivial() -> None:
    g = _make_gate()
    g._client.chat.completions.create = AsyncMock(return_value=_mock_response("TRIVIAL"))
    route = run(g.gate("hi there"))
    assert route.trivial
    assert route.subagent is None
    assert not route.rejected


def test_gate_act() -> None:
    g = _make_gate()
    g._client.chat.completions.create = AsyncMock(return_value=_mock_response("ACT"))
    route = run(g.gate("create a module"))
    assert not route.trivial
    assert not route.rejected
    assert route.subagent is None


def test_gate_rejected_with_name() -> None:
    g = _make_gate()
    g._client.chat.completions.create = AsyncMock(return_value=_mock_response("REJECTED agent-xxx"))
    route = run(g.gate("use agent-xxx to do this"))
    assert route.rejected
    assert route.reason == "agent-xxx"
    assert not route.trivial


def test_gate_rejected_bare() -> None:
    g = _make_gate()
    g._client.chat.completions.create = AsyncMock(return_value=_mock_response("REJECTED"))
    route = run(g.gate("use some agent"))
    assert route.rejected
    assert route.reason == ""


def test_gate_unknown_token_falls_back_to_act() -> None:
    g = _make_gate()
    g._client.chat.completions.create = AsyncMock(return_value=_mock_response("QUERY"))
    route = run(g.gate("where is the router module?"))
    assert not route.trivial
    assert not route.rejected


def test_gate_strips_extra_whitespace() -> None:
    g = _make_gate()
    g._client.chat.completions.create = AsyncMock(return_value=_mock_response("  TRIVIAL  "))
    route = run(g.gate("hi"))
    assert route.trivial


def test_gate_nonsense_token_falls_back_to_act() -> None:
    g = _make_gate()
    g._client.chat.completions.create = AsyncMock(return_value=_mock_response("nonsense-token"))
    route = run(g.gate("do something"))
    assert not route.trivial
    assert not route.rejected


def test_gate_passes_history_to_model() -> None:
    g = _make_gate()
    create_mock = AsyncMock(return_value=_mock_response("ACT"))
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


def test_route_default_rejected_is_false() -> None:
    assert Route().rejected is False


def test_route_no_plan_or_query_fields() -> None:
    # Route must not have plan/query/plan_requested — these were removed in plan 25 Improvement 1
    r = Route()
    assert not hasattr(r, "plan")
    assert not hasattr(r, "query")
    assert not hasattr(r, "plan_requested")
