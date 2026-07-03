"""Tests for the Estimator scope classifier (plan 26 Improvement 1 — the
scope-estimate pre-pass, mechanism #3).

The estimator outputs exactly TRIVIAL | IMPLEMENTATION <names>. Any parse
failure or exception falls back to a trivial (all-default) ScopeEstimate —
fail open, mirroring the gate's unrecognized-token -> ACT fallback.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent.pipeline.estimate import Estimator, ScopeEstimate


def run(coro):
    return asyncio.run(coro)


def _make_estimator() -> Estimator:
    with patch("agent.pipeline.estimate.AsyncOpenAI"):
        return Estimator(model="test-model", api_key="key", api_base="http://localhost")


def _mock_response(text: str):
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice])


def test_estimate_trivial() -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(return_value=_mock_response("TRIVIAL"))
    result = run(e.estimate("write a small script"))
    assert result == ScopeEstimate()
    assert result.implementation_sized is False
    assert result.specialists == []


def test_estimate_implementation_with_valid_names() -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(
        return_value=_mock_response("IMPLEMENTATION code-expert, test-expert")
    )
    result = run(e.estimate("build a module and its test suite"))
    assert result.implementation_sized is True
    assert result.specialists == ["code-expert", "test-expert"]


def test_estimate_filters_unknown_names() -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(
        return_value=_mock_response("IMPLEMENTATION code-expert, made-up-agent")
    )
    result = run(e.estimate("build a module"))
    assert result.implementation_sized is True
    assert result.specialists == ["code-expert"]


def test_estimate_parse_failure_falls_back_to_trivial() -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(return_value=_mock_response("nonsense-token"))
    result = run(e.estimate("do something"))
    assert result == ScopeEstimate()


def test_estimate_exception_falls_back_to_trivial() -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    result = run(e.estimate("do something"))
    assert result == ScopeEstimate()


def test_estimate_strips_extra_whitespace() -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(return_value=_mock_response("  TRIVIAL  "))
    result = run(e.estimate("hi"))
    assert result == ScopeEstimate()
