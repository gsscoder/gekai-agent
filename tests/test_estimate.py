"""Tests for the Estimator scope classifier (plan 27 improvement 5 —
repurposed to the trivial-vs-mutate binary; decomposition/specialist
assignment is now the planner's job, not the estimator's).

The estimator outputs exactly TRIVIAL | MUTATE. Any parse failure or
exception falls back to a trivial (all-default) ScopeEstimate — fail open,
mirroring the gate's unrecognized-token -> ACT fallback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent.pipeline.estimate import Estimator, ScopeEstimate
from tests.conftest import mock_llm_response, run


def _make_estimator() -> Estimator:
    with patch("agent.pipeline.estimate.AsyncOpenAI"):
        return Estimator(model="test-model", api_key="key", api_base="http://localhost")


def test_estimate_trivial() -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(return_value=mock_llm_response("TRIVIAL"))
    result = run(e.estimate("write a small script"))
    assert result == ScopeEstimate()
    assert result.mutate is False


def test_estimate_mutate() -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(return_value=mock_llm_response("MUTATE"))
    result = run(e.estimate("build a module and its test suite"))
    assert result.mutate is True


@pytest.mark.parametrize("response_text", ["nonsense-token", "  TRIVIAL  "])
def test_estimate_parse_failure_falls_back_to_trivial(response_text: str) -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(return_value=mock_llm_response(response_text))
    result = run(e.estimate("do something"))
    assert result == ScopeEstimate()


def test_estimate_exception_falls_back_to_trivial() -> None:
    e = _make_estimator()
    e._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    result = run(e.estimate("do something"))
    assert result == ScopeEstimate()
