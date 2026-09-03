"""Coverage for the shared one-shot call helper: every non-agentic touchpoint
gets the same connect/write/pool timeout policy, with only `read_timeout`
varying per caller, and the same message assembly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from agent.oneshot import build_client, complete
from agent.tiers.resolve import ResolvedTier
from tests.conftest import mock_llm_response, run

TIER = ResolvedTier(model="test-model", api_key="key", api_base="http://localhost", extra_params={})


def test_default_read_timeout_matches_short_calls() -> None:
    client = build_client("key", "http://localhost")
    timeout = client.timeout
    assert timeout.connect == 5.0
    assert timeout.read == 30.0
    assert timeout.write == 30.0
    assert timeout.pool == 30.0


def test_explicit_read_timeout_overrides_default_for_long_calls() -> None:
    client = build_client("key", "http://localhost", 120.0)
    timeout = client.timeout
    assert timeout.connect == 5.0
    assert timeout.read == 120.0
    assert timeout.write == 30.0
    assert timeout.pool == 30.0


def test_client_carries_api_key_and_base_url() -> None:
    client = build_client("key", "http://localhost")
    assert client.api_key == "key"
    assert str(client.base_url) == "http://localhost"


def test_clients_are_reused_per_credential_endpoint_and_timeout() -> None:
    """A client owns an httpx connection pool, so a fresh one per call would
    leak a pool every turn."""
    assert build_client("key", "http://localhost") is build_client("key", "http://localhost")
    assert build_client("key", "http://localhost") is not build_client("key", "http://localhost", 60.0)


def _captured_call(**kwargs) -> dict:
    create = AsyncMock(return_value=mock_llm_response("answer"))
    with patch("agent.oneshot.build_client") as mock_build:
        mock_build.return_value.chat.completions.create = create
        result = run(complete(TIER, **kwargs))
    assert result == "answer"
    return create.await_args.kwargs


def test_messages_are_system_then_context_then_user() -> None:
    kwargs = _captured_call(
        system="sys",
        user="ask",
        context=[{"role": "user", "content": "earlier"}],
    )
    assert kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "earlier"},
        {"role": "user", "content": "ask"},
    ]
    assert kwargs["model"] == "test-model"


def test_temperature_is_omitted_unless_given() -> None:
    assert "temperature" not in _captured_call(system="sys", user="ask")
    assert _captured_call(system="sys", user="ask", temperature=0)["temperature"] == 0


def test_tier_extra_params_are_forwarded() -> None:
    tier = ResolvedTier(
        model="m", api_key="k", api_base=None, extra_params={"reasoning_effort": "high"},
    )
    create = AsyncMock(return_value=mock_llm_response("answer"))
    with patch("agent.oneshot.build_client") as mock_build:
        mock_build.return_value.chat.completions.create = create
        run(complete(tier, system="sys", user="ask"))
    assert create.await_args.kwargs["reasoning_effort"] == "high"
