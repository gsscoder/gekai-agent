"""Coverage for the shared `AsyncOpenAI` client factory: the property this
dedup exists to guarantee is that every call site gets the same connect/
write/pool timeout policy, with only `read_timeout` varying per caller.
"""

from __future__ import annotations

from agent.openai_client import build_openai_client


def test_default_read_timeout_matches_short_calls() -> None:
    client = build_openai_client("key", "http://localhost")
    timeout = client.timeout
    assert timeout.connect == 5.0
    assert timeout.read == 30.0
    assert timeout.write == 30.0
    assert timeout.pool == 30.0


def test_explicit_read_timeout_overrides_default_for_long_calls() -> None:
    client = build_openai_client("key", "http://localhost", read_timeout=120.0)
    timeout = client.timeout
    assert timeout.connect == 5.0
    assert timeout.read == 120.0
    assert timeout.write == 30.0
    assert timeout.pool == 30.0


def test_client_carries_api_key_and_base_url() -> None:
    client = build_openai_client("key", "http://localhost")
    assert client.api_key == "key"
    assert str(client.base_url) == "http://localhost"
