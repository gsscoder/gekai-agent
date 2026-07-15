from __future__ import annotations

import pytest

from agent import credentials


@pytest.fixture(autouse=True)
def _fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    store: dict[str, str] = {}
    monkeypatch.setattr(credentials.keyring, "set_password", lambda service, name, key: store.__setitem__(name, key))
    monkeypatch.setattr(credentials.keyring, "get_password", lambda service, name: store.get(name))
    return store


def test_has_api_key_false_when_absent() -> None:
    assert credentials.has_api_key("some-model") is False


def test_set_then_get_round_trip() -> None:
    credentials.set_api_key("some-model", "sk-secret")
    assert credentials.get_api_key("some-model") == "sk-secret"
    assert credentials.has_api_key("some-model") is True


def test_get_raises_when_absent() -> None:
    with pytest.raises(LookupError, match="no API key stored"):
        credentials.get_api_key("ghost-model")


def test_dedup_across_tiers_same_model(_fake_keyring: dict[str, str]) -> None:
    """One credential per model name — assigning the same model to a second
    tier must find the existing entry, never prompt/store again."""
    credentials.set_api_key("claude-sonnet-4-6", "sk-first")
    assert credentials.has_api_key("claude-sonnet-4-6") is True
    # a second "assignment" of the same model just re-reads, doesn't re-store
    assert credentials.get_api_key("claude-sonnet-4-6") == "sk-first"
    assert len(_fake_keyring) == 1
