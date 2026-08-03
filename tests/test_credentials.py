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


def test_dedup_same_credential_key(_fake_keyring: dict[str, str]) -> None:
    """Re-storing under the exact same key just overwrites/re-reads the
    existing entry, never creates a second one."""
    credentials.set_api_key("claude-sonnet-4-6", "sk-first")
    assert credentials.has_api_key("claude-sonnet-4-6") is True
    assert credentials.get_api_key("claude-sonnet-4-6") == "sk-first"
    assert len(_fake_keyring) == 1


def test_credential_key_format() -> None:
    assert credentials.credential_key("fast", "deepseek-v4-flash", "low", False) == "fast-deepseek-v4-flash-low-n"
    assert credentials.credential_key("core", "deepseek-v4-pro", "high", True) == "core-deepseek-v4-pro-high-y"


def test_credential_key_distinguishes_tiers_sharing_a_model() -> None:
    """Two tiers bound to the same model at the same effort/thinking still
    get distinct keyring entries — the exact bug this scheme fixes (see
    tests/test_tiers_grid_flow.py::test_key_is_scoped_to_the_row_it_was_pasted_in)."""
    fast_key = credentials.credential_key("fast", "model-a", "low", False)
    supp_key = credentials.credential_key("supp", "model-a", "low", False)
    assert fast_key != supp_key
