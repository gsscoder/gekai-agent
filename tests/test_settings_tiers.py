from __future__ import annotations

from pathlib import Path

import pytest

from agent import settings
from agent.llm.tiers import DEFAULT_MODEL_CATALOG, ModelCatalogEntry, TierBinding, TierName
from tests.conftest import TIER_BINDINGS, TIER_CATALOG


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings.Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def test_load_model_catalog_is_a_live_read_of_the_code_side_catalog() -> None:
    # The catalog of *available* models is never persisted to disk — it's
    # always a live dict built straight from DEFAULT_MODEL_CATALOG, so a
    # model added there shows up immediately with no seed/round-trip step.
    loaded = settings.load_model_catalog()
    assert loaded.keys() == {e.name for e in DEFAULT_MODEL_CATALOG}
    for entry in DEFAULT_MODEL_CATALOG:
        assert loaded[entry.name] == entry


def test_load_model_catalog_reflects_catalog_additions_live(monkeypatch: pytest.MonkeyPatch) -> None:
    extra = ModelCatalogEntry(name="new-model", base_url="https://example.com", efforts=("low", "high"), thinking=True)
    monkeypatch.setattr(settings, "DEFAULT_MODEL_CATALOG", (*DEFAULT_MODEL_CATALOG, extra))

    loaded = settings.load_model_catalog()

    assert loaded["new-model"] == extra


def test_tier_binding_round_trip() -> None:
    binding = TierBinding(model="claude-sonnet-4-6", default_effort="medium", thinking=False)
    settings.save_tier_binding(TierName.SUPP, binding)

    loaded = settings.load_tier_bindings()
    assert loaded[TierName.SUPP] == binding


def test_tier_binding_save_overwrites_same_tier() -> None:
    settings.save_tier_binding(TierName.CORE, TierBinding(model="a", default_effort="low"))
    settings.save_tier_binding(TierName.CORE, TierBinding(model="b", default_effort="high"))

    loaded = settings.load_tier_bindings()
    assert len(loaded) == 1
    assert loaded[TierName.CORE].model == "b"


def test_load_tier_bindings_empty_when_no_settings_file() -> None:
    assert settings.load_tier_bindings() == {}


# --- tiers_configured() (bug fix: "fully resolvable", not merely "has a binding") ---

def _seed_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    # The catalog is a live read of DEFAULT_MODEL_CATALOG now — there's no
    # disk write to seed, so tests that need TIER_CATALOG's fake models
    # resolvable must monkeypatch the code-side catalog directly instead.
    monkeypatch.setattr(settings, "DEFAULT_MODEL_CATALOG", tuple(TIER_CATALOG.values()))


def _seed_catalog_and_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog(monkeypatch)
    for tier, binding in TIER_BINDINGS.items():
        settings.save_tier_binding(tier, binding)


def test_tiers_configured_false_when_no_bindings() -> None:
    assert settings.tiers_configured() is False


def test_tiers_configured_false_when_only_partially_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog(monkeypatch)
    settings.save_tier_binding(TierName.FAST, TIER_BINDINGS[TierName.FAST])
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)

    assert settings.tiers_configured() is False


def test_tiers_configured_false_when_bound_but_missing_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    # The exact bug this fixes: all three tiers have a saved binding (the old
    # binding-only tiers_configured() would have returned True here), but one
    # tier has no stored keyring credential, so `resolve_tier` would still
    # raise on first use — tiers_configured() must now agree and report
    # False.
    _seed_catalog_and_bindings(monkeypatch)
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: name != "openai:fast-model")

    assert settings.tiers_configured() is False


def test_tiers_configured_true_when_fully_resolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog_and_bindings(monkeypatch)
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)

    assert settings.tiers_configured() is True
