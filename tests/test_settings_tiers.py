from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import settings
from agent.llm.tiers import DEFAULT_MODEL_CATALOG, ModelCatalogEntry, TierBinding, TierName
from tests.conftest import TIER_BINDINGS, TIER_CATALOG


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings.Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def test_seed_writes_default_catalog_when_absent(tmp_path: Path) -> None:
    settings.seed_model_catalog_if_absent()
    data = json.loads((tmp_path / ".gekai" / "settings.json").read_text())
    assert len(data["models"]) == len(DEFAULT_MODEL_CATALOG)
    names = {m["name"] for m in data["models"]}
    assert names == {e.name for e in DEFAULT_MODEL_CATALOG}


def test_seed_never_touches_existing_models_node(tmp_path: Path) -> None:
    settings_path = tmp_path / ".gekai" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps({"models": [{"name": "custom", "base_url": None, "efforts": ["low"], "thinking": False, "suitability": {}}]}))

    settings.seed_model_catalog_if_absent()

    data = json.loads(settings_path.read_text())
    assert len(data["models"]) == 1
    assert data["models"][0]["name"] == "custom"


def test_catalog_round_trip() -> None:
    entry = ModelCatalogEntry(name="my-model", base_url="https://example.com", efforts=("low", "high"), thinking=True)
    settings.save_model_catalog_entry(entry)

    loaded = settings.load_model_catalog()
    assert loaded["my-model"] == entry


def test_save_model_catalog_entry_replaces_existing_by_name() -> None:
    settings.save_model_catalog_entry(ModelCatalogEntry(name="m", base_url=None, efforts=("low",), thinking=False))
    settings.save_model_catalog_entry(ModelCatalogEntry(name="m", base_url="https://new.example.com", efforts=("low", "high"), thinking=True))

    loaded = settings.load_model_catalog()
    assert len(loaded) == 1
    assert loaded["m"].base_url == "https://new.example.com"


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


def test_load_model_catalog_empty_when_no_settings_file() -> None:
    assert settings.load_model_catalog() == {}


def test_load_tier_bindings_empty_when_no_settings_file() -> None:
    assert settings.load_tier_bindings() == {}


# --- tiers_configured() (bug fix: "fully resolvable", not merely "has a binding") ---

def _seed_catalog_and_bindings() -> None:
    for entry in TIER_CATALOG.values():
        settings.save_model_catalog_entry(entry)
    for tier, binding in TIER_BINDINGS.items():
        settings.save_tier_binding(tier, binding)


def test_tiers_configured_false_when_no_bindings() -> None:
    assert settings.tiers_configured() is False


def test_tiers_configured_false_when_only_partially_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    for entry in TIER_CATALOG.values():
        settings.save_model_catalog_entry(entry)
    settings.save_tier_binding(TierName.FAST, TIER_BINDINGS[TierName.FAST])
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)

    assert settings.tiers_configured() is False


def test_tiers_configured_false_when_bound_but_missing_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    # The exact bug this fixes: all three tiers have a saved binding (the old
    # binding-only tiers_configured() would have returned True here), but one
    # bound model has no stored keyring credential, so `resolve_tier` would
    # still raise on first use — tiers_configured() must now agree and report
    # False.
    _seed_catalog_and_bindings()
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: name != "fast-model")

    assert settings.tiers_configured() is False


def test_tiers_configured_true_when_fully_resolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog_and_bindings()
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)

    assert settings.tiers_configured() is True
