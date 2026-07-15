from __future__ import annotations

import pytest

from agent.llm.tiers import (
    ModelCatalogEntry,
    TierBinding,
    TierName,
    TierSuitability,
    validate_binding,
)


def test_catalog_entry_rejects_empty_efforts() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ModelCatalogEntry(name="m", base_url=None, efforts=(), thinking=False)


def test_catalog_entry_rejects_unordered_efforts() -> None:
    with pytest.raises(ValueError, match="ascending order"):
        ModelCatalogEntry(name="m", base_url=None, efforts=("high", "low"), thinking=False)


def test_catalog_entry_rejects_duplicate_efforts() -> None:
    with pytest.raises(ValueError, match="ascending order"):
        ModelCatalogEntry(name="m", base_url=None, efforts=("low", "low"), thinking=False)


def test_catalog_entry_allows_gaps() -> None:
    entry = ModelCatalogEntry(name="m", base_url=None, efforts=("low", "high", "max"), thinking=False)
    assert entry.efforts == ("low", "high", "max")  # no "medium"/"xhigh" — a real provider gap


def test_catalog_entry_rejects_core_thinking_override_without_thinking() -> None:
    with pytest.raises(ValueError, match="core_thinking"):
        ModelCatalogEntry(
            name="m", base_url=None, efforts=("low",), thinking=False,
            suitability=TierSuitability(core_thinking="ok"),
        )


def test_tier_binding_rejects_unknown_effort() -> None:
    with pytest.raises(ValueError, match="unknown effort"):
        TierBinding(model="m", default_effort="not-a-level")


def test_validate_binding_rejects_missing_model() -> None:
    with pytest.raises(ValueError, match="not in the catalog"):
        validate_binding(TierName.SUPP, TierBinding(model="ghost", default_effort="low"), catalog={})


def test_validate_binding_rejects_effort_not_declared_by_model() -> None:
    catalog = {"m": ModelCatalogEntry(name="m", base_url=None, efforts=("low", "high"), thinking=False)}
    with pytest.raises(ValueError, match="declared efforts"):
        validate_binding(TierName.SUPP, TierBinding(model="m", default_effort="medium"), catalog)


def test_validate_binding_rejects_thinking_outside_core() -> None:
    catalog = {"m": ModelCatalogEntry(name="m", base_url=None, efforts=("low",), thinking=True)}
    with pytest.raises(ValueError, match="CORE-only"):
        validate_binding(TierName.SUPP, TierBinding(model="m", default_effort="low", thinking=True), catalog)


def test_validate_binding_rejects_thinking_on_non_thinking_model() -> None:
    catalog = {"m": ModelCatalogEntry(name="m", base_url=None, efforts=("low",), thinking=False)}
    with pytest.raises(ValueError, match="does not support thinking"):
        validate_binding(TierName.CORE, TierBinding(model="m", default_effort="low", thinking=True), catalog)


def test_validate_binding_accepts_well_formed_binding() -> None:
    catalog = {"m": ModelCatalogEntry(name="m", base_url=None, efforts=("low", "high"), thinking=True)}
    validate_binding(TierName.CORE, TierBinding(model="m", default_effort="high", thinking=True), catalog)
