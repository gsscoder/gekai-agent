from __future__ import annotations

import pytest

from agent.llm.tiers import (
    ModelCatalogEntry,
    TierBinding,
    TierName,
    TierSuitability,
    validate_binding,
)


@pytest.mark.parametrize(
    "efforts, match",
    [
        ((), "non-empty"),
        (("high", "low"), "ascending order"),
        (("low", "low"), "ascending order"),
    ],
)
def test_catalog_entry_rejects_invalid_efforts(efforts: tuple[str, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ModelCatalogEntry(name="m", base_url=None, efforts=efforts, thinking=False)


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


@pytest.mark.parametrize(
    "tier, catalog, binding, match",
    [
        (
            TierName.SUPP,
            {},
            TierBinding(model="ghost", default_effort="low"),
            "not in the catalog",
        ),
        (
            TierName.SUPP,
            {"m": ModelCatalogEntry(name="m", base_url=None, efforts=("low", "high"), thinking=False)},
            TierBinding(model="m", default_effort="medium"),
            "declared efforts",
        ),
        (
            TierName.SUPP,
            {"m": ModelCatalogEntry(name="m", base_url=None, efforts=("low",), thinking=True)},
            TierBinding(model="m", default_effort="low", thinking=True),
            "CORE-only",
        ),
        (
            TierName.CORE,
            {"m": ModelCatalogEntry(name="m", base_url=None, efforts=("low",), thinking=False)},
            TierBinding(model="m", default_effort="low", thinking=True),
            "does not support thinking",
        ),
    ],
)
def test_validate_binding_rejects_invalid_bindings(
    tier: TierName,
    catalog: dict[str, ModelCatalogEntry],
    binding: TierBinding,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_binding(tier, binding, catalog)


def test_validate_binding_accepts_well_formed_binding() -> None:
    catalog = {"m": ModelCatalogEntry(name="m", base_url=None, efforts=("low", "high"), thinking=True)}
    validate_binding(TierName.CORE, TierBinding(model="m", default_effort="high", thinking=True), catalog)
