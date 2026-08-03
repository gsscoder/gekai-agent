from __future__ import annotations

import pytest

from agent.llm.resolve import TierResolutionError, all_tiers_ready, resolve_tier, resolve_touchpoint, tier_status
from agent.llm.tiers import ModelCatalogEntry, TierBinding, TierName, TierSuitability


CATALOG = {
    "flash": ModelCatalogEntry(name="flash", base_url="https://api.example.com", efforts=("low", "medium"), thinking=False),
    "pro": ModelCatalogEntry(name="pro", base_url="https://api.example.com", efforts=("high", "xhigh"), thinking=True),
}


@pytest.mark.parametrize(
    "bindings, match",
    [
        ({}, "not configured"),
        ({TierName.FAST: TierBinding(model="ghost", default_effort="low")}, "stale"),
    ],
)
def test_resolve_tier_raises_for_unresolvable_binding(bindings, match):
    with pytest.raises(TierResolutionError, match=match):
        resolve_tier(TierName.FAST, CATALOG, bindings)


def test_resolve_tier_missing_credential(monkeypatch):
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: False)
    bindings = {TierName.FAST: TierBinding(model="flash", default_effort="low")}
    with pytest.raises(TierResolutionError, match="no stored credential"):
        resolve_tier(TierName.FAST, CATALOG, bindings)


def test_resolve_tier_success(monkeypatch):
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)
    monkeypatch.setattr("agent.llm.resolve.credentials.get_api_key", lambda name: "secret-key")
    bindings = {TierName.CORE: TierBinding(model="pro", default_effort="high", thinking=True)}
    resolved = resolve_tier(TierName.CORE, CATALOG, bindings)
    assert resolved.model == "pro"
    assert resolved.api_key == "secret-key"
    assert resolved.api_base == "https://api.example.com"
    assert isinstance(resolved.extra_params, dict)  # model_caps.py owns the actual param shape


def test_resolve_touchpoint_uses_nominal_tier(monkeypatch):
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)
    monkeypatch.setattr("agent.llm.resolve.credentials.get_api_key", lambda name: "secret-key")
    bindings = {
        TierName.FAST: TierBinding(model="flash", default_effort="low"),
        TierName.CORE: TierBinding(model="pro", default_effort="high"),
    }
    resolved = resolve_touchpoint("gate", CATALOG, bindings)  # gate is nominal FAST
    assert resolved.model == "flash"


def test_tier_status_no_binding():
    status = tier_status(TierName.FAST, CATALOG, {})
    assert status.binding is None
    assert status.entry is None
    assert status.ready is False
    assert status.label == "not configured"


def test_tier_status_stale_model_missing_from_catalog():
    bindings = {TierName.FAST: TierBinding(model="ghost", default_effort="low")}
    status = tier_status(TierName.FAST, CATALOG, bindings)
    assert status.binding is not None
    assert status.entry is None
    assert status.ready is False
    assert status.label == "stale — model missing from catalog"


def test_tier_status_no_credential(monkeypatch):
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: False)
    bindings = {TierName.FAST: TierBinding(model="flash", default_effort="low")}
    status = tier_status(TierName.FAST, CATALOG, bindings)
    assert status.entry is not None
    assert status.has_credential is False
    assert status.ready is False
    assert status.label == "no key"


def test_tier_status_invalid_binding_thinking_on_non_thinking_model():
    # thinking=True on "flash" (thinking=False in the catalog) — validate_binding
    # rejects this; tier_status must report it as stale, not silently accept it.
    bindings = {TierName.CORE: TierBinding(model="flash", default_effort="low", thinking=True)}
    status = tier_status(TierName.CORE, CATALOG, bindings)
    assert status.entry is not None
    assert status.invalid_reason is not None
    assert status.ready is False
    assert "stale" in status.label


def test_tier_status_ready(monkeypatch):
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)
    bindings = {TierName.FAST: TierBinding(model="flash", default_effort="low")}
    status = tier_status(TierName.FAST, CATALOG, bindings)
    assert status.ready is True
    assert status.label == "ready"


def test_tier_status_verdict_surfaces_in_label(monkeypatch):
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)
    catalog = dict(CATALOG)
    catalog["flash"] = ModelCatalogEntry(
        name="flash", base_url="https://api.example.com", efforts=("low", "medium"), thinking=False,
        suitability=TierSuitability(fast="warning"),
    )
    bindings = {TierName.FAST: TierBinding(model="flash", default_effort="low")}
    status = tier_status(TierName.FAST, catalog, bindings)
    assert status.ready is True  # suitability is advisory-only — never blocks readiness
    assert status.label == "warning"


def test_all_tiers_ready_false_until_every_tier_resolves(monkeypatch):
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)
    bindings = {
        TierName.FAST: TierBinding(model="flash", default_effort="low"),
        TierName.SUPP: TierBinding(model="flash", default_effort="low"),
    }
    assert all_tiers_ready(CATALOG, bindings) is False  # CORE still missing

    bindings[TierName.CORE] = TierBinding(model="pro", default_effort="high")
    assert all_tiers_ready(CATALOG, bindings) is True


def test_all_tiers_ready_false_when_binding_present_but_no_credential(monkeypatch):
    # This is the exact incoherence bug: a binding exists for every tier but
    # one tier has no stored key — old tiers_configured() (binding-only)
    # would have reported True here.
    def _has_key(name: str) -> bool:
        return not name.startswith("fast-")

    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", _has_key)
    bindings = {
        TierName.FAST: TierBinding(model="flash", default_effort="low"),
        TierName.SUPP: TierBinding(model="flash", default_effort="low"),
        TierName.CORE: TierBinding(model="pro", default_effort="high"),
    }
    assert all_tiers_ready(CATALOG, bindings) is False
