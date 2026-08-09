from __future__ import annotations

import pytest

from agent.harness.touchpoints import TOUCHPOINTS
from agent.llm import model_caps
from agent.llm.model_caps import MODEL_CAPS, ModelCaps
from agent.llm.resolve import TierResolutionError, all_tiers_ready, resolve_tier, resolve_touchpoint, tier_status
from agent.llm.tiers import EFFORT_LADDER, ModelCatalogEntry, TierBinding, TierName, TierSuitability


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


def test_resolve_tier_thinking_false_resolves_to_explicit_disable_payload(monkeypatch):
    # plan 34 phase 1: a `thinking: false` binding must not silently resolve
    # to `{}` (which DeepSeek interprets as "unspecified" -> reasoning ON).
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)
    monkeypatch.setattr("agent.llm.resolve.credentials.get_api_key", lambda name: "secret-key")
    monkeypatch.setitem(MODEL_CAPS, "flash", ModelCaps(thinking=False, thinking_style="deepseek"))
    bindings = {TierName.FAST: TierBinding(model="flash", default_effort="low", thinking=False)}
    resolved = resolve_tier(TierName.FAST, CATALOG, bindings)
    assert resolved.extra_params == {"extra_body": {"thinking": {"type": "disabled"}}}


def test_resolve_tier_thinking_true_unchanged(monkeypatch):
    # Regression guard: a thinking=True binding's extra_params must be
    # byte-identical to before the `enabled` param existed.
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)
    monkeypatch.setattr("agent.llm.resolve.credentials.get_api_key", lambda name: "secret-key")
    monkeypatch.setitem(
        MODEL_CAPS, "pro", ModelCaps(thinking=True, thinking_style="deepseek", default_effort="high")
    )
    # _EFFORT_TO_PARAMS is now keyed by model id, not thinking_style (each
    # real DeepSeek model folds effort differently) — this fixture's "pro" id
    # isn't one of the real ids, so give it its own fold-down entry to keep
    # this test's model-agnostic intent.
    monkeypatch.setitem(model_caps._EFFORT_TO_PARAMS, "pro", {"high": {"reasoning_effort": "high"}})
    bindings = {TierName.CORE: TierBinding(model="pro", default_effort="high", thinking=True)}
    resolved = resolve_tier(TierName.CORE, CATALOG, bindings)
    assert resolved.extra_params == {"reasoning_effort": "high", "extra_body": {"thinking": {"type": "enabled"}}}


def test_resolve_touchpoint_uses_nominal_tier(monkeypatch):
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)
    monkeypatch.setattr("agent.llm.resolve.credentials.get_api_key", lambda name: "secret-key")
    bindings = {
        TierName.FAST: TierBinding(model="flash", default_effort="low"),
        TierName.CORE: TierBinding(model="pro", default_effort="high"),
    }
    resolved = resolve_touchpoint("estimator", CATALOG, bindings)  # estimator is nominal FAST
    assert resolved.model == "flash"


# --- per-touchpoint operating point (effort/thinking declared in code, on
# top of the tier binding that supplies the model + credentials) ---


def _record_credential_keys(monkeypatch) -> list[str]:
    seen: list[str] = []

    def _has_api_key(name: str) -> bool:
        seen.append(name)
        return True

    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", _has_api_key)
    monkeypatch.setattr("agent.llm.resolve.credentials.get_api_key", lambda name: f"key-for-{name}")
    return seen


def test_touchpoint_override_changes_params_but_never_the_credential_key(monkeypatch):
    # CRITICAL constraint: the keyring account name names what the user
    # actually stored via `/tiers` — the *binding's* operating point. Building
    # it from the code-side override instead would miss every stored key and
    # make each override demand a fresh `/tiers` entry, which is exactly the
    # churn per-touchpoint operating points exist to avoid.
    keys_seen = _record_credential_keys(monkeypatch)
    monkeypatch.setitem(MODEL_CAPS, "pro", ModelCaps(thinking=True, thinking_style="deepseek", default_effort="high"))
    bindings = {TierName.CORE: TierBinding(model="pro", default_effort="xhigh", thinking=True)}

    resolved = resolve_touchpoint("sequencer", CATALOG, bindings)  # declares effort="high", thinking=False

    assert keys_seen == ["core-pro-xhigh-y"], "credential lookup must use the binding's effort/thinking"
    assert resolved.api_key == "key-for-core-pro-xhigh-y"
    assert resolved.model == "pro"  # the binding still decides *which* model
    assert resolved.extra_params == {"extra_body": {"thinking": {"type": "disabled"}}}


def test_touchpoint_without_an_override_resolves_identically_to_its_bare_tier(monkeypatch):
    # Regression guard for every touchpoint but the sequencer: declaring no
    # operating point must resolve byte-identically to resolving its tier
    # directly, exactly as before per-touchpoint overrides existed.
    monkeypatch.setattr("agent.llm.resolve.credentials.has_api_key", lambda name: True)
    monkeypatch.setattr("agent.llm.resolve.credentials.get_api_key", lambda name: f"key-for-{name}")
    bindings = {
        TierName.FAST: TierBinding(model="flash", default_effort="low"),
        TierName.SUPP: TierBinding(model="flash", default_effort="medium"),
        TierName.CORE: TierBinding(model="pro", default_effort="high", thinking=True),
    }
    plain = [t for t in TOUCHPOINTS if t.effort is None and t.thinking is None]
    assert {t.name for t in plain} == {"estimator", "root-dispatch", "subagent-dispatch", "micro"}
    for tp in plain:
        assert resolve_touchpoint(tp.name, CATALOG, bindings) == resolve_tier(tp.nominal_tier, CATALOG, bindings), tp.name


def test_touchpoint_override_follows_whatever_tier_scaling_landed_on(monkeypatch):
    # `scale()` may demote the sequencer CORE->SUPP; its declared operating
    # point must then apply to whatever model SUPP resolves to, not silently
    # revert to that binding's own effort/thinking.
    keys_seen = _record_credential_keys(monkeypatch)
    asked: list[tuple] = []

    def _spy(model: str, effort: str | None = None, *, enabled: bool = True) -> dict:
        asked.append((model, effort, enabled))
        return {}

    monkeypatch.setattr("agent.llm.resolve.resolve_thinking_params", _spy)
    catalog = dict(CATALOG)
    catalog["flash"] = ModelCatalogEntry(
        name="flash", base_url="https://api.example.com", efforts=EFFORT_LADDER, thinking=False
    )
    bindings = {TierName.SUPP: TierBinding(model="flash", default_effort="low")}

    resolved = resolve_tier(TierName.SUPP, catalog, bindings, "sequencer")

    assert resolved.model == "flash"  # the demoted tier's model...
    assert asked == [("flash", "high", False)]  # ...operated at the sequencer's own point
    assert keys_seen == ["supp-flash-low-n"]  # ...on the demoted tier's own credential


def test_touchpoint_override_effort_the_model_does_not_declare_fails_loud(monkeypatch):
    # House style is a chat-visible TierResolutionError, never a silent
    # fallback to some nearby effort the model does declare.
    _record_credential_keys(monkeypatch)
    catalog = dict(CATALOG)
    catalog["pro"] = ModelCatalogEntry(
        name="pro", base_url="https://api.example.com", efforts=("xhigh", "max"), thinking=True
    )
    bindings = {TierName.CORE: TierBinding(model="pro", default_effort="xhigh", thinking=True)}
    with pytest.raises(TierResolutionError, match="'sequencer' asks for effort 'high'"):
        resolve_touchpoint("sequencer", catalog, bindings)


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
