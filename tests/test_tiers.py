from __future__ import annotations

import pytest

from agent.llm.tiers import (
    TierName,
    TierPolicy,
    TierSuitability,
    suitability,
)


def test_degenerate_policy_is_non_scalable() -> None:
    """A one-tier space (a future file-explorer's allowed=(fast,)) has
    exactly one legal tier — the whole mobility model, no separate flag."""
    policy = TierPolicy(default=TierName.FAST, allowed=(TierName.FAST,))
    assert policy.allowed == (TierName.FAST,)
    assert policy.default is TierName.FAST


def test_multi_tier_policy_allows_movement_within_declared_tiers() -> None:
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE))
    assert policy.allowed == (TierName.SUPP, TierName.CORE)
    assert policy.default is TierName.SUPP


def test_policy_rejects_default_not_in_allowed() -> None:
    with pytest.raises(ValueError, match="is not in allowed tiers"):
        TierPolicy(default=TierName.CORE, allowed=(TierName.FAST, TierName.SUPP))


def test_policy_rejects_non_ascending_allowed() -> None:
    with pytest.raises(ValueError, match="ascending"):
        TierPolicy(default=TierName.FAST, allowed=(TierName.CORE, TierName.FAST))


def test_policy_rejects_duplicate_allowed() -> None:
    with pytest.raises(ValueError, match="ascending"):
        TierPolicy(default=TierName.FAST, allowed=(TierName.FAST, TierName.FAST))


def test_suitability_defaults_to_ok_for_unlisted_model() -> None:
    assert suitability("some-unlisted-model", TierName.CORE) == "ok"


def test_suitability_known_model_verdicts() -> None:
    # deepseek-v4-flash is the FAST/SUPP-tier model, deepseek-v4-pro is
    # reasoning-only (can't disable thinking) so FAST/SUPP are deprecated.
    assert suitability("deepseek-v4-flash", TierName.FAST) == "ok"
    assert suitability("deepseek-v4-flash", TierName.CORE) == "warning"
    assert suitability("deepseek-v4-pro", TierName.FAST) == "deprecated"
    assert suitability("deepseek-v4-pro", TierName.CORE) == "ok"


def test_suitability_thinking_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.llm.tiers import MODEL_SUITABILITY
    monkeypatch.setitem(
        MODEL_SUITABILITY, "synthetic-model",
        TierSuitability(core="warning", core_thinking="ok"),
    )
    assert suitability("synthetic-model", TierName.CORE, thinking=False) == "warning"
    assert suitability("synthetic-model", TierName.CORE, thinking=True) == "ok"


def test_suitability_no_thinking_override_falls_back_to_core() -> None:
    rated = TierSuitability(core="warning")
    assert rated.verdict(TierName.CORE, thinking=True) == "warning"
