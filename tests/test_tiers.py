from __future__ import annotations

import pytest

from agent.llm.tiers import (
    TierLimit,
    TierName,
    TierPoint,
    TierPolicy,
    TierSuitability,
    effort_in_range,
    suitability,
)


def test_effort_in_range() -> None:
    assert effort_in_range("medium", "low", "high") is True
    assert effort_in_range("low", "medium", "high") is False
    assert effort_in_range("max", "low", "high") is False


def test_tier_limit_rejects_thinking_outside_core() -> None:
    with pytest.raises(ValueError, match="CORE-only"):
        TierLimit(tier=TierName.FAST, min_effort="low", max_effort="high", thinking=True)


def test_tier_limit_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="above"):
        TierLimit(tier=TierName.SUPP, min_effort="high", max_effort="low")


def test_tier_limit_allows_within_range() -> None:
    lim = TierLimit(tier=TierName.CORE, min_effort="low", max_effort="high", thinking=True)
    assert lim.allows("medium") is True
    assert lim.allows("medium", thinking=True) is True
    assert lim.allows("max") is False  # outside range


def test_tier_point_rejects_thinking_outside_core() -> None:
    with pytest.raises(ValueError, match="CORE-only"):
        TierPoint(tier=TierName.SUPP, effort="high", thinking=True)


def test_degenerate_space_is_non_scalable() -> None:
    """A one-point space (a future file-explorer's tiers=[fast(low)]) has
    exactly one legal point — the whole mobility model, no separate flag."""
    policy = TierPolicy(
        default=TierPoint(tier=TierName.FAST, effort="low"),
        limits=(TierLimit(tier=TierName.FAST, min_effort="low", max_effort="low"),),
    )
    assert policy.allows(TierPoint(tier=TierName.FAST, effort="low")) is True
    assert policy.allows(TierPoint(tier=TierName.FAST, effort="medium")) is False
    assert policy.allows(TierPoint(tier=TierName.SUPP, effort="low")) is False


def test_multi_tier_space_allows_movement_within_declared_limits() -> None:
    policy = TierPolicy(
        default=TierPoint(tier=TierName.SUPP, effort="medium"),
        limits=(
            TierLimit(tier=TierName.SUPP, min_effort="medium", max_effort="high"),
            TierLimit(tier=TierName.CORE, min_effort="low", max_effort="high", thinking=True),
        ),
    )
    assert policy.allows(TierPoint(tier=TierName.SUPP, effort="high")) is True
    assert policy.allows(TierPoint(tier=TierName.CORE, effort="low", thinking=True)) is True
    assert policy.allows(TierPoint(tier=TierName.FAST, effort="low")) is False  # not in declared limits


def test_policy_rejects_unrealizable_default() -> None:
    with pytest.raises(ValueError, match="not realizable"):
        TierPolicy(
            default=TierPoint(tier=TierName.CORE, effort="max"),
            limits=(TierLimit(tier=TierName.CORE, min_effort="low", max_effort="high"),),
        )


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
