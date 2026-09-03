from __future__ import annotations

import pytest

from agent.tiers.catalog import (
    TierName,
    TierPolicy,
    TierSuitability,
    suitability,
)


@pytest.mark.parametrize(
    "default, allowed",
    [
        # A one-tier space (a future file-explorer's allowed=(fast,)) has
        # exactly one legal tier — the whole mobility model, no separate flag.
        (TierName.FAST, (TierName.FAST,)),
        (TierName.SUPP, (TierName.SUPP, TierName.CORE)),
    ],
)
def test_policy_accepts_default_within_allowed(default: TierName, allowed: tuple[TierName, ...]) -> None:
    policy = TierPolicy(default=default, allowed=allowed)
    assert policy.allowed == allowed
    assert policy.default is default


@pytest.mark.parametrize(
    "default, allowed, match",
    [
        (TierName.CORE, (TierName.FAST, TierName.SUPP), "is not in allowed tiers"),
        (TierName.FAST, (TierName.CORE, TierName.FAST), "ascending"),
        (TierName.FAST, (TierName.FAST, TierName.FAST), "ascending"),
    ],
)
def test_policy_rejects_invalid_configuration(
    default: TierName, allowed: tuple[TierName, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        TierPolicy(default=default, allowed=allowed)


@pytest.mark.parametrize(
    "model, tier, expected",
    [
        ("some-unlisted-model", TierName.CORE, "ok"),
        # deepseek-v4-flash is the FAST/SUPP-tier model; deepseek-v4-pro can
        # disable thinking (confirmed by live probe) but is still rated
        # deprecated/warning there pending a cost/latency evaluation.
        ("deepseek-v4-flash", TierName.FAST, "ok"),
        ("deepseek-v4-flash", TierName.CORE, "warning"),
        ("deepseek-v4-pro", TierName.FAST, "deprecated"),
        ("deepseek-v4-pro", TierName.CORE, "ok"),
    ],
)
def test_suitability_known_model_verdicts(model: str, tier: TierName, expected: str) -> None:
    assert suitability(model, tier) == expected


def test_suitability_thinking_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.tiers.catalog import MODEL_SUITABILITY
    monkeypatch.setitem(
        MODEL_SUITABILITY, "synthetic-model",
        TierSuitability(core="warning", core_thinking="ok"),
    )
    assert suitability("synthetic-model", TierName.CORE, thinking=False) == "warning"
    assert suitability("synthetic-model", TierName.CORE, thinking=True) == "ok"


def test_suitability_no_thinking_override_falls_back_to_core() -> None:
    rated = TierSuitability(core="warning")
    assert rated.verdict(TierName.CORE, thinking=True) == "warning"
