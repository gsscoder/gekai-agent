from __future__ import annotations

import pytest

from agent.harness.touchpoints import TOUCHPOINTS, touchpoint
from agent.llm.tiers import TierName, TierPolicy


def test_all_touchpoints_have_unique_names() -> None:
    names = [t.name for t in TOUCHPOINTS]
    assert len(names) == len(set(names))


def test_expected_touchpoints_are_registered() -> None:
    names = {t.name for t in TOUCHPOINTS}
    assert names == {"gate", "estimator", "sequencer", "main-dispatch", "subagent-dispatch", "micro"}


def test_gate_and_micro_are_fast() -> None:
    assert touchpoint("gate").nominal_tier is TierName.FAST
    assert touchpoint("micro").nominal_tier is TierName.FAST


def test_sequencer_is_core() -> None:
    assert touchpoint("sequencer").nominal_tier is TierName.CORE


def test_unknown_touchpoint_raises() -> None:
    with pytest.raises(ValueError, match="unknown touchpoint"):
        touchpoint("nonexistent")


def test_scaled_touchpoints_carry_the_expected_policy() -> None:
    assert touchpoint("main-dispatch").policy == TierPolicy(
        default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE)
    )
    assert touchpoint("subagent-dispatch").policy == TierPolicy(
        default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE)
    )
    assert touchpoint("sequencer").policy == TierPolicy(
        default=TierName.CORE, allowed=(TierName.SUPP, TierName.CORE)
    )


def test_unscaled_touchpoints_have_no_policy() -> None:
    assert touchpoint("gate").policy is None
    assert touchpoint("estimator").policy is None
    assert touchpoint("micro").policy is None
