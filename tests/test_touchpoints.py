from __future__ import annotations

import pytest

from agent.harness.touchpoints import TOUCHPOINTS, touchpoint
from agent.llm.tiers import TierName, TierPolicy


def test_all_touchpoints_have_unique_names() -> None:
    names = [t.name for t in TOUCHPOINTS]
    assert len(names) == len(set(names))


def test_expected_touchpoints_are_registered() -> None:
    names = {t.name for t in TOUCHPOINTS}
    assert names == {"estimator", "sequencer", "root-dispatch", "subagent-dispatch", "micro", "directive-audit"}


def test_estimator_and_micro_are_fast() -> None:
    assert touchpoint("estimator").nominal_tier is TierName.FAST
    assert touchpoint("micro").nominal_tier is TierName.FAST


def test_sequencer_is_core() -> None:
    assert touchpoint("sequencer").nominal_tier is TierName.CORE


def test_unknown_touchpoint_raises() -> None:
    with pytest.raises(ValueError, match="unknown touchpoint"):
        touchpoint("nonexistent")


def test_scaled_touchpoints_carry_the_expected_policy() -> None:
    assert touchpoint("root-dispatch").policy == TierPolicy(
        default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE)
    )
    assert touchpoint("subagent-dispatch").policy == TierPolicy(
        default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE)
    )
    assert touchpoint("sequencer").policy == TierPolicy(
        default=TierName.CORE, allowed=(TierName.SUPP, TierName.CORE)
    )


def test_unscaled_touchpoints_have_no_policy() -> None:
    assert touchpoint("estimator").policy is None
    assert touchpoint("micro").policy is None


def test_only_the_sequencer_declares_its_own_operating_point() -> None:
    # The sequencer emits a short structured JSON graph — measured 90.7s
    # median at CORE's thinking-on binding vs 13.2s thinking-off for equal or
    # better graphs. Every other touchpoint inherits its binding's operating
    # point (effort/thinking both None), unchanged.
    assert (touchpoint("sequencer").effort, touchpoint("sequencer").thinking) == ("high", False)
    for name in ("estimator", "root-dispatch", "subagent-dispatch", "micro"):
        assert touchpoint(name).effort is None, name
        assert touchpoint(name).thinking is None, name
