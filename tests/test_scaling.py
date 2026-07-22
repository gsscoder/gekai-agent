from __future__ import annotations

import itertools

import pytest

from agent.harness.scaling import WorkSignal, _sequencer_signal, node_signal, scale
from agent.llm.tiers import TierName, TierPolicy
from agent.pipeline.plan import Task


def test_neutral_signal_returns_default() -> None:
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.FAST, TierName.SUPP, TierName.CORE))
    tier, reason = scale(policy, WorkSignal())
    assert tier is TierName.SUPP
    assert reason == "default"


def test_retry_promotes() -> None:
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE))
    tier, reason = scale(policy, WorkSignal(retry=1))
    assert tier is TierName.CORE
    assert "retry" in reason


def test_direction_promotes_one_tier() -> None:
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.FAST, TierName.SUPP, TierName.CORE))
    tier, reason = scale(policy, WorkSignal(direction=1))
    assert tier is TierName.CORE
    assert reason == "reasoning-shaped"


def test_direction_demotes_one_tier() -> None:
    policy = TierPolicy(default=TierName.CORE, allowed=(TierName.FAST, TierName.SUPP, TierName.CORE))
    tier, reason = scale(policy, WorkSignal(direction=-1))
    assert tier is TierName.SUPP
    assert reason == "easy-demote"


def test_degenerate_policy_never_moves() -> None:
    policy = TierPolicy(default=TierName.FAST, allowed=(TierName.FAST,))
    for signal in (
        WorkSignal(),
        WorkSignal(direction=1),
        WorkSignal(direction=-1),
        WorkSignal(retry=5),
        WorkSignal(direction=1, retry=5),
        WorkSignal(direction=-1, retry=5),
    ):
        tier, _ = scale(policy, signal)
        assert tier is TierName.FAST


def test_policy_missing_core_clamps_at_top_allowed_tier() -> None:
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.FAST, TierName.SUPP))
    tier, reason = scale(policy, WorkSignal(direction=1, retry=3))
    assert tier is TierName.SUPP
    assert "clamped" in reason


def test_combined_signal_moves_two_steps_then_clamps() -> None:
    policy = TierPolicy(default=TierName.FAST, allowed=(TierName.FAST, TierName.SUPP, TierName.CORE))
    tier, reason = scale(policy, WorkSignal(direction=1, retry=1))
    assert tier is TierName.CORE
    assert "clamped" not in reason

    # one more step of delta would clamp at CORE (top of the ladder)
    tier, reason = scale(policy, WorkSignal(direction=1, retry=2))
    assert tier is TierName.CORE
    assert "clamped" in reason


def test_scale_never_raises_and_always_returns_allowed_member() -> None:
    policy = TierPolicy(default=TierName.SUPP, allowed=(TierName.FAST, TierName.SUPP, TierName.CORE))
    directions = range(-5, 6)
    retries = range(-5, 6)
    for direction, retry in itertools.product(directions, retries):
        tier, reason = scale(policy, WorkSignal(direction=direction, retry=retry))
        assert tier in policy.allowed
        assert isinstance(reason, str)


def _task(verify: str | None) -> Task:
    return Task(agent="main", instruction="do the thing", mission="do the thing quickly", verify=verify)


@pytest.mark.parametrize(
    "verify,expected",
    [
        ("mechanical", WorkSignal(direction=1)),
        (None, WorkSignal()),
        # a post-planning verify/repair agent name, not the literal "mechanical"
        ("code-reviewer", WorkSignal()),
    ],
)
def test_node_signal(verify: str | None, expected: WorkSignal) -> None:
    assert node_signal(_task(verify)) == expected


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("rename this variable", WorkSignal(direction=-1)),
        ("add a login endpoint and write tests for it", WorkSignal()),
        (" ".join(["word"] * 20), WorkSignal()),
        # exactly at the word-count limit, with no multi-step language: demote
        (" ".join(["word"] * 15), WorkSignal(direction=-1)),
        (" ".join(["word"] * 16), WorkSignal()),
    ],
)
def test_sequencer_signal(prompt: str, expected: WorkSignal) -> None:
    assert _sequencer_signal(prompt) == expected
