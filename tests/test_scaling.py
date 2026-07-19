from __future__ import annotations

import itertools

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


def test_node_signal_promotes_on_mechanical_verify() -> None:
    assert node_signal(_task("mechanical")) == WorkSignal(direction=1)


def test_node_signal_neutral_when_no_verify() -> None:
    assert node_signal(_task(None)) == WorkSignal()


def test_node_signal_neutral_for_other_verify_agent() -> None:
    # a post-planning verify/repair agent name, not the literal "mechanical"
    assert node_signal(_task("code-reviewer")) == WorkSignal()


def test_sequencer_signal_demotes_short_simple_prompt() -> None:
    assert _sequencer_signal("rename this variable") == WorkSignal(direction=-1)


def test_sequencer_signal_neutral_on_multi_step_prompt() -> None:
    assert _sequencer_signal("add a login endpoint and write tests for it") == WorkSignal()


def test_sequencer_signal_neutral_on_long_prompt() -> None:
    long_prompt = " ".join(["word"] * 20)
    assert _sequencer_signal(long_prompt) == WorkSignal()


def test_sequencer_signal_boundary_word_count_demotes() -> None:
    # exactly at the word-count limit, with no multi-step language: demote
    prompt = " ".join(["word"] * 15)
    assert _sequencer_signal(prompt) == WorkSignal(direction=-1)


def test_sequencer_signal_one_over_boundary_stays_neutral() -> None:
    prompt = " ".join(["word"] * 16)
    assert _sequencer_signal(prompt) == WorkSignal()
