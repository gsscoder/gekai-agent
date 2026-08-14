"""Harness touchpoint registry (plan 28 Phase 0).

Enumerates every place the harness invokes a model — the operational
definition of "the harness" (concept 1, plan 28): all engineered, non-LLM,
per-turn machinery; root is one configuration of the LLM loop the harness
deploys, not the harness itself. Adding a model interaction means adding an
entry here, not scattering a new client call into a module.

Inert: not yet consulted by any dispatch path. Phase 1 wires each entry's
tier binding to the global tier config; Phase 2 lets the harness move a
touchpoint's operating point within its declared space at assignment time.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.tiers import TierName, TierPolicy


@dataclass(frozen=True)
class Touchpoint:
    name: str
    job: str
    nominal_tier: TierName
    # Assignment-time scaling (Phase 2): a component's declared tier mobility.
    # None for the two touchpoints not scaled in v1 (estimator, micro) — they
    # run at a bare `nominal_tier` with no mobility. When set, `policy.default`
    # is expected to equal `nominal_tier`.
    policy: TierPolicy | None = None
    # The touchpoint's own operating point, applied on top of whatever tier it
    # resolves at: the tier binding says WHICH model + credentials fill the
    # capability slot (user config, /tier), the touchpoint says HOW to operate
    # that model for this particular job (harness engineering, code). `None` =
    # inherit the binding's value, which is what every touchpoint but the
    # sequencer does. Never used to build the credential key — that stays keyed
    # on the binding's configured operating point (see resolve.py).
    effort: str | None = None
    thinking: bool | None = None


TOUCHPOINTS: tuple[Touchpoint, ...] = (
    Touchpoint("estimator", "trivial vs mutate", TierName.FAST),
    Touchpoint(
        "sequencer",
        "build the task graph (formerly 'planner')",
        TierName.CORE,
        policy=TierPolicy(default=TierName.CORE, allowed=(TierName.SUPP, TierName.CORE)),
        # Sequencing is a short, structured JSON emission, not open-ended
        # reasoning: a live probe (real credentials, real sequencer prompt)
        # measured thinking-ON at a median 90.7s (10k-26k reasoning chars)
        # against thinking-OFF at 13.2s with equally good — occasionally
        # better — task graphs. Inheriting CORE's binding wholesale made
        # planning cost about as much as the execution it plans, and the only
        # user-facing fix was toggling CORE's thinking flag, which is a model-
        # capability knob, not a per-workload one. So the sequencer declares
        # its own operating point here while still taking CORE's model+creds.
        effort="high",
        thinking=False,
    ),
    Touchpoint(
        "root-dispatch",
        "run root on a task",
        TierName.SUPP,
        policy=TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE)),
    ),
    Touchpoint(
        "subagent-dispatch",
        "run a specialist cold",
        TierName.SUPP,
        policy=TierPolicy(default=TierName.SUPP, allowed=(TierName.SUPP, TierName.CORE)),
    ),
    Touchpoint("micro", "one-shot summaries / labels / fs-support", TierName.FAST),
    Touchpoint(
        "directive-audit",
        "yes/no: does a markdown file contain agent directives",
        TierName.FAST,
    ),
)

TOUCHPOINTS_BY_NAME: dict[str, Touchpoint] = {t.name: t for t in TOUCHPOINTS}

if len(TOUCHPOINTS_BY_NAME) != len(TOUCHPOINTS):
    raise ValueError("duplicate touchpoint name in TOUCHPOINTS")


def touchpoint(name: str) -> Touchpoint:
    try:
        return TOUCHPOINTS_BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown touchpoint {name!r}; known: {sorted(TOUCHPOINTS_BY_NAME)}") from None


__all__ = ["Touchpoint", "TOUCHPOINTS", "TOUCHPOINTS_BY_NAME", "touchpoint"]
