"""Harness touchpoint registry (plan 28 Phase 0).

Enumerates every place the harness invokes a model — the operational
definition of "the harness" (concept 1, plan 28): all engineered, non-LLM,
per-turn machinery; main is one configuration of the LLM loop the harness
deploys, not the harness itself. Adding a model interaction means adding an
entry here, not scattering a new client call into a module.

Inert: not yet consulted by any dispatch path. Phase 1 wires each entry's
tier binding to the global /tiers config; Phase 2 lets the harness move a
touchpoint's operating point within its declared space at assignment time.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.tiers import TierName


@dataclass(frozen=True)
class Touchpoint:
    name: str
    job: str
    nominal_tier: TierName


TOUCHPOINTS: tuple[Touchpoint, ...] = (
    Touchpoint("gate", "chit-chat vs act (one token)", TierName.FAST),
    Touchpoint("estimator", "trivial vs mutate", TierName.FAST),
    Touchpoint("sequencer", "build the task graph (formerly 'planner')", TierName.CORE),
    Touchpoint("main-dispatch", "run main on a task", TierName.SUPP),
    Touchpoint("subagent-dispatch", "run a specialist cold", TierName.SUPP),
    Touchpoint("micro", "one-shot summaries / labels / fs-support", TierName.FAST),
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
