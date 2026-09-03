"""Assignment-time tier scaling (plan 28 Phase 2).

Lets the harness move a touchpoint's operating point within its declared
tier space (`TierPolicy.allowed`) at dispatch time, driven by an engineered
signal about the shape of the work — never by asking the model to grade its
own difficulty (decision 7). The signal is stateless and per-dispatch
(decision 6): it is computed fresh for one dispatch and never accumulated or
carried across turns. `scale()` is the pure decision function; touchpoints
supply the `WorkSignal`, dispatch applies the returned tier.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..tiers.catalog import TierName, TierPolicy
from ..pipeline.plan import Task


@dataclass(frozen=True)
class WorkSignal:
    """The engineered (never model-self-assessed — plan 28 decision 7) signal
    that tells the harness whether a component's configured default tier is
    adequate for one dispatch. Stateless and per-dispatch (decision 6) — never
    carried across turns."""
    direction: int = 0  # +1 = reasoning-shaped/promote toward CORE, -1 = clearly-easy/demote toward FAST, 0 = neutral
    retry: int = 0      # verify-failure retries: each adds +1 promotion on top of direction


def scale(policy: TierPolicy, signal: WorkSignal) -> tuple[TierName, str]:
    """Move `policy.default` within `policy.allowed` by `signal`, clamped to
    the declared space. Pure: no I/O, no randomness, no state."""
    ladder = policy.allowed
    i = ladder.index(policy.default)
    delta = signal.direction + signal.retry
    desired = i + delta
    j = max(0, min(len(ladder) - 1, desired))

    if delta == 0:
        reason = "default"
    elif j != desired:
        # clamped: the signal wanted to move further than the declared space allows
        reason = f"promote (clamped at {ladder[j].value})" if delta > 0 else f"demote (clamped at {ladder[j].value})"
    elif signal.direction and signal.retry:
        reason = f"reasoning-shaped+retry{signal.retry:+d}"
    elif signal.retry:
        reason = f"retry+{signal.retry}"
    elif signal.direction > 0:
        reason = "reasoning-shaped"
    else:
        reason = "easy-demote"

    return ladder[j], reason


def node_signal(step: Task) -> WorkSignal:
    """Mechanical (no LLM call, decision 7): the sequencer already marks a
    step `verify == "mechanical"` when its own complexity warrants a
    preventive check (see sequencer.py) — reuse that judgment as the
    reasoning-shaped promotion signal, rather than parsing the free-form
    `mission` prose, which has no fixed vocabulary to match against."""
    return WorkSignal(direction=1) if step.verify == "mechanical" else WorkSignal()


# First-pass heuristic only — a placeholder until plan 27's/plan 28's shared
# complexity metric (open point 1) is designed properly. Not a tuned
# classifier: a short word-count cutoff plus a small, explicit keyword list
# for multi-step/plural-scope language. Deliberately conservative (only
# ever demotes, never promotes) per this project's stated risk tolerance —
# a wrong demote strips CORE-thinking from the sequencer before the task
# graph even exists, so any ambiguity defaults to neutral.
_SEQUENCER_WORD_LIMIT = 15
_SEQUENCER_MULTI_STEP_WORDS = frozenset(
    {"and", "also", "then", "multiple", "all", "both", "additionally", "next"}
)


def _sequencer_signal(user_input: str) -> WorkSignal:
    """Pre-plan signal for the sequencer touchpoint: runs before the task
    graph exists, so there is no `Task.verify` to read yet — it must work
    off the raw user prompt text alone. Only demotes (direction=-1) on a
    request that reads clearly, mechanically simple (short AND no
    multi-step/plural-scope language); everything else, including any
    ambiguous case, stays neutral."""
    words = user_input.split()
    if len(words) > _SEQUENCER_WORD_LIMIT:
        return WorkSignal()
    lowered = {w.strip(".,!?;:").lower() for w in words}
    if lowered & _SEQUENCER_MULTI_STEP_WORDS:
        return WorkSignal()
    return WorkSignal(direction=-1)


__all__ = ["WorkSignal", "scale", "node_signal"]
