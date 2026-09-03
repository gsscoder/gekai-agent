"""Dynamic directive pump (plan 28 Phase 3): root borrows domain *expertise*
without a cold spawn borrowing the *role* (decision 13). Only the shallow,
mission-free `namespace_directives` groups (`agent/subagents/*/__init__.py`)
ever reach here — a `Subagent.directives` (deep, mission-presupposing) is
never pumped, by construction: it is only assembled inside
`Subagent.build_system_base()`, which this module never calls.

Directives are pumped unconditionally into every eligible turn, rather than
being detected from the prompt's text shape. The original design detected
domains from backtick-quoted file paths and a keyword lexicon in the raw
prompt — but the backtick shape was only ever produced by `PromptRewriter`,
which is never constructed anywhere in the codebase (dead code). That made
detection a silent no-op on any plain-English request: zero domains
detected, zero directives pumped, even though root was about to write code.
Eligibility is now decided by the caller instead: root gets the pump on
every turn where it has tool access to write code, and skips it only on the
"chat" rung (pure greeting/chit-chat, no code involved -- the rung still
carries root's normal tool schemas, it just never needs domain expertise).

Open point 2 (escape-rank representation) is settled as: rank is a small
int per namespace group (`namespace_directive_rank`, lower = higher
priority), authored next to that namespace's `namespace_directives`. The
budget is a cap on *domains* pumped, not individual directive lines — each
namespace's directives are already a short, cohesive craft block, so
ranking at that granularity is sufficient and keeps authoring in the same
plain-string convention the codebase already uses. No conflict-resolution
step is needed beyond the budget cutoff: escaping directives are mission-
free by construction (decision 11), so they cannot goal-conflict, only
style-conflict, which is low-stakes and left for the model to reconcile.
"""

from __future__ import annotations

from .subagents import NAMESPACE_DIRECTIVES, NAMESPACE_DIRECTIVE_RANK

PUMP_BUDGET = 2  # cap on domains pumped into root per turn (hard problem 3)


def pump() -> tuple[str, list[str]]:
    """Assemble the escaping directives for the top-`PUMP_BUDGET` registered
    namespaces, ranked by `namespace_directive_rank`. Returns
    (directive_text, domains_used) — domains_used is "" / [] only when no
    namespace directives are registered at all."""
    if not NAMESPACE_DIRECTIVES:
        return "", []
    chosen = sorted(
        NAMESPACE_DIRECTIVES.keys(),
        key=lambda d: (NAMESPACE_DIRECTIVE_RANK.get(d, 100), d),
    )[:PUMP_BUDGET]
    return "\n".join(NAMESPACE_DIRECTIVES[d] for d in chosen), chosen
