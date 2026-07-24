"""Assignment-time tool scoping (plan 31 Phase 1).

Lets a task-graph step narrow a unit's (main's or a subagent's) tool grant
below its `ToolPolicy.ceiling` for one dispatch, driven by the sequencer's
per-step scope classification — never by asking the model to grade its own
tool needs (mirrors plan 28 decision 7). Tighten-only: a unit never gets
more tools than its ceiling declares, regardless of what a step requests.

`scope()` is the pure decision function; a later phase wires the result
into dispatch.

Inert: not consulted by any dispatch path yet — this module defines the
shape only.
"""

from __future__ import annotations

from ..subagents import ToolPolicy
from ..tools.catalog import RUNGS

_SCOPE_TO_RUNG: dict[str, int] = {"read": 0, "edit": 1, "fs": 2}

# main's declared ceiling: full — tighten-only, never widened beyond this
MAIN_TOOL_POLICY = ToolPolicy(ceiling=len(RUNGS) - 1)


def scope(policy: ToolPolicy | None, step_scope: str | None) -> tuple[frozenset[str], str]:
    """Resolve one task-graph step's effective tool set for a unit (main or a subagent).

    Tighten-only: a unit never gets more than its ceiling (RUNGS[-1]/full if policy is None).
    step_scope is the sequencer's per-step classification ("read"/"edit"/"fs") or None —
    absent/unrecognized values are neutral (unit's ceiling, unchanged). Pure: no I/O."""
    ceiling = policy.ceiling if policy is not None else len(RUNGS) - 1
    if step_scope not in _SCOPE_TO_RUNG:
        return frozenset(RUNGS[ceiling]), "default"
    requested = _SCOPE_TO_RUNG[step_scope]
    if requested >= ceiling:
        return frozenset(RUNGS[ceiling]), "default"
    return frozenset(RUNGS[requested]), f"narrowed to {step_scope!r}"


__all__ = ["ToolPolicy", "scope", "MAIN_TOOL_POLICY"]
