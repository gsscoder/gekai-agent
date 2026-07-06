"""Coverage for `_route_decision`, the TUI's pure Route -> debug-label mapper.

Source: agent/tui/app.py (plan 27 improvement 4 — `Route` drops `subagent`
and `rejected`; the gate is a pure chit-chat/act binary. A `/agent-x` seed
is tracked separately from `Route` now, at the call site, not on the Route
object itself — see `_stream`'s `forced_seed` handling.
"""

from __future__ import annotations

from agent.pipeline.gate import Route
from agent.tui.app import _route_decision


def test_route_decision_trivial() -> None:
    assert _route_decision(Route(trivial=True)) == "trivial"


def test_route_decision_default_route_is_act() -> None:
    assert _route_decision(Route()) == "act"
