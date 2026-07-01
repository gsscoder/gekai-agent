"""Coverage for `_route_decision`, the TUI's pure Route -> debug-label mapper.

Source: agent/tui/app.py (plan 25 Improvement 1 — gate replaces router).
After plan 25, the function checks: rejected > subagent > trivial > "act".
"query" and "plan(n)" labels are gone; the default is now "act".
"""

from __future__ import annotations

from agent.pipeline.gate import Route
from agent.subagents import SUBAGENTS, Subagent
from agent.tui.app import _route_decision

_CODE_EXPERT = next(p for p in SUBAGENTS if p.name == "code-expert")


def test_route_decision_subagent_uses_namespace_and_name() -> None:
    assert _route_decision(Route(subagent=_CODE_EXPERT)) == "coding/code-expert"


def test_route_decision_subagent_with_constructed_instance() -> None:
    sub = Subagent(name="custom-agent", namespace="generic", description="d")
    assert _route_decision(Route(subagent=sub)) == "generic/custom-agent"


def test_route_decision_trivial() -> None:
    assert _route_decision(Route(trivial=True)) == "trivial"


def test_route_decision_default_route_is_act() -> None:
    # plan 25: default (ACT) route maps to "act", not "main" or "query"
    assert _route_decision(Route()) == "act"


def test_route_decision_subagent_takes_precedence_over_trivial() -> None:
    route = Route(subagent=_CODE_EXPERT, trivial=True)
    assert _route_decision(route) == "coding/code-expert"


def test_route_decision_rejected() -> None:
    assert _route_decision(Route(rejected=True)) == "rejected"


def test_route_decision_rejected_takes_precedence_over_trivial() -> None:
    assert _route_decision(Route(rejected=True, trivial=True)) == "rejected"


def test_route_decision_rejected_takes_precedence_over_subagent() -> None:
    assert _route_decision(Route(rejected=True, subagent=_CODE_EXPERT)) == "rejected"
