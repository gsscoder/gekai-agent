"""Coverage for `_route_decision`, the TUI's pure Route -> debug-label mapper.

Source: agent/tui/app.py:268-279 (plan 17, multi-step router).
The function checks fields in this precedence order: plan, subagent,
trivial, explore, else "main". Kept in its own file (rather than
tests/test_router.py, which only imports agent.pipeline.router) to avoid
mixing agent.tui.app imports into the router-only test module.
"""

from __future__ import annotations

from agent.pipeline.router import PlanStep, Route
from agent.subagents import SUBAGENTS, Subagent
from agent.tui.app import _route_decision

# REQ-001: Route(subagent=...) -> "{namespace}/{name}"; use a real registered
# subagent so the test is anchored to the actual registry, not a fabricated one.
_CODE_EXPERT = next(p for p in SUBAGENTS if p.name == "code-expert")


def test_route_decision_plan_reports_step_count() -> None:
    # REQ: Route.plan -> f"plan({len(route.plan)})"; build 2 minimal PlanSteps.
    plan = [
        PlanStep(subagent=None, raw="step one"),
        PlanStep(subagent=_CODE_EXPERT, raw="step two"),
    ]
    assert _route_decision(Route(plan=plan)) == "plan(2)"


def test_route_decision_plan_single_step() -> None:
    # EDGE: a plan with exactly one step still reports its real count (the
    # router itself collapses single-step plans to a bare subagent route
    # before this point, but _route_decision has no such guard, so this
    # exercises the boundary explicitly).
    plan = [PlanStep(subagent=None, raw="only step")]
    assert _route_decision(Route(plan=plan)) == "plan(1)"


def test_route_decision_subagent_uses_namespace_and_name() -> None:
    # REQ: Route.subagent -> f"{namespace}/{name}" using a real Subagent instance.
    assert _route_decision(Route(subagent=_CODE_EXPERT)) == "coding/code-expert"


def test_route_decision_subagent_with_constructed_instance() -> None:
    # EDGE: a manually constructed Subagent (not from the registry) is
    # formatted identically — the function only reads namespace/name.
    sub = Subagent(name="custom-agent", namespace="generic", description="d")
    assert _route_decision(Route(subagent=sub)) == "generic/custom-agent"


def test_route_decision_trivial() -> None:
    # REQ: Route.trivial=True -> "trivial".
    assert _route_decision(Route(trivial=True)) == "trivial"


def test_route_decision_explore() -> None:
    # REQ: Route.explore=True -> "explore".
    assert _route_decision(Route(explore=True)) == "explore"


def test_route_decision_default_route_is_main() -> None:
    # REQ: an all-default Route (handled directly by the coding agent) -> "main".
    assert _route_decision(Route()) == "main"


def test_route_decision_plan_takes_precedence_over_subagent() -> None:
    # REQ: precedence order — plan is checked before subagent.
    plan = [PlanStep(subagent=None, raw="ignored")]
    route = Route(plan=plan, subagent=_CODE_EXPERT)
    assert _route_decision(route) == "plan(1)"


def test_route_decision_subagent_takes_precedence_over_trivial() -> None:
    # REQ: precedence order — subagent is checked before trivial.
    route = Route(subagent=_CODE_EXPERT, trivial=True)
    assert _route_decision(route) == "coding/code-expert"


def test_route_decision_trivial_takes_precedence_over_explore() -> None:
    # REQ: precedence order — trivial is checked before explore.
    route = Route(trivial=True, explore=True)
    assert _route_decision(route) == "trivial"
