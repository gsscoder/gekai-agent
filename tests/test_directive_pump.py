"""Coverage for `agent/directive_pump.py` (plan 28 Phase 3): the budgeted,
unconditional pump of shallow `namespace_directives` into main's system
prompt. Deep, mission-presupposing `Subagent.directives` (e.g.
code-refactorer's "changing a signature... is out of scope") must never
appear here — only the two isolated test-double calls in
test_harness_core.py/test_harness_stream_plan.py prove the harness wiring;
this file proves the pump function itself.
"""

from __future__ import annotations

import agent.directive_pump as directive_pump
from agent.directive_pump import pump


def test_pump_returns_empty_when_no_namespaces_registered(monkeypatch) -> None:
    monkeypatch.setattr(directive_pump, "NAMESPACE_DIRECTIVES", {})
    monkeypatch.setattr(directive_pump, "NAMESPACE_DIRECTIVE_RANK", {})
    text, domains = pump()
    assert text == ""
    assert domains == []


def test_pump_truncates_to_budget_by_rank(monkeypatch) -> None:
    """More namespaces are registered than PUMP_BUDGET allows; only the
    lowest-rank ones survive."""
    monkeypatch.setattr(directive_pump, "NAMESPACE_DIRECTIVES", {
        "a": "A directive", "b": "B directive", "c": "C directive",
    })
    monkeypatch.setattr(directive_pump, "NAMESPACE_DIRECTIVE_RANK", {"a": 1, "b": 2, "c": 3})
    monkeypatch.setattr(directive_pump, "PUMP_BUDGET", 2)
    text, domains = pump()
    assert domains == ["a", "b"]
    assert text == "A directive\nB directive"


def test_pump_tie_breaks_equal_rank_by_name(monkeypatch) -> None:
    """Namespaces with no explicit rank all default to the same rank
    (100); the tie is then broken alphabetically by name."""
    monkeypatch.setattr(directive_pump, "NAMESPACE_DIRECTIVES", {
        "zeta": "Z directive", "alpha": "A directive", "beta": "B directive",
    })
    monkeypatch.setattr(directive_pump, "NAMESPACE_DIRECTIVE_RANK", {})
    monkeypatch.setattr(directive_pump, "PUMP_BUDGET", 2)
    text, domains = pump()
    assert domains == ["alpha", "beta"]
    assert text == "A directive\nB directive"


def test_pump_explicit_rank_outranks_alphabetical_default(monkeypatch) -> None:
    """An explicitly low-ranked namespace is chosen over ones sorting
    earlier alphabetically but left at the default rank."""
    monkeypatch.setattr(directive_pump, "NAMESPACE_DIRECTIVES", {
        "alpha": "A directive", "beta": "B directive", "zeta": "Z directive",
    })
    monkeypatch.setattr(directive_pump, "NAMESPACE_DIRECTIVE_RANK", {"zeta": 0})
    monkeypatch.setattr(directive_pump, "PUMP_BUDGET", 2)
    text, domains = pump()
    assert domains == ["zeta", "alpha"]
    assert text == "Z directive\nA directive"
