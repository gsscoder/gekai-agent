"""Coverage for `agent/directive_pump.py` (plan 28 Phase 3): mechanical
domain detection + the budgeted pump of shallow `namespace_directives` into
main's system prompt. Deep, mission-presupposing `Subagent.directives`
(e.g. code-refactorer's "changing a signature... is out of scope") must
never appear here — only the two isolated test-double calls in
test_harness_core.py/test_harness_stream_plan.py prove the harness wiring;
this file proves the pump function itself.
"""

from __future__ import annotations

import pytest

from agent.directive_pump import PUMP_BUDGET, detect_domains, pump
from agent.subagents import NAMESPACE_DIRECTIVES


def test_detect_domains_from_backtick_path_extension() -> None:
    assert detect_domains("update `src/app/foo.py` to fix the bug") == {"coding"}


def test_detect_domains_from_test_path_adds_testing_alongside_coding() -> None:
    assert detect_domains("update `tests/test_foo.py`") == {"coding", "testing"}


def test_detect_domains_from_keyword_with_no_file_path() -> None:
    """No located files (bare prose ask) — keyword lexicon only."""
    assert detect_domains("write a pytest for the parser") == {"testing"}


def test_detect_domains_empty_for_unrelated_prose() -> None:
    assert detect_domains("what's the weather like today") == set()


def test_pump_returns_empty_for_no_detected_domain() -> None:
    text, domains = pump("what's the weather like today")
    assert text == ""
    assert domains == []


def test_pump_python_turn_yields_coding_craft_only() -> None:
    text, domains = pump("fix the bug in `src/app/foo.py`")
    assert domains == ["coding"]
    assert text == NAMESPACE_DIRECTIVES["coding"]
    # a mission-deep directive (e.g. code-refactorer's) must never leak in
    assert "changing a signature" not in text


def test_pump_multi_domain_turn_respects_budget() -> None:
    text, domains = pump("add `tests/test_foo.py` covering `src/app/foo.py`")
    assert len(domains) <= PUMP_BUDGET
    assert set(domains) <= {"coding", "testing"}
    for domain in domains:
        assert NAMESPACE_DIRECTIVES[domain] in text


def test_pump_budget_cutoff_keeps_lowest_rank_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """coding (rank 1) outranks testing (rank 2); budget=1 must drop testing."""
    import agent.directive_pump as directive_pump

    monkeypatch.setattr(directive_pump, "PUMP_BUDGET", 1)
    text, domains = directive_pump.pump("add `tests/test_foo.py` covering `src/app/foo.py`")
    assert domains == ["coding"]
    assert text == NAMESPACE_DIRECTIVES["coding"]


def test_pump_never_includes_subagent_specific_directives() -> None:
    """namespace_directives (escaping) is craft-only; Subagent.directives
    (confined, per-role) is a disjoint field the pump never reads."""
    text, _ = pump("refactor `src/app/foo.py` and add `tests/test_foo.py`")
    assert "out of scope" not in text
