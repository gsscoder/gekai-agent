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


def test_detect_languages_finds_extension_in_task_text(tmp_path) -> None:
    assert directive_pump.detect_languages("fix agent/session.py", tmp_path) == ["python"]


def test_detect_languages_falls_back_to_manifest_when_no_extension_hit(tmp_path) -> None:
    (tmp_path / "pyproject.toml").touch()
    assert directive_pump.detect_languages("say hi", tmp_path) == ["python"]


def test_detect_languages_returns_empty_when_neither_signal_fires(tmp_path) -> None:
    assert directive_pump.detect_languages("say hi", tmp_path) == []


def test_detect_languages_ignores_unrecognized_extension(tmp_path) -> None:
    assert directive_pump.detect_languages("rename the .rs handler", tmp_path) == []


def test_detect_languages_truncates_to_budget_by_count_then_name(tmp_path, monkeypatch) -> None:
    """More hit languages are found than LANGUAGE_BUDGET allows; only the
    top-N survive, ranked by hit count descending then name ascending."""
    monkeypatch.setattr(directive_pump, "_EXT_TO_LANG", {
        ".aa": "alpha", ".bb": "beta", ".cc": "gamma",
    })
    monkeypatch.setattr(directive_pump, "LANGUAGE_BUDGET", 2)
    task = "touch file.aa and file.bb again file.bb and file.cc"
    assert directive_pump.detect_languages(task, tmp_path) == ["beta", "alpha"]


def test_detect_languages_stays_linear_on_long_extensionless_task(tmp_path) -> None:
    """`task` is unbounded, user-controlled text — a long run with no
    extension anywhere must not blow up. An earlier regex-based
    implementation degraded to O(n^2) on exactly this shape (every one of
    n starting positions cost O(n) to rule out), so this asserts on wall
    time, not just the result, to guard against that regression coming
    back in a different form."""
    import time

    task = "a" * 200_000
    start = time.monotonic()
    assert directive_pump.detect_languages(task, tmp_path) == []
    assert time.monotonic() - start < 1.0
