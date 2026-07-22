"""Coverage for the TUI's pure status-bar formatting helpers (agent/tui/app.py)."""

from __future__ import annotations

import pytest

from agent.tui.app import (
    _fmt_status_left,
    _fmt_status_right,
    _fmt_tokens_k,
    _truncate_path_middle,
)


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, "0.0k"),
        (300, "0.3k"),
        (1234, "1.2k"),
        (45700, "45.7k"),
    ],
)
def test_fmt_tokens_k(n: int, expected: str) -> None:
    assert _fmt_tokens_k(n) == expected


@pytest.mark.parametrize(
    "path, budget, expected",
    [
        ("short/path", 30, "short/path"),
        ("x" * 30, 30, "x" * 30),
        (
            "C:/Users/Coder/MEGA_Repos/private_repos/gekai-agent",
            30,
            "C:/Users/Coder…pos/gekai-agent",
        ),
    ],
)
def test_truncate_path_middle(path: str, budget: int, expected: str) -> None:
    result = _truncate_path_middle(path, budget)
    assert result == expected
    if len(path) > budget:
        assert result.endswith(path[-1])


def test_fmt_status_left_includes_both_token_figures_and_pct() -> None:
    line = _fmt_status_left("deepseek-v4-flash", "high", 1234, 5600, 1234, 128_000)
    assert line == "\\[deepseek-v4-flash (high)] | 1.2k · 5.6k tokens | 1.0% context"


def test_fmt_status_left_omits_effort_when_absent() -> None:
    line = _fmt_status_left("deepseek-v4-flash", None, 0, 0, 0, 128_000)
    assert line.startswith("\\[deepseek-v4-flash]")


def test_fmt_status_right_with_branch() -> None:
    assert _fmt_status_right("/repo/gekai-agent", "main") == "📁 /repo/gekai-agent | ⎇ main"


def test_fmt_status_right_without_branch() -> None:
    assert _fmt_status_right("/repo/gekai-agent", None) == "📁 /repo/gekai-agent"
