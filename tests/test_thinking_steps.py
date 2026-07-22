"""Coverage for `_split_thinking_steps`, the TUI's pure sentence-segmentation
helper for the rolling thinking-preview window (agent/tui/app.py).
"""

from __future__ import annotations

import pytest

from agent.tui.app import _split_thinking_steps


@pytest.mark.parametrize(
    "text, expected_steps, expected_tail",
    [
        ("still working through this", [], "still working through this"),
        ("first I'll check the file. ", ["first I'll check the file."], ""),
        ("step one. step two! then step three", ["step one.", "step two!"], "then step three"),
        ("is this right? ", ["is this right?"], ""),
    ],
)
def test_split_thinking_steps(text: str, expected_steps: list[str], expected_tail: str) -> None:
    steps, tail = _split_thinking_steps(text)
    assert steps == expected_steps
    assert tail == expected_tail


def test_incremental_calls_accumulate_like_streamed_tokens() -> None:
    """Mirrors how `thinking_chunk` actually calls this: token-by-token,
    re-parsing the tail each time."""
    tail = ""
    all_steps: list[str] = []
    for chunk in ["step ", "one. ", "step ", "two."]:
        tail += chunk
        steps, tail = _split_thinking_steps(tail)
        all_steps.extend(steps)
    assert all_steps == ["step one.", "step two."]
    assert tail == ""
