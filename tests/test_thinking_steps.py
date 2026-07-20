"""Coverage for `_split_thinking_steps`, the TUI's pure sentence-segmentation
helper for the rolling thinking-preview window (agent/tui/app.py).
"""

from __future__ import annotations

from agent.tui.app import _split_thinking_steps


def test_no_terminator_yields_no_steps_and_full_tail() -> None:
    steps, tail = _split_thinking_steps("still working through this")
    assert steps == []
    assert tail == "still working through this"


def test_single_completed_sentence() -> None:
    steps, tail = _split_thinking_steps("first I'll check the file. ")
    assert steps == ["first I'll check the file."]
    assert tail == ""


def test_multiple_completed_sentences_and_leftover_tail() -> None:
    steps, tail = _split_thinking_steps("step one. step two! then step three")
    assert steps == ["step one.", "step two!"]
    assert tail == "then step three"


def test_question_mark_terminates_a_step() -> None:
    steps, tail = _split_thinking_steps("is this right? ")
    assert steps == ["is this right?"]
    assert tail == ""


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
