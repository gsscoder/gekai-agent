"""Coverage for the TUI's thinking-preview machinery (agent/tui/app.py):
`_split_thinking_steps`, the pure sentence-segmentation helper, and
`_ThinkingLine`, the shared per-turn ticker line it feeds into via
`update_chunk`.
"""

from __future__ import annotations

import pytest

from agent.tui.app import (
    _BRAILLE_FRAMES,
    _THINKING_LINE_CAP,
    _THINKING_LINE_SENTENCE_RESET,
    _ThinkingLine,
    _split_thinking_steps,
)


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
    """Mirrors how `_ThinkingLine.update_chunk` actually calls this:
    token-by-token, re-parsing the tail each time."""
    tail = ""
    all_steps: list[str] = []
    for chunk in ["step ", "one. ", "step ", "two."]:
        tail += chunk
        steps, tail = _split_thinking_steps(tail)
        all_steps.extend(steps)
    assert all_steps == ["step one.", "step two."]
    assert tail == ""


def test_render_before_any_chunk_shows_thinking_placeholder() -> None:
    # Fresh `_ThinkingLine`, never fed via `update_chunk` — the empty-buffer
    # placeholder must show, not a blank tail.
    line = _ThinkingLine(conversation=None)
    dot_char = _BRAILLE_FRAMES[0]
    assert line._render().plain == f"{dot_char} Thinking..."


def test_render_shows_full_buffer_with_no_ellipsis_when_under_cap() -> None:
    # A buffer shorter than `_THINKING_LINE_CAP` renders whole, unprefixed.
    line = _ThinkingLine(conversation=None)
    text = "x" * (_THINKING_LINE_CAP - 1)
    line.update_chunk(text)
    dot_char = _BRAILLE_FRAMES[0]
    assert line._render().plain == f"{dot_char} {text}"


def test_render_at_exact_cap_length_shows_no_ellipsis() -> None:
    # Boundary: a buffer exactly at the cap is still shown whole (ellipsis
    # only appears once the buffer exceeds, not reaches, the cap).
    line = _ThinkingLine(conversation=None)
    text = "x" * _THINKING_LINE_CAP
    line.update_chunk(text)
    dot_char = _BRAILLE_FRAMES[0]
    assert line._render().plain == f"{dot_char} {text}"


def test_render_truncates_to_last_80_chars_with_ellipsis_when_over_cap() -> None:
    # A buffer past the cap shows only its last `_THINKING_LINE_CAP`
    # characters, prefixed by the single-char ellipsis marker.
    line = _ThinkingLine(conversation=None)
    text = "a" * (_THINKING_LINE_CAP + 20)
    line.update_chunk(text)
    dot_char = _BRAILLE_FRAMES[0]
    expected_tail = text[-_THINKING_LINE_CAP:]
    assert len(expected_tail) == _THINKING_LINE_CAP
    assert line._render().plain == f"{dot_char} …{expected_tail}"


def test_four_completed_sentences_do_not_reset_buffer() -> None:
    # Boundary below `_THINKING_LINE_SENTENCE_RESET` (5): the buffer keeps
    # accumulating every completed sentence untouched.
    line = _ThinkingLine(conversation=None)
    for chunk in ["First. ", "Second. ", "Third. ", "Fourth. "]:
        line.update_chunk(chunk)
    assert line._buffer == "First. Second. Third. Fourth. "


def test_four_completed_sentences_render_as_stacked_lines() -> None:
    # Below the reset threshold, every completed sentence renders as its own
    # line, oldest first; the dot marks only the newest (in-progress) line.
    line = _ThinkingLine(conversation=None)
    for chunk in ["First. ", "Second. ", "Third. ", "Fourth."]:
        line.update_chunk(chunk)
    dot_char = _BRAILLE_FRAMES[0]
    lines = line._render().plain.split("\n")
    assert lines[:3] == ["  First.", "  Second.", "  Third."]
    assert lines[3] == f"{dot_char} Fourth."


def test_fifth_completed_sentence_wipes_the_whole_block() -> None:
    # At `_THINKING_LINE_SENTENCE_RESET` (5) completed sentences, the entire
    # block wipes — including any in-flight tail — and restarts from a bare
    # "Thinking..." rather than scrolling.
    line = _ThinkingLine(conversation=None)
    for chunk in ["First. ", "Second. ", "Third. ", "Fourth. "]:
        line.update_chunk(chunk)

    line.update_chunk("Fifth. Sixth-partial")

    assert line._buffer == ""
    dot_char = _BRAILLE_FRAMES[0]
    assert line._render().plain == f"{dot_char} Thinking..."


def test_finish_is_a_safe_noop_before_mount() -> None:
    # `finish()` is reachable from `_run_step`'s cleanup path even if the
    # line was never mounted (e.g. an early failure); it must not raise, and
    # must still stop the (never-started) spinner cleanly.
    line = _ThinkingLine(conversation=None)
    line.finish("Done (1.2s)")
    assert line._widget is None
    assert line._spinner_task is None
