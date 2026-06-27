from __future__ import annotations

from agent.workspace.chunker import CHUNK_LINES, CHUNK_OVERLAP, chunk_text


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []


def test_chunk_text_binary_skip() -> None:
    assert chunk_text("hello\x00world") == []


def test_chunk_text_single_chunk() -> None:
    text = "\n".join(f"line {i}" for i in range(10))
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert "line 0" in chunks[0]
    assert "line 9" in chunks[0]


def test_chunk_text_overlap() -> None:
    # Enough lines to produce multiple chunks; verify adjacent chunks share lines.
    lines = [f"L{i}" for i in range(CHUNK_LINES + CHUNK_OVERLAP + 5)]
    text = "\n".join(lines)
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    # Last line of chunk 0 must appear in chunk 1 (overlap)
    last_line_chunk0 = chunks[0].splitlines()[-1]
    assert last_line_chunk0 in chunks[1]


def test_chunk_text_whitespace_only_skipped() -> None:
    text = "\n   \n\t\n"
    assert chunk_text(text) == []
