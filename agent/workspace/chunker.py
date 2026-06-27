from __future__ import annotations

CHUNK_LINES = 40
CHUNK_OVERLAP = 10
MAX_FILE_BYTES = 512 * 1024  # ponytail: skip files > 512 KB


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping line-window chunks.

    Returns [] for binary content (null bytes in first 1 KB) or empty files.
    """
    if not text or "\x00" in text[:1024]:
        return []
    lines = text.splitlines()
    if not lines:
        return []
    step = max(1, CHUNK_LINES - CHUNK_OVERLAP)
    chunks: list[str] = []
    for i in range(0, len(lines), step):
        chunk = "\n".join(lines[i : i + CHUNK_LINES])
        if chunk.strip():
            chunks.append(chunk)
    return chunks
