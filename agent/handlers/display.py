from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from ..router import Session
from ..tools import _read_file

_PATH_TOKEN = re.compile(r"[\w./\\-]+\.[\w]+")
_TAIL_RE = re.compile(r"\blast\s+(\d+)\s+lines?\b", re.IGNORECASE)
_HEAD_RE = re.compile(r"\bfirst\s+(\d+)\s+lines?\b", re.IGNORECASE)
_RANGE_RE = re.compile(r"\b(?:from\s+|lines?\s+)?(\d+)\s+(?:to|-)\s+(\d+)\b", re.IGNORECASE)


@dataclass
class _LineRange:
    start: int | None  # 1-based inclusive
    end: int | None    # 1-based inclusive
    tail: int | None


def _parse_range(text: str) -> _LineRange:
    m = _TAIL_RE.search(text)
    if m:
        return _LineRange(start=None, end=None, tail=int(m.group(1)))
    m = _HEAD_RE.search(text)
    if m:
        return _LineRange(start=1, end=int(m.group(1)), tail=None)
    m = _RANGE_RE.search(text)
    if m:
        return _LineRange(start=int(m.group(1)), end=int(m.group(2)), tail=None)
    return _LineRange(start=None, end=None, tail=None)


def _extract_path(sub_prompt: str) -> str | None:
    match = _PATH_TOKEN.search(sub_prompt)
    return match.group(0) if match else None


def _resolve_path(raw: str, working_dir: Path) -> str | None:
    if (working_dir / raw).is_file():
        return raw
    name = Path(raw).name
    matches = sorted(working_dir.rglob(name))
    if matches:
        return str(matches[0].relative_to(working_dir))
    return None


class DisplayHandler:
    def __init__(self, working_dir: Path) -> None:
        self._working_dir = working_dir

    async def handle(self, session: Session, user_input: str) -> str:
        chunks = [c async for c in self.stream(session, user_input)]
        return "".join(chunks)

    async def stream(self, session: Session, user_input: str) -> AsyncIterator[str]:
        raw = _extract_path(user_input)
        if raw is None:
            yield f"error: no file path found in: {user_input!r}"
            return
        path = _resolve_path(raw, self._working_dir)
        if path is None:
            yield f"error: file not found: {raw!r}"
            return
        contents = await _read_file(path, working_dir=self._working_dir)
        lines = contents.splitlines()
        lr = _parse_range(user_input)
        if lr.tail is not None:
            first = max(0, len(lines) - lr.tail)
            lines = lines[first:]
            first_lineno = first + 1
        elif lr.start is not None:
            first_lineno = lr.start
            lines = lines[lr.start - 1 : lr.end]
        else:
            first_lineno = 1
        numbered = "\n".join(
            f"{i:4d}  {line}" for i, line in enumerate(lines, first_lineno)
        )
        ext = Path(path).suffix.lstrip(".")
        fence = f"```{ext}\n" if ext else "```\n"
        yield f"`{path}`\n{fence}{numbered}\n```"
