from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Literal

from rich.text import Text

DiffKind = Literal["add", "del", "context", "header"]

_CAP = 40  # max changed lines before truncation


@dataclass(frozen=True, slots=True)
class DiffLine:
    kind: DiffKind
    text: str


def build_diff(old: str, new: str, cap: int = _CAP) -> list[DiffLine]:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    raw = list(difflib.unified_diff(old_lines, new_lines, n=3))

    result: list[DiffLine] = []
    changed = 0
    truncated_at: int | None = None

    for i, line in enumerate(raw):
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@"):
            result.append(DiffLine(kind="header", text=line.rstrip("\n")))
            continue
        if line.startswith("+"):
            kind: DiffKind = "add"
            changed += 1
        elif line.startswith("-"):
            kind = "del"
            changed += 1
        else:
            kind = "context"

        if changed > cap:
            truncated_at = len(raw) - i
            break

        result.append(DiffLine(kind=kind, text=line.rstrip("\n")))

    if truncated_at is not None:
        result.append(DiffLine(kind="context", text=f"… {truncated_at} more lines"))

    return result


def render_diff(lines: list[DiffLine]) -> Text:
    text = Text()
    for i, dl in enumerate(lines):
        if i > 0:
            text.append("\n")
        if dl.kind == "add":
            text.append(dl.text, style="on #1a3a1a")
        elif dl.kind == "del":
            text.append(dl.text, style="on #3a1a1a")
        elif dl.kind == "header":
            text.append(dl.text, style="dim")
        else:
            text.append(dl.text)
    return text
