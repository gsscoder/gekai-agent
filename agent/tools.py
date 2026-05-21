from __future__ import annotations

import re
from pathlib import Path

from llmstitch import tool

_MAX_RESULTS = 200


async def _read_file(path: str, *, working_dir: Path) -> str:
    target = (working_dir / path).resolve()
    if not target.is_relative_to(working_dir.resolve()):
        return "error: path outside working directory"
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return f"error: file not found: {path}"
    except Exception as exc:
        return f"error: {exc}"


async def _list_files(pattern: str, *, working_dir: Path) -> str:
    matches = sorted(
        str(p.relative_to(working_dir))
        for p in working_dir.glob(pattern)
        if p.is_file()
    )[:_MAX_RESULTS]
    return "\n".join(matches) if matches else "(no matches)"


async def _grep(pattern: str, path: str | None = None, *, working_dir: Path) -> str:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"error: invalid pattern: {exc}"

    root = working_dir.resolve()
    if path:
        target = (working_dir / path).resolve()
        if not target.is_relative_to(root):
            return "error: path outside working directory"
        candidates = [target] if target.is_file() else [p for p in target.rglob("*") if p.is_file()]
    else:
        candidates = [p for p in root.rglob("*") if p.is_file()]

    results: list[str] = []
    for file in candidates:
        if len(results) >= _MAX_RESULTS:
            break
        try:
            for i, line in enumerate(
                file.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if regex.search(line):
                    results.append(f"{file.relative_to(root)}:{i}: {line}")
                    if len(results) >= _MAX_RESULTS:
                        break
        except Exception:
            continue

    return "\n".join(results) if results else "(no matches)"


def make_tools(working_dir: Path) -> list:
    @tool
    async def read_file(path: str) -> str:
        """Read the full contents of a file in the repository."""
        return await _read_file(path, working_dir=working_dir)

    @tool
    async def list_files(pattern: str) -> str:
        """List files matching a glob pattern (e.g. '**/*.py')."""
        return await _list_files(pattern, working_dir=working_dir)

    @tool
    async def grep(pattern: str, path: str | None = None) -> str:
        """Search file contents for a regex pattern. Returns matching lines as file:line: content."""
        return await _grep(pattern, path=path, working_dir=working_dir)

    return [read_file, list_files, grep]
