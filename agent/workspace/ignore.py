from __future__ import annotations

from pathlib import Path

import pathspec

# Always hidden, even with no .gitignore present (VCS internals, build/venv noise).
# ".*" covers any dotfile/dotdir at any depth (.git, .venv, .gekai, etc.)
_BUILTIN_HIDDEN_PATTERNS: list[str] = [
    ".*",
    "node_modules/",
    "__pycache__/",
    "bin/",
    "obj/",
]

_GITIGNORE = ".gitignore"
_AIIGNORE = ".aiignore"


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


class IgnoreRules:
    """Two-tier ignore rules for a workspace.

    hidden: built-in floor + .gitignore + .aiignore. Discovery tools (scanner
    walk, grep, list_files, the workspace indexer) skip these.

    forbidden: .aiignore only. Every file tool denies these, even via an
    explicit path - a red zone with no override.

    Callers pass POSIX-style relative paths (`/` separators, no leading `/`).
    For directories, append a trailing `/` - gitwildmatch directory-only
    patterns (e.g. "build/") only match when the tested path ends with `/`;
    patterns without a trailing slash (e.g. ".*" or "build") match both forms.
    """

    def __init__(self, working_dir: Path) -> None:
        gitignore = _read_lines(working_dir / _GITIGNORE)
        aiignore = _read_lines(working_dir / _AIIGNORE)
        self._hidden = pathspec.PathSpec.from_lines(
            "gitignore", _BUILTIN_HIDDEN_PATTERNS + gitignore + aiignore
        )
        self._forbidden = pathspec.PathSpec.from_lines("gitignore", aiignore)

    def is_hidden(self, rel_path: str) -> bool:
        if rel_path in ("", "."):
            return False
        return self._hidden.match_file(rel_path)

    def is_forbidden(self, rel_path: str) -> bool:
        if rel_path in ("", "."):
            return False
        return self._forbidden.match_file(rel_path)


def load(working_dir: Path) -> IgnoreRules:
    """Build IgnoreRules for a workspace. Cheap (~2 small file reads); call
    fresh per operation rather than caching - alpha, no invalidation logic."""
    return IgnoreRules(working_dir)
