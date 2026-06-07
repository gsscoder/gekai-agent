from __future__ import annotations

import os
import subprocess
from pathlib import Path

_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "node_modules", ".venv", "__pycache__", "bin", "obj"}
)


def _walk(working_dir: Path):
    """Yield all (dir, files) pairs, skipping _SKIP_DIRS."""
    for dirpath, dirnames, filenames in os.walk(working_dir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        yield Path(dirpath), filenames


def list_files(working_dir: Path) -> list[str]:
    """Return sorted flat list of all file paths relative to working_dir."""
    paths: list[str] = []
    for dirpath, filenames in _walk(working_dir):
        for fname in filenames:
            rel = (dirpath / fname).relative_to(working_dir)
            paths.append(str(rel).replace("\\", "/"))
    return sorted(paths)


def get_git_branch(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            return branch if branch != "HEAD" else None
    except FileNotFoundError:
        pass
    return None
