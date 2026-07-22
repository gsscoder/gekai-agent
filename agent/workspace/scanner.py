from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .ignore import load as _load_ignore_rules


def _walk(start: Path, *, root: Path | None = None):
    """Yield (dir, dirnames, files) tuples, skipping paths IgnoreRules.is_hidden.

    `root` is the workspace root used to load .gitignore/.aiignore and as the
    base for computing relative paths checked against those rules; it defaults
    to `start`. Pass an explicit `root` when walking a subdirectory of a larger
    workspace (so the root-level ignore files still apply).
    """
    root = (root or start).resolve()
    rules = _load_ignore_rules(root)
    start = start.resolve()
    for dirpath, dirnames, filenames in os.walk(start):
        dp = Path(dirpath)
        rel_dir = dp.relative_to(root)
        prefix = "" if str(rel_dir) == "." else str(rel_dir).replace("\\", "/") + "/"
        dirnames[:] = [d for d in dirnames if not rules.is_hidden(prefix + d + "/")]
        filenames[:] = [f for f in filenames if not rules.is_hidden(prefix + f)]
        yield dp, dirnames, filenames


def list_files(working_dir: Path) -> list[str]:
    """Return sorted flat list of all file paths relative to working_dir."""
    paths: list[str] = []
    for dirpath, _, filenames in _walk(working_dir):
        for fname in filenames:
            rel = (dirpath / fname).relative_to(working_dir)
            paths.append(str(rel).replace("\\", "/"))
    return sorted(paths)


def list_dirs(working_dir: Path) -> list[str]:
    """Return sorted flat list of all directory paths relative to working_dir, each suffixed with '/'."""
    paths: list[str] = []
    for dirpath, dirnames, _ in _walk(working_dir):
        for dname in dirnames:
            rel = (dirpath / dname).relative_to(working_dir)
            paths.append(str(rel).replace("\\", "/") + "/")
    return sorted(paths)


def get_git_branch(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "--short", "-q", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return None
