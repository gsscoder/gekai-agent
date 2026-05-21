from __future__ import annotations

import subprocess
from pathlib import Path


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
