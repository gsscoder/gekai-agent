from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from .. import Subagent
from ...workspace.indexer import IndexStats, build_index
from ...tools.catalog import EDIT_TOOLS, FS_TOOLS, READ_TOOLS

subagent = Subagent(
    name="ws-manager",
    short_description="repo scaffolding: project skeletons, dirs, manifests",
    namespace="worker",
    description=(
        "repo/filesystem scaffolding: project skeletons, directories, manifest files, conventional "
        "project layout. Not for application logic — no features, fixes, or business code"
    ),
    mandate=(
        "you act as a workspace operations specialist — repo/filesystem scaffolding only: project "
        "skeletons, directories, manifest files, conventional layout; never application logic"
    ),
    directives=(
        "scaffold only — directories, manifest files (package.json, pyproject.toml, etc.), conventional "
        "project layout for the target stack; never write business logic, features, or application code\n"
        "match the conventional layout for the project's stack; do not invent a bespoke structure\n"
        "after scaffolding, report what was created"
    ),
    tools=list(READ_TOOLS + EDIT_TOOLS + FS_TOOLS),
    is_fallback=True,
    user_invocable=True,
)


async def run(task: str, working_dir: Path, conn: sqlite3.Connection) -> IndexStats:
    """Dispatch a system-managed workspace task by name."""
    if task == "onboard":
        return await asyncio.to_thread(build_index, working_dir, conn)
    raise ValueError(f"ws-manager: unknown task {task!r}")
