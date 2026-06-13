from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from .. import Subagent
from ...workspace.indexer import IndexStats, build_index

subagent = Subagent(
    name="ws-manager",
    namespace="worker",
    description="",
    user_invocable=False,
)


async def run(task: str, working_dir: Path, conn: sqlite3.Connection) -> IndexStats:
    """Dispatch a system-managed workspace task by name."""
    if task == "onboard":
        return await asyncio.to_thread(build_index, working_dir, conn)
    raise ValueError(f"ws-manager: unknown task {task!r}")
