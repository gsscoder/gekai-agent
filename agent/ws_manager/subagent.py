from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from ..subagent import SubAgent, SubAgentEvent, SubAgentStartEvent, DoneEvent
from .. import workspace_db


class WsManager(SubAgent):
    name = "ws-manager"
    color = "#008000"

    @property
    def description(self) -> str:
        return "Initializing workspace"

    def __init__(self, working_dir: Path) -> None:
        self._working_dir = working_dir

    async def run(self) -> AsyncIterator[SubAgentEvent]:
        yield SubAgentStartEvent(
            name=self.name,
            description=self.description,
            color=self.color,
        )
        conn = await asyncio.to_thread(workspace_db.ensure, self._working_dir)
        conn.close()
        yield DoneEvent()
