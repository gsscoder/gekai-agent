from __future__ import annotations

from ..router import Session


class ActionHandler:
    async def handle(self, session: Session, user_input: str) -> str:
        return "(action handler not implemented)"
