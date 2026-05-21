from __future__ import annotations

from ..router import Session


class QueryHandler:
    async def handle(self, session: Session, user_input: str) -> str:
        return "(query handler not implemented)"
