from __future__ import annotations

from typing import Protocol

from ..router import Session


class Handler(Protocol):
    async def handle(self, session: Session, user_input: str) -> str: ...
