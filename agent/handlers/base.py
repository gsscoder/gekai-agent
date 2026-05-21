from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from ..router import Session


class Handler(Protocol):
    async def handle(self, session: Session, user_input: str) -> str: ...
    async def stream(self, session: Session, user_input: str) -> AsyncIterator[str]: ...
