from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .settings import Permissions


PermissionCallback = Callable[[str, str], Awaitable[bool]]


@dataclass
class PermissionGate:
    permissions: Permissions
    on_request: PermissionCallback | None = None
    _pending: set[str] = field(default_factory=set, repr=False)

    async def check(self, tool: "Tool") -> bool:  # noqa: F821 — forward ref to llm.tools.Tool
        perm = tool.required_permission
        if perm == "none":
            return True
        granted = getattr(self.permissions, perm, False)
        if granted:
            return True
        if perm in self._pending:
            return False
        if self.on_request is not None:
            self._pending.add(perm)
            try:
                if await self.on_request(perm, tool.name):
                    setattr(self.permissions, perm, True)
                    return True
            finally:
                self._pending.discard(perm)
        return False
