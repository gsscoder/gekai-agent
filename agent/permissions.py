from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass
class Permissions:
    read: bool
    write: bool
    exec: bool = False


# The choices the first-run permission picker offers, in display order.
PERMISSION_CHOICES = [
    ("read_only", "Read Only — scan and read files, no modifications"),
    ("full", "Full Access — read, write, and delete files"),
    ("deny", "No Access — chat only, no file operations"),
]


def resolve_permissions(choice: str) -> Permissions | None:
    """Turn a `PERMISSION_CHOICES` key into a grant. `None` for an unknown key."""
    if choice == "read_only":
        return Permissions(read=True, write=False)
    if choice == "full":
        return Permissions(read=True, write=True)
    if choice == "deny":
        return Permissions(read=False, write=False, exec=False)
    return None


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
