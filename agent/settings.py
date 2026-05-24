from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Permissions:
    read: bool
    write: bool


PERMISSION_CHOICES = [
    ("read_only", "Read Only — scan and read files, no modifications"),
    ("full", "Full Access — read, write, and delete files"),
    ("deny", "Deny — exit to terminal"),
]


def _settings_path(working_dir: Path) -> Path:
    return working_dir / ".gekai" / "settings.local.json"


def load_permissions(working_dir: Path) -> Permissions | None:
    path = _settings_path(working_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    perms = data.get("permissions", {})
    return Permissions(
        read=perms.get("read") == "allow",
        write=perms.get("write") == "allow",
    )


def save_permissions(working_dir: Path, permissions: Permissions) -> None:
    path = _settings_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "permissions": {
            "read": "allow" if permissions.read else "deny",
            "write": "allow" if permissions.write else "deny",
        }
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def resolve_permissions(choice: str) -> Permissions | None:
    if choice == "read_only":
        return Permissions(read=True, write=False)
    if choice == "full":
        return Permissions(read=True, write=True)
    return None
