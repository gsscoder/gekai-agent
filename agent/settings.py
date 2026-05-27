from __future__ import annotations

import json
import os
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


def bootstrap_global_settings() -> None:
    path = Path.home() / ".gekai" / "settings.json"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"env": {}, "ws_scan_staleness_min": 30}, indent=2) + "\n")


def load_global_settings() -> None:
    try:
        path = Path.home() / ".gekai" / "settings.json"
        data = json.loads(path.read_text())
        for key, value in data.get("env", {}).items():
            os.environ[key] = value
    except (OSError, ValueError):
        pass


def resolve_permissions(choice: str) -> Permissions | None:
    if choice == "read_only":
        return Permissions(read=True, write=False)
    if choice == "full":
        return Permissions(read=True, write=True)
    return None


def load_ws_scan_staleness_min(working_dir: Path) -> float:
    local_path = _settings_path(working_dir)
    try:
        data = json.loads(local_path.read_text())
        val = data.get("ws_scan_staleness_min")
        if val is not None:
            return float(val)
    except (OSError, ValueError):
        pass
    global_path = Path.home() / ".gekai" / "settings.json"
    try:
        data = json.loads(global_path.read_text())
        val = data.get("ws_scan_staleness_min")
        if val is not None:
            return float(val)
    except (OSError, ValueError):
        pass
    return 30.0
