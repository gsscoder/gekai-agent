from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Permissions:
    read: bool
    write: bool
    exec: bool = False


PERMISSION_CHOICES = [
    ("read_only", "Read Only — scan and read files, no modifications"),
    ("full", "Full Access — read, write, and delete files"),
    ("deny", "No Access — chat only, no file operations"),
]


def _settings_path(working_dir: Path) -> Path:
    return working_dir / ".gekai" / "settings.local.json"


def load_permissions(working_dir: Path) -> Permissions | None:
    path = _settings_path(working_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    ws = data.get("permissions", {}).get("workspace", {})
    return Permissions(
        read=ws.get("read") == "allow",
        write=ws.get("write") == "allow",
        exec=ws.get("exec") == "allow",
    )


def save_permissions(working_dir: Path, permissions: Permissions) -> None:
    path = _settings_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    perms = data.setdefault("permissions", {})
    perms["workspace"] = {
        "read": "allow" if permissions.read else "deny",
        "write": "allow" if permissions.write else "deny",
        "exec": "allow" if permissions.exec else "deny",
    }
    perms.setdefault("external", [])
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
    if choice == "deny":
        return Permissions(read=False, write=False, exec=False)
    return None


def load_scope_gate(working_dir: Path) -> bool:
    for path in (_settings_path(working_dir), Path.home() / ".gekai" / "settings.json"):
        try:
            val = json.loads(path.read_text()).get("scope_gate")
            if isinstance(val, bool):
                return val
        except (OSError, ValueError):
            pass
    return True


def save_scope_gate(working_dir: Path, enabled: bool) -> None:
    path = _settings_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    data["scope_gate"] = enabled
    path.write_text(json.dumps(data, indent=2) + "\n")


def validate_gate_config(working_dir: Path) -> list[str]:
    errors: list[str] = []
    paths = [_settings_path(working_dir), Path.home() / ".gekai" / "settings.json"]
    for path in paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        val = data.get("scope_gate")
        if val is not None and not isinstance(val, bool):
            errors.append(f"{path}: scope_gate must be a boolean (got: {val})")
    return errors


def load_blast_radius_limit(working_dir: Path) -> int:
    for path in (_settings_path(working_dir), Path.home() / ".gekai" / "settings.json"):
        try:
            val = json.loads(path.read_text()).get("blast_radius_limit")
            if isinstance(val, int) and val > 0:
                return val
        except (OSError, ValueError):
            pass
    return 5


def load_context_limit(working_dir: Path) -> int | None:
    for path in (_settings_path(working_dir), Path.home() / ".gekai" / "settings.json"):
        try:
            val = json.loads(path.read_text()).get("context_limit")
            if isinstance(val, int) and val > 0:
                return val
        except (OSError, ValueError):
            pass
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
