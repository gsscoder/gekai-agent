from __future__ import annotations

import json
import os
from pathlib import Path

from .atomic_io import atomic_write, lock_for
from .permissions import Permissions


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
    with lock_for(path):
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
        atomic_write(path, json.dumps(data, indent=2) + "\n")


def load_allow_hidden(working_dir: Path) -> set[str]:
    path = _settings_path(working_dir)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return set()
    return set(data.get("permissions", {}).get("allow_hidden", []))


def save_allow_hidden(working_dir: Path, rel: str) -> None:
    path = _settings_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_for(path):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        perms = data.setdefault("permissions", {})
        allow_hidden = perms.setdefault("allow_hidden", [])
        if rel not in allow_hidden:
            allow_hidden.append(rel)
        atomic_write(path, json.dumps(data, indent=2) + "\n")


def load_external(working_dir: Path) -> list[Path]:
    path = _settings_path(working_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    return [Path(p) for p in data.get("permissions", {}).get("external", [])]


def save_external(working_dir: Path, root: Path) -> None:
    root = root.resolve()
    path = _settings_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_for(path):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        perms = data.setdefault("permissions", {})
        existing = [Path(p) for p in perms.get("external", [])]
        if any(root.is_relative_to(e) for e in existing):
            return
        remaining = [e for e in existing if not e.is_relative_to(root)]
        remaining.append(root)
        perms["external"] = [str(p) for p in remaining]
        atomic_write(path, json.dumps(data, indent=2) + "\n")


def load_directive_audit_enabled(working_dir: Path) -> bool:
    """Off switch for plan 35's directive auditor (decision 15, unmeasurable
    features get deleted). Defaults to enabled — absent from a fresh
    `settings.local.json`, like `load_allow_hidden`'s empty-set default, so
    a project that never touched this setting gets the audit rather than
    silently losing it; disabling is an explicit opt-out, not an implicit
    one. Phase 0: nothing reads this yet."""
    path = _settings_path(working_dir)
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return True
    return bool(data.get("directive_audit", {}).get("enabled", True))


def save_directive_audit_enabled(working_dir: Path, enabled: bool) -> None:
    path = _settings_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_for(path):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        data.setdefault("directive_audit", {})["enabled"] = enabled
        atomic_write(path, json.dumps(data, indent=2) + "\n")


def bootstrap_global_settings() -> None:
    """Ensures ~/.gekai/settings.json exists."""
    path = Path.home() / ".gekai" / "settings.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"env": {}}, indent=2) + "\n")


def load_global_settings() -> None:
    try:
        path = Path.home() / ".gekai" / "settings.json"
        data = json.loads(path.read_text())
        for key, value in data.get("env", {}).items():
            os.environ[key] = value
    except (OSError, ValueError):
        pass


def load_context_limit(working_dir: Path) -> int | None:
    for path in (_settings_path(working_dir), Path.home() / ".gekai" / "settings.json"):
        try:
            val = json.loads(path.read_text()).get("context_limit")
            if isinstance(val, int) and val > 0:
                return val
        except (OSError, ValueError):
            pass
    return None
