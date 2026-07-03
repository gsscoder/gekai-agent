from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Plan 26 mechanism #1 ("action budget wrap"): the cap on main's direct
# edit_file/write_file calls per user request, applied when the Improvement 1
# scope estimate (agent/pipeline/estimate.py) says the request is
# implementation-sized, or when the user explicitly named a specialist, or
# when no estimator is configured at all (pre-Improvement-1 flat behavior).
# A trivial estimate or an estimate parse failure gets an unbounded budget
# instead — fail open, per the plan's risk section.
DEFAULT_EDIT_BUDGET: int = 5


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
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    perms = data.setdefault("permissions", {})
    allow_hidden = perms.setdefault("allow_hidden", [])
    if rel not in allow_hidden:
        allow_hidden.append(rel)
    path.write_text(json.dumps(data, indent=2) + "\n")


def bootstrap_global_settings() -> None:
    path = Path.home() / ".gekai" / "settings.json"
    if path.exists():
        return
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


def resolve_permissions(choice: str) -> Permissions | None:
    if choice == "read_only":
        return Permissions(read=True, write=False)
    if choice == "full":
        return Permissions(read=True, write=True)
    if choice == "deny":
        return Permissions(read=False, write=False, exec=False)
    return None


def load_context_limit(working_dir: Path) -> int | None:
    for path in (_settings_path(working_dir), Path.home() / ".gekai" / "settings.json"):
        try:
            val = json.loads(path.read_text()).get("context_limit")
            if isinstance(val, int) and val > 0:
                return val
        except (OSError, ValueError):
            pass
    return None
