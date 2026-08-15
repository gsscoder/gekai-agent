from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from .llm.tiers import DEFAULT_MODEL_CATALOG, ModelCatalogEntry, TierBinding, TierName


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


# Per-path locks (plus a guard lock protecting the dict itself) serialize the
# read-modify-write critical section in each save_* below against concurrent
# callers (e.g. overlapping permission-grant callbacks) racing to update the
# same settings file.
_locks: dict[Path, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _locks[path] = lock
        return lock


def _atomic_write(path: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


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
    with _lock_for(path):
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
        _atomic_write(path, json.dumps(data, indent=2) + "\n")


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
    with _lock_for(path):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        perms = data.setdefault("permissions", {})
        allow_hidden = perms.setdefault("allow_hidden", [])
        if rel not in allow_hidden:
            allow_hidden.append(rel)
        _atomic_write(path, json.dumps(data, indent=2) + "\n")


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
    with _lock_for(path):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = {}
        data.setdefault("directive_audit", {})["enabled"] = enabled
        _atomic_write(path, json.dumps(data, indent=2) + "\n")


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


# --- model tiers (plan 28 Phase 1a) --------------------------------------
# Global-only for now (user-home ~/.gekai/settings.json); project-level
# override is explicitly deferred (decision 3b).

def _global_settings_path() -> Path:
    return Path.home() / ".gekai" / "settings.json"


def _load_global_data(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _save_global_data(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(data, indent=2) + "\n")


def _binding_to_dict(binding: TierBinding) -> dict:
    return {"model": binding.model, "default_effort": binding.default_effort, "thinking": binding.thinking}


def _binding_from_dict(d: dict) -> TierBinding:
    return TierBinding(model=d["model"], default_effort=d["default_effort"], thinking=d.get("thinking", False))


def load_model_catalog() -> dict[str, ModelCatalogEntry]:
    """Always a live read of the code-side catalog — the set of *available*
    models is never persisted to disk, so a newly-added model in
    `DEFAULT_MODEL_CATALOG` shows up immediately without a stale on-disk
    copy to go out of date."""
    return {e.name: e for e in DEFAULT_MODEL_CATALOG}


def load_tier_bindings() -> dict[TierName, TierBinding]:
    data = _load_global_data(_global_settings_path())
    bindings: dict[TierName, TierBinding] = {}
    for tier_key, binding_dict in data.get("tiers", {}).items():
        try:
            tier = TierName(tier_key)
        except ValueError:
            continue
        bindings[tier] = _binding_from_dict(binding_dict)
    return bindings


def save_tier_binding(tier: TierName, binding: TierBinding) -> None:
    path = _global_settings_path()
    with _lock_for(path):
        data = _load_global_data(path)
        tiers = data.setdefault("tiers", {})
        tiers[tier.value] = _binding_to_dict(binding)
        _save_global_data(path, data)


def tiers_configured() -> bool:
    """True only once all three tiers (FAST/SUPP/CORE) are fully resolvable
    — bound to a model still present in the catalog, with a valid effort for
    that model, and a stored keyring credential — not merely "has a binding"
    (a binding alone can still fail to resolve at dispatch time, e.g. no
    stored credential, which is the exact incoherence this used to allow).
    Partial or unresolvable configuration still counts as "not configured"
    for the purpose of the startup/prompt nudge (there's no
    reduced-functionality mode).

    Deferred import to avoid a circular import: `agent.llm.resolve` imports
    `agent.harness.touchpoints`, which imports the `agent.harness` package,
    which imports `agent.harness.core`, which does `from ..settings import
    Permissions` — a module-level import here would be circular. See
    `agent/llm/tiers.py::_build_default_catalog()` for the identical
    pattern."""
    from .llm.resolve import all_tiers_ready

    return all_tiers_ready(load_model_catalog(), load_tier_bindings())
