from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .llm.tiers import DEFAULT_MODEL_CATALOG, ModelCatalogEntry, TierBinding, TierName, TierSuitability


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
    """Ensures ~/.gekai/settings.json exists and every node it should carry
    is present — the single entry point for global-settings initialization,
    so a new node (like `models`) never needs its own call site wired in
    separately. Each node's own seed function stays independently
    idempotent (all-or-nothing per node, not per file — plan 28 decision 3)."""
    path = Path.home() / ".gekai" / "settings.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"env": {}}, indent=2) + "\n")
    seed_model_catalog_if_absent()


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
    path.write_text(json.dumps(data, indent=2) + "\n")


def _catalog_entry_to_dict(entry: ModelCatalogEntry) -> dict:
    return {
        "name": entry.name,
        "base_url": entry.base_url,
        "efforts": list(entry.efforts),
        "thinking": entry.thinking,
        "suitability": asdict(entry.suitability),
    }


def _catalog_entry_from_dict(d: dict) -> ModelCatalogEntry:
    return ModelCatalogEntry(
        name=d["name"],
        base_url=d.get("base_url"),
        efforts=tuple(d["efforts"]),
        thinking=d["thinking"],
        suitability=TierSuitability(**d.get("suitability", {})),
    )


def _binding_to_dict(binding: TierBinding) -> dict:
    return {"model": binding.model, "default_effort": binding.default_effort, "thinking": binding.thinking}


def _binding_from_dict(d: dict) -> TierBinding:
    return TierBinding(model=d["model"], default_effort=d["default_effort"], thinking=d.get("thinking", False))


def seed_model_catalog_if_absent() -> None:
    """All-or-nothing seed (plan 28 decision 3): writes the code-side
    `DEFAULT_MODEL_CATALOG` only if the `models` node is entirely absent
    from settings.json. Never re-touched once present — a future version's
    larger catalog only reaches an existing install if the user deletes the
    node to force a re-seed."""
    path = _global_settings_path()
    data = _load_global_data(path)
    if "models" in data:
        return
    data["models"] = [_catalog_entry_to_dict(e) for e in DEFAULT_MODEL_CATALOG]
    _save_global_data(path, data)


def load_model_catalog() -> dict[str, ModelCatalogEntry]:
    data = _load_global_data(_global_settings_path())
    return {e["name"]: _catalog_entry_from_dict(e) for e in data.get("models", [])}


def save_model_catalog_entry(entry: ModelCatalogEntry) -> None:
    path = _global_settings_path()
    data = _load_global_data(path)
    models = data.setdefault("models", [])
    models[:] = [m for m in models if m.get("name") != entry.name]
    models.append(_catalog_entry_to_dict(entry))
    _save_global_data(path, data)


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
