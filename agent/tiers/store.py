"""Persistence for tier bindings: which model fills each tier, and at what
operating point. Global-only (user-home `~/.gekai/settings.json`); a
project-level override is deliberately not offered.

The *catalog* (which models exist at all) is never persisted — it is read
live from the code-side `DEFAULT_MODEL_CATALOG`, so a newly-added model
shows up immediately with no stale on-disk copy to go out of date.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..atomic_io import atomic_write, lock_for
from .catalog import DEFAULT_MODEL_CATALOG, ModelCatalogEntry, TierBinding, TierName
from .resolve import all_tiers_ready


def _global_settings_path() -> Path:
    return Path.home() / ".gekai" / "settings.json"


def _load_global_data(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _save_global_data(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(data, indent=2) + "\n")


def _binding_to_dict(binding: TierBinding) -> dict:
    return {"model": binding.model, "default_effort": binding.default_effort, "thinking": binding.thinking}


def _binding_from_dict(d: dict) -> TierBinding:
    return TierBinding(model=d["model"], default_effort=d["default_effort"], thinking=d.get("thinking", False))


def load_model_catalog() -> dict[str, ModelCatalogEntry]:
    """Always a live read of the code-side catalog — see the module docstring."""
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
    with lock_for(path):
        data = _load_global_data(path)
        tiers = data.setdefault("tiers", {})
        tiers[tier.value] = _binding_to_dict(binding)
        _save_global_data(path, data)


def tiers_configured() -> bool:
    """True only once all three tiers (FAST/SUPP/CORE) are fully resolvable
    — bound to a model still present in the catalog, with a valid effort for
    that model, and a stored keyring credential — not merely "has a binding".
    A binding alone can still fail to resolve at dispatch time, which is the
    exact incoherence this rules out. Partial or unresolvable configuration
    counts as "not configured" for the startup/prompt nudge; there is no
    reduced-functionality mode."""
    return all_tiers_ready(load_model_catalog(), load_tier_bindings())


__all__ = [
    "load_model_catalog",
    "load_tier_bindings",
    "save_tier_binding",
    "tiers_configured",
]
