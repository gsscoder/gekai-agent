"""Model tiers: the capability contracts Gekai dispatches against.

`catalog` holds the data types and the code-side model catalog, `store`
persists which model fills each tier, `credentials` holds the keyring
access, and `resolve` turns a tier into runnable API params. None of it
depends on the harness or the TUI, so a dispatch site can resolve a tier
without importing either.
"""

from __future__ import annotations

from . import credentials
from .catalog import (
    DEFAULT_MODEL_CATALOG,
    EFFORT_LADDER,
    ModelCatalogEntry,
    Suitability,
    TierBinding,
    TierName,
    TierPolicy,
    TierSuitability,
    validate_binding,
)
from .resolve import (
    ResolvedTier,
    TierResolutionError,
    TierRowStatus,
    all_tiers_ready,
    resolve_tier,
    tier_status,
)
from .store import (
    load_model_catalog,
    load_tier_bindings,
    save_tier_binding,
    tiers_configured,
)

__all__ = [
    "credentials",
    "DEFAULT_MODEL_CATALOG",
    "EFFORT_LADDER",
    "ModelCatalogEntry",
    "Suitability",
    "TierBinding",
    "TierName",
    "TierPolicy",
    "TierSuitability",
    "validate_binding",
    "ResolvedTier",
    "TierResolutionError",
    "TierRowStatus",
    "all_tiers_ready",
    "resolve_tier",
    "tier_status",
    "load_model_catalog",
    "load_tier_bindings",
    "save_tier_binding",
    "tiers_configured",
]
