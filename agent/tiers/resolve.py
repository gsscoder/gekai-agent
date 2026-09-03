"""Tier resolver: the one place a dispatch site turns a tier name into
runnable API params. Layers catalog + binding + keyring credential +
`model_caps` realizability on top of each other; every failure mode is a
`TierResolutionError` (chat-visible, never a silent fallback).

Returns the tier's own configured default_effort/thinking unless the caller
overrides either — see `resolve_tier`'s `effort`/`thinking` arguments. The
harness passes a touchpoint's own operating point that way
(`harness/touchpoints.py`), which is what keeps this module free of any
dependency on the harness.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.model_caps import resolve_thinking_params
from . import credentials
from .catalog import ModelCatalogEntry, Suitability, TierBinding, TierName, validate_binding


class TierResolutionError(Exception):
    """A tier binding is missing, stale (model dropped from the catalog),
    unrealizable (bad effort/thinking pairing), or has no stored credential."""


@dataclass(frozen=True)
class ResolvedTier:
    model: str
    api_key: str
    api_base: str | None
    extra_params: dict


def resolve_tier(
    tier: TierName,
    catalog: dict[str, ModelCatalogEntry],
    bindings: dict[TierName, TierBinding],
    *,
    effort: str | None = None,
    thinking: bool | None = None,
) -> ResolvedTier:
    """Turn a tier name into runnable API params. `effort`/`thinking`, when
    given, override the binding's own operating point — the binding still
    decides WHICH model and WHICH credential, the caller only decides HOW
    that model is operated for its job. Passing neither resolves purely from
    the binding."""
    binding = bindings.get(tier)
    if binding is None:
        raise TierResolutionError(f"tier {tier.value!r} is not configured — run /tier")
    try:
        validate_binding(tier, binding, catalog)
    except ValueError as exc:
        raise TierResolutionError(f"tier {tier.value!r} binding is stale: {exc}") from exc
    entry = catalog[binding.model]
    # Keyed by the model, not the tier's operating point: a credential is a
    # property of the model (`/models`), independent of which tier currently
    # points at it (`/tier`), so retuning effort/thinking never invalidates a
    # stored key.
    cred_key = credentials.credential_key(entry.provider, binding.model)
    try:
        api_key = credentials.get_api_key(cred_key)
    except LookupError:
        raise TierResolutionError(f"no stored credential for model {binding.model!r} — run /models")
    effective_effort = binding.default_effort if effort is None else effort
    effective_thinking = binding.thinking if thinking is None else thinking
    if effective_effort not in entry.efforts:
        raise TierResolutionError(
            f"effort {effective_effort!r} is not among {binding.model!r}'s declared "
            f"efforts {entry.efforts}"
        )
    return ResolvedTier(
        model=binding.model,
        api_key=api_key,
        api_base=entry.base_url,
        extra_params=resolve_thinking_params(
            binding.model, effective_effort, enabled=effective_thinking,
        ),
    )


@dataclass(frozen=True)
class TierRowStatus:
    """Everything `/tier`'s listing (or the coherence check below) needs for one
    tier, reported rather than raised — unlike `resolve_tier`, this never
    throws; every failure mode a real dispatch would hit instead shows up as
    a `None`/`False` field or a non-empty `invalid_reason`."""
    tier: TierName
    binding: TierBinding | None        # None = no binding saved yet
    entry: ModelCatalogEntry | None    # None = no binding, or binding's model missing from catalog (stale)
    has_credential: bool
    verdict: Suitability | None        # None when there's no model yet to rate
    invalid_reason: str | None         # set when validate_binding rejects an otherwise-cataloged binding

    @property
    def ready(self) -> bool:
        """True iff `resolve_tier(self.tier, ...)` would succeed against the
        same catalog+bindings — the one fact both the coherence predicate and
        a grid row's status column need."""
        return (
            self.binding is not None
            and self.entry is not None
            and self.invalid_reason is None
            and self.has_credential
        )

    @property
    def label(self) -> str:
        if self.binding is None:
            return "not configured"
        if self.entry is None:
            return "stale — model missing from catalog"
        if self.invalid_reason is not None:
            return f"stale — {self.invalid_reason}"
        if not self.has_credential:
            return "no key"
        if self.verdict in ("warning", "deprecated"):
            return self.verdict
        return "ready"


def tier_status(
    tier: TierName,
    catalog: dict[str, ModelCatalogEntry],
    bindings: dict[TierName, TierBinding],
) -> TierRowStatus:
    binding = bindings.get(tier)
    if binding is None:
        return TierRowStatus(tier=tier, binding=None, entry=None, has_credential=False, verdict=None, invalid_reason=None)
    entry = catalog.get(binding.model)
    if entry is None:
        return TierRowStatus(tier=tier, binding=binding, entry=None, has_credential=False, verdict=None, invalid_reason=None)
    invalid_reason: str | None = None
    try:
        validate_binding(tier, binding, catalog)
    except ValueError as exc:
        invalid_reason = str(exc)
    return TierRowStatus(
        tier=tier,
        binding=binding,
        entry=entry,
        has_credential=credentials.has_api_key(credentials.credential_key(entry.provider, binding.model)),
        verdict=entry.suitability.verdict(tier, binding.thinking),
        invalid_reason=invalid_reason,
    )


def all_tiers_ready(catalog: dict[str, ModelCatalogEntry], bindings: dict[TierName, TierBinding]) -> bool:
    """The single coherence predicate (fixes the "'fast' is not configured"
    incoherence: a tier can have a saved binding yet still be unusable — a
    stale model, a bad effort, or simply no stored credential — and the old
    `tiers_configured()` only checked "has a binding", so it could report
    ready while `resolve_touchpoint` still raised on first use). True iff
    every one of FAST/SUPP/CORE is fully resolvable right now."""
    return all(tier_status(t, catalog, bindings).ready for t in TierName)


__all__ = [
    "ResolvedTier",
    "TierResolutionError",
    "resolve_tier",
    "TierRowStatus",
    "tier_status",
    "all_tiers_ready",
]
