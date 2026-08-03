"""Model-tier data types (plan 28, Phase 0).

Inert: defines the shape of a per-component tier policy, not consulted by
any dispatch path yet. A tier is a capability contract, not a model — the
same model may occupy multiple tiers at different operating points. Phase 1
wires a global tier->(model, creds, url) config; Phase 2 wires assignment-
time scaling within a component's declared space.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

# Reuses the effort vocabulary already established in model_caps.py's
# _EFFORT_TO_PARAMS ladder — not a new scale.
EFFORT_LADDER: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


class TierName(Enum):
    """Fixed at three for v1 (plan 28 open point 4). Not open string keys
    yet — revisit if a fourth tier (e.g. BULK) is ever warranted."""
    FAST = "fast"
    SUPP = "supp"
    CORE = "core"


_TIER_RANK: dict[TierName, int] = {TierName.FAST: 0, TierName.SUPP: 1, TierName.CORE: 2}


def _effort_index(effort: str) -> int:
    try:
        return EFFORT_LADDER.index(effort)
    except ValueError:
        raise ValueError(f"unknown effort level {effort!r}; expected one of {EFFORT_LADDER}") from None


@dataclass(frozen=True)
class TierPolicy:
    """A component's tier mobility: an ordered, ascending array of tiers the
    harness may run it at, plus which one is the configured default absent
    any assignment-time scaling. Effort/thinking are NOT a per-component axis
    — they come entirely from the tier's own global binding (TierBinding).
    A degenerate space (len(allowed) == 1) is how a non-scalable component
    is expressed — no separate freeze flag."""
    default: TierName
    allowed: tuple[TierName, ...]  # ascending by tier rank (FAST < SUPP < CORE); default must be a member

    def __post_init__(self) -> None:
        if not self.allowed:
            raise ValueError("a tier policy must declare at least one allowed tier")
        ranks = [_TIER_RANK[t] for t in self.allowed]
        if ranks != sorted(ranks) or len(set(ranks)) != len(ranks):
            raise ValueError(f"allowed tiers must be ascending with no duplicates, got {self.allowed}")
        if self.default not in self.allowed:
            raise ValueError(f"configured default {self.default} is not in allowed tiers {self.allowed}")


Suitability = Literal["ok", "warning", "deprecated"]


@dataclass(frozen=True)
class TierSuitability:
    """Advisory-only per-model verdicts (locked decision 9): surfaced as a
    config-time warning, never a block. `core_thinking` overrides `core`
    when the operating point has thinking enabled (e.g. Sonnet: warning at
    CORE non-thinking, ok at CORE thinking)."""
    fast: Suitability = "ok"
    supp: Suitability = "ok"
    core: Suitability = "ok"
    core_thinking: Suitability | None = None  # None = same verdict as `core`

    def verdict(self, tier: TierName, thinking: bool = False) -> Suitability:
        if tier is TierName.FAST:
            return self.fast
        if tier is TierName.SUPP:
            return self.supp
        if thinking and self.core_thinking is not None:
            return self.core_thinking
        return self.core


# Seed table for the models already known to the codebase (model_caps.py) —
# advisory only; an unlisted model defaults to "ok" everywhere (no false
# alarm on a model we simply haven't rated yet). DeepSeek only for now —
# see the note on MODEL_CAPS.
MODEL_SUITABILITY: dict[str, TierSuitability] = {
    # reasoning-only (can_disable_thinking=False in model_caps.py) can never
    # actually run non-thinking, so FAST/SUPP aren't merely suboptimal —
    # they're structurally unrealizable; rated deprecated, not warning.
    "deepseek-v4-pro":    TierSuitability(fast="deprecated", supp="warning", core="ok"),
    # this project's actual FAST- and SUPP-tier model.
    "deepseek-v4-flash":  TierSuitability(fast="ok", supp="ok", core="warning"),
}

# Models with a genuinely public, documented canonical endpoint — filled in
# from fact (confirmed via this project's own tests/.env.test), never guessed.
_KNOWN_BASE_URLS: dict[str, str] = {
    "deepseek-v4-pro": "https://api.deepseek.com",
    "deepseek-v4-flash": "https://api.deepseek.com",
}


def suitability(model: str, tier: TierName, thinking: bool = False) -> Suitability:
    rated = MODEL_SUITABILITY.get(model)
    if rated is None:
        return "ok"
    return rated.verdict(tier, thinking)


def _check_efforts_explicit_ascending(efforts: tuple[str, ...]) -> None:
    """Model-catalog efforts are a hard provider fact, not policy — an
    explicit array, listed in ascending EFFORT_LADDER order, gaps allowed
    (a real model may lack e.g. `xhigh`). Distinct from a component's
    `TierPolicy.allowed`, which expresses policy intent (which tiers a
    component may run at), not a provider capability (plan 28 decision 3a)."""
    if not efforts:
        raise ValueError("efforts must be a non-empty array")
    indices = [_effort_index(e) for e in efforts]
    if indices != sorted(indices) or len(set(indices)) != len(indices):
        raise ValueError(f"efforts must be listed in ascending order with no duplicates, got {efforts}")


@dataclass(frozen=True)
class ModelCatalogEntry:
    """One model known to Gekai (plan 28 Phase 1a): the durable registry
    entry, independent of whether/where it's currently bound to a tier.
    `base_url=None` means "not yet configured" — deployment-specific
    (a proxy/gateway URL), never guessed or fabricated by the seed catalog."""
    name: str
    base_url: str | None
    efforts: tuple[str, ...]  # explicit, ascending — plan 28 decision 3a
    thinking: bool
    suitability: TierSuitability = TierSuitability()

    def __post_init__(self) -> None:
        _check_efforts_explicit_ascending(self.efforts)
        if not self.thinking and self.suitability.core_thinking is not None:
            raise ValueError(f"model {self.name!r} has thinking=False but declares a core_thinking suitability override")


@dataclass(frozen=True)
class TierBinding:
    """What a tier (FAST/SUPP/CORE) currently resolves to: a model from the
    catalog plus a default effort and a thinking flag. No base_url/creds
    here — base_url comes from the catalog entry (keyed by `model`),
    credentials from the keyring (keyed by `credentials.credential_key`:
    tier-model-effort-thinking, so tiers sharing a model, or a tier whose
    effort/thinking changes, still hold independent keys)."""
    model: str
    default_effort: str
    thinking: bool = False

    def __post_init__(self) -> None:
        _effort_index(self.default_effort)


def validate_binding(tier: TierName, binding: TierBinding, catalog: dict[str, ModelCatalogEntry]) -> None:
    """Fail loud (plan 28: chat-visible error, never a silent fallback) if a
    binding references a model absent from the catalog, an effort the model
    doesn't declare, or thinking outside CORE / on a non-thinking model."""
    entry = catalog.get(binding.model)
    if entry is None:
        raise ValueError(f"tier {tier.value} is bound to model {binding.model!r}, which is not in the catalog")
    if binding.default_effort not in entry.efforts:
        raise ValueError(
            f"tier {tier.value}'s default_effort {binding.default_effort!r} is not among "
            f"{binding.model!r}'s declared efforts {entry.efforts}"
        )
    if binding.thinking:
        if tier is not TierName.CORE:
            raise ValueError(f"tier {tier.value} binding has thinking=True; thinking is CORE-only")
        if not entry.thinking:
            raise ValueError(f"tier {tier.value} is bound to {binding.model!r}, which does not support thinking")


def _build_default_catalog() -> tuple[ModelCatalogEntry, ...]:
    """The code-side seed catalog (plan 28 decision 3): every model already
    known to model_caps.py, given a `base_url=None` (deployment-specific —
    a proxy/gateway URL we cannot guess, never fabricated here) and the full
    effort ladder (no known gaps for these models today; a future entry with
    a real gap declares its own explicit, shorter `efforts` tuple)."""
    from .model_caps import MODEL_CAPS  # same package; no cycle (model_caps.py has no tiers.py dependency)
    return tuple(
        ModelCatalogEntry(
            name=name,
            base_url=_KNOWN_BASE_URLS.get(name),
            efforts=EFFORT_LADDER,
            thinking=caps.thinking,
            suitability=MODEL_SUITABILITY.get(name, TierSuitability()),
        )
        for name, caps in MODEL_CAPS.items()
    )


DEFAULT_MODEL_CATALOG: tuple[ModelCatalogEntry, ...] = _build_default_catalog()


__all__ = [
    "EFFORT_LADDER",
    "TierName",
    "TierPolicy",
    "Suitability",
    "TierSuitability",
    "MODEL_SUITABILITY",
    "ModelCatalogEntry",
    "TierBinding",
    "DEFAULT_MODEL_CATALOG",
    "suitability",
    "validate_binding",
]
