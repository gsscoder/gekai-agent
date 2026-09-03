"""Model-tier data types and the code-side model catalog.

A tier is a capability contract, not a model — the same model may occupy
multiple tiers at different operating points. `store.py` persists which model
fills each tier; `resolve.py` turns a tier into runnable API params.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from ..llm.model_caps import MODEL_CAPS

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
    any assignment-time scaling. A tier binding decides WHICH model (and
    credential) fills the slot; effort/thinking default to that binding's
    values but are a per-component axis after all — a touchpoint may declare
    its own (`Touchpoint.effort`/`.thinking`), because how hard to run a model
    for one specific harness job is an engineering decision belonging next to
    the touchpoint, not a knob a user retunes in `/tier` per workload (the
    sequencer is the standing case: CORE's model at effort=high, thinking
    off). A degenerate space (len(allowed) == 1) is how a non-scalable
    component is expressed — no separate freeze flag."""
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
    # A live probe (see model_caps.py's MODEL_CAPS comment) confirmed
    # deepseek-v4-pro CAN disable thinking (can_disable_thinking=True), so
    # FAST/SUPP are no longer structurally unrealizable — these ratings are a
    # cost/latency judgment (this model is ~7x slower with thinking on, and
    # thinking-off behavior at these tiers hasn't been evaluated), not a
    # realizability one. Left as deprecated/warning pending that evaluation.
    "deepseek-v4-pro":    TierSuitability(fast="deprecated", supp="warning", core="ok"),
    # this project's actual FAST- and SUPP-tier model.
    "deepseek-v4-flash":  TierSuitability(fast="ok", supp="ok", core="warning"),
}

# Wire protocols Gekai can actually speak. "openai" is the OpenAI-compatible
# chat-completions shape (agent/llm/providers/openai.py) — the vendor behind
# it varies (DeepSeek, QwenCloud), which is why base_url and credentials stay
# per-model rather than per-provider.
SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai",)

# Models with a genuinely public, documented canonical endpoint — filled in
# from fact (confirmed via this project's own tests/.env.test), never guessed.
_KNOWN_BASE_URLS: dict[str, str] = {
    "deepseek-v4-pro": "https://api.deepseek.com",
    "deepseek-v4-flash": "https://api.deepseek.com",
    "qwen3.8-max": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
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
    (a proxy/gateway URL), never guessed or fabricated by the seed catalog.
    `provider` names the wire protocol that handles this model; it is what
    `/models` groups by and what `credentials.credential_key` namespaces
    with, so a model is never usable through a protocol Gekai can't speak."""
    name: str
    base_url: str | None
    efforts: tuple[str, ...]  # explicit, ascending — plan 28 decision 3a
    thinking: bool
    suitability: TierSuitability = TierSuitability()
    provider: str = "openai"

    def __post_init__(self) -> None:
        _check_efforts_explicit_ascending(self.efforts)
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"model {self.name!r} declares provider {self.provider!r}; supported: {SUPPORTED_PROVIDERS}")
        if not self.thinking and self.suitability.core_thinking is not None:
            raise ValueError(f"model {self.name!r} has thinking=False but declares a core_thinking suitability override")


@dataclass(frozen=True)
class TierBinding:
    """What a tier (FAST/SUPP/CORE) currently resolves to: a model from the
    catalog plus a default effort and a thinking flag. No base_url/creds
    here — base_url comes from the catalog entry (keyed by `model`),
    credentials from the keyring (keyed by `credentials.credential_key`:
    provider:model, so retuning a tier's effort/thinking never invalidates a
    stored key, and two tiers on the same model share one)."""
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
    "SUPPORTED_PROVIDERS",
    "ModelCatalogEntry",
    "TierBinding",
    "DEFAULT_MODEL_CATALOG",
    "suitability",
    "validate_binding",
]
