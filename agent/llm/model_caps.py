from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCaps:
    thinking: bool = False
    thinking_style: str | None = None  # "deepseek" for now; "openai"/"anthropic" retired until exercised
    default_effort: str = "medium"
    # Realizability (plan 28 hard problem 1): effort and thinking are not
    # independent axes on every provider. `can_disable_thinking=False` means
    # `reasoning_effort` IMPLIES thinking — there is no non-thinking mode
    # (o-series is the known example; DeepSeek was wrongly assumed to be one
    # too until a live probe — real credentials, real sequencer prompt, 3
    # runs/variant — showed `deepseek-v4-pro` returns `reasoning_content` of
    # length 0 under `extra_body={"thinking": {"type": "disabled"}}` and
    # still produces valid output, ~7x faster; see MODEL_CAPS below).
    # `effort_requires_thinking=True` means the effort ladder has no
    # observable realization without thinking — a non-thinking call carries
    # no effort-differentiated params (current Anthropic case: effort is
    # implemented purely as thinking budget in _EFFORT_TO_PARAMS below).
    can_disable_thinking: bool = True
    effort_requires_thinking: bool = False


MODEL_CAPS: dict[str, ModelCaps] = {
    # DeepSeek only for now (confirmed in this project's tests/.env.test) —
    # other providers need credentials this project hasn't tested against yet;
    # add them back once actually exercised, not as untested examples.
    # can_disable_thinking=True: confirmed by live probe (real credentials,
    # real sequencer prompt, 3 runs/variant) — `thinking: {"type": "disabled"}`
    # yields reasoning_content of length 0 and still valid task-graph output,
    # median 13.2s vs 90.7s with thinking on. effort_requires_thinking=False:
    # the same probe's two thinking-off variants (with vs. without an
    # explicit `reasoning_effort`) produced different, valid outputs, so the
    # effort ladder does have some observable realization without thinking —
    # unlike the Anthropic case this flag was written for.
    "deepseek-v4-pro":   ModelCaps(thinking=True, thinking_style="deepseek", default_effort="high",
                                    can_disable_thinking=True, effort_requires_thinking=False),
    # this project's actual FAST/SUPP-tier model — bound with thinking=False,
    # but empirically it reasons on every call unless explicitly told not to
    # (an absent `thinking` param means "unspecified" to the DeepSeek API,
    # which defaults to reasoning ON; plan 34 phase 1). `thinking_style` is
    # set so `resolve_thinking_params(..., enabled=False)` can find the
    # DeepSeek disable payload.
    "deepseek-v4-flash": ModelCaps(thinking_style="deepseek"),
}

_EFFORT_TO_PARAMS: dict[str, dict[str, dict]] = {
    # DeepSeek only for now — openai/anthropic styles retired until those
    # providers are actually exercised (see the MODEL_CAPS note above).
    "deepseek": {
        # DeepSeek API maps low/medium → high, xhigh → max internally.
        # We align our map to the two real values to be explicit.
        "low":    {"reasoning_effort": "high"},
        "medium": {"reasoning_effort": "high"},
        "high":   {"reasoning_effort": "high"},
        "xhigh":  {"reasoning_effort": "max"},
        "max":    {"reasoning_effort": "max"},
    },
}


def resolve_thinking_params(model: str, effort: str | None = None, *, enabled: bool = True) -> dict:
    """Return extra_params that realize the requested thinking state for the
    given model and effort.

    `enabled=True` (default): params to turn thinking on, as before.
    `enabled=False` (plan 34 phase 1): params to explicitly turn thinking
    off. A bare `{}` means "unspecified" to providers like DeepSeek, which
    then default to reasoning ON — so a `thinking: false` binding must
    resolve to an explicit disable payload, not an empty dict.

    Returns empty dict if the model is not in MODEL_CAPS, has no
    `thinking_style` (no known provider-specific realization), or (when
    `enabled=True`) does not support thinking at all.
    """
    caps = MODEL_CAPS.get(model)
    if caps is None or caps.thinking_style is None:
        return {}
    if not enabled:
        if caps.thinking_style == "deepseek":
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        return {}
    if not caps.thinking:
        return {}
    effective_effort = effort or caps.default_effort
    style_map = _EFFORT_TO_PARAMS.get(caps.thinking_style, {})
    params = dict(style_map.get(effective_effort, style_map.get("high", {})))
    if caps.thinking_style == "deepseek":
        params["extra_body"] = {"thinking": {"type": "enabled"}}
    return params


def thinking_realizable(model: str, thinking: bool) -> bool:
    """Pure check (plan 28 Phase 0, inert): can `model` actually run with the
    requested thinking state? False only when the model has no thinking-off
    mode (`can_disable_thinking=False`, e.g. o-series) and `thinking=False`
    is requested. Not wired into any dispatch path yet —
    Phase 1/2 consult this when resolving a component's tier point against
    real model realizability (hard problem 1).
    """
    caps = MODEL_CAPS.get(model)
    if caps is None or not caps.thinking:
        return not thinking  # unknown/non-thinking model: only thinking=False is realizable
    if not thinking:
        return caps.can_disable_thinking
    return True


__all__ = ["ModelCaps", "MODEL_CAPS", "resolve_thinking_params", "thinking_realizable"]
