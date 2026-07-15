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
    # (o-series, DeepSeek); such a model can never be bound to FAST/SUPP.
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
    "deepseek-v4-pro":   ModelCaps(thinking=True, thinking_style="deepseek", default_effort="high",
                                    can_disable_thinking=False, effort_requires_thinking=True),
    # this project's actual SUPP-tier model — never observed with thinking
    # enabled (absent here previously meant resolve_thinking_params silently
    # no-op'd for it; explicit now).
    "deepseek-v4-flash": ModelCaps(),
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


def resolve_thinking_params(model: str, effort: str | None = None) -> dict:
    """Return extra_params to enable thinking for the given model and effort.

    Returns empty dict if the model is not in MODEL_CAPS or does not support thinking.
    """
    caps = MODEL_CAPS.get(model)
    if caps is None or not caps.thinking or caps.thinking_style is None:
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
    mode (`can_disable_thinking=False`, e.g. o-series/DeepSeek) and
    `thinking=False` is requested. Not wired into any dispatch path yet —
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
