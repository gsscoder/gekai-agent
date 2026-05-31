from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCaps:
    thinking: bool = False
    thinking_style: str | None = None  # "openai" | "deepseek" | "anthropic"
    default_effort: str = "medium"


MODEL_CAPS: dict[str, ModelCaps] = {
    "deepseek-v4-pro":   ModelCaps(thinking=True, thinking_style="deepseek", default_effort="high"),
    "o3":                ModelCaps(thinking=True, thinking_style="openai"),
    "o4-mini":           ModelCaps(thinking=True, thinking_style="openai"),
    "o3-mini":           ModelCaps(thinking=True, thinking_style="openai"),
    "claude-opus-4-8":   ModelCaps(thinking=True, thinking_style="anthropic"),
    "claude-sonnet-4-6": ModelCaps(thinking=True, thinking_style="anthropic"),
}

_EFFORT_TO_PARAMS: dict[str, dict[str, dict]] = {
    "openai": {
        "low":    {"reasoning_effort": "low"},
        "medium": {"reasoning_effort": "medium"},
        "high":   {"reasoning_effort": "high"},
        "xhigh":  {"reasoning_effort": "high"},
        "max":    {"reasoning_effort": "high"},
    },
    "deepseek": {
        # DeepSeek API maps low/medium → high, xhigh → max internally.
        # We align our map to the two real values to be explicit.
        "low":    {"reasoning_effort": "high"},
        "medium": {"reasoning_effort": "high"},
        "high":   {"reasoning_effort": "high"},
        "xhigh":  {"reasoning_effort": "max"},
        "max":    {"reasoning_effort": "max"},
    },
    "anthropic": {
        "low":    {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        "medium": {"thinking": {"type": "enabled", "budget_tokens": 8000}},
        "high":   {"thinking": {"type": "enabled", "budget_tokens": 16000}},
        "xhigh":  {"thinking": {"type": "enabled", "budget_tokens": 32000}},
        "max":    {"thinking": {"type": "enabled", "budget_tokens": 32768}},
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


__all__ = ["ModelCaps", "MODEL_CAPS", "resolve_thinking_params"]
