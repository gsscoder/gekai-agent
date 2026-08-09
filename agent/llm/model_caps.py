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
    # Doc-derived, NOT live-probe-confirmed (no credentials exercised this
    # session) — facts come from QwenCloud's official API reference
    # (https://docs.qwencloud.com/api-reference/chat/openai-chat).
    # thinking=True, default_effort="xhigh": the doc states qwen3.8-max is a
    # thinking model enabled by default, and its own `reasoning_effort`
    # default is "xhigh". can_disable_thinking=True: the doc documents an
    # explicit disable path, `reasoning_effort="none"` (this model is NOT in
    # the doc's `enable_thinking` boolean-toggle family — that path doesn't
    # apply here). effort_requires_thinking=False: low/medium/xhigh remain
    # distinct native values while thinking stays on, so the ladder has an
    # observable realization independent of the thinking on/off axis.
    # thinking_style="qwen-max" (distinct from "deepseek"): the wire shape
    # differs — DeepSeek's `reasoning_effort` is a bare top-level kwarg, but
    # the doc says qwen3.8-max's `reasoning_effort` is "not a standard OpenAI
    # parameter" and must be nested under `extra_body` instead.
    "qwen3.8-max": ModelCaps(thinking=True, thinking_style="qwen-max", default_effort="xhigh",
                              can_disable_thinking=True, effort_requires_thinking=False),
}

_EFFORT_TO_PARAMS: dict[str, dict[str, dict]] = {
    # Keyed by model id, NOT thinking_style. DeepSeek's real API folds
    # requested effort to actual effort differently per model (confirmed
    # against https://api-docs.deepseek.com/guides/thinking_mode) — a single
    # shared "deepseek" bucket was wrong: it silently applied pro's fold-down
    # (low/medium → high) to flash too, when flash's API actually honors a
    # bare "low". Each model below gets its own complete fold-down map
    # covering all 5 canonical EFFORT_LADDER rungs (tiers.py) — "medium" has
    # no direct DeepSeek wire value at all (the API only accepts
    # low/high/xhigh/max), so it is folded UP to the nearest defined rung
    # rather than passed through. Any rung above a model's ceiling (e.g.
    # flash caps at "high") folds down to that ceiling — the same pattern a
    # future lower-ceiling model should follow for its own xhigh/max entries.
    #
    # qwen3.8-max's entry is shaped differently on purpose: each value below
    # is already a self-contained `{"extra_body": {"reasoning_effort": ...}}`
    # dict, not a bare `{"reasoning_effort": ...}` like DeepSeek's entries.
    # That's because DeepSeek's `reasoning_effort` is a top-level kwarg (the
    # `extra_body` wrapper resolve_thinking_params adds afterward is only the
    # separate thinking-enable/disable payload), while QwenCloud's API
    # reference (https://docs.qwencloud.com/api-reference/chat/openai-chat)
    # states `reasoning_effort` is "not a standard OpenAI parameter" for this
    # model/provider and must itself be nested under `extra_body`. The fold
    # values are doc-derived: the model only accepts three native
    # `reasoning_effort` values (low/medium/xhigh), so the doc's own
    # "OpenAI standard value mapping" (high->xhigh, max->xhigh) is used to
    # fill the remaining two EFFORT_LADDER rungs.
    "deepseek-v4-flash": {
        "low":    {"reasoning_effort": "low"},
        "medium": {"reasoning_effort": "high"},
        "high":   {"reasoning_effort": "high"},
        "xhigh":  {"reasoning_effort": "high"},
        "max":    {"reasoning_effort": "max"},
    },
    "deepseek-v4-pro": {
        "low":    {"reasoning_effort": "high"},
        "medium": {"reasoning_effort": "high"},
        "high":   {"reasoning_effort": "high"},
        "xhigh":  {"reasoning_effort": "max"},
        "max":    {"reasoning_effort": "max"},
    },
    "qwen3.8-max": {
        "low":    {"extra_body": {"reasoning_effort": "low"}},
        "medium": {"extra_body": {"reasoning_effort": "medium"}},
        "high":   {"extra_body": {"reasoning_effort": "xhigh"}},
        "xhigh":  {"extra_body": {"reasoning_effort": "xhigh"}},
        "max":    {"extra_body": {"reasoning_effort": "xhigh"}},
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
        if caps.thinking_style == "qwen-max":
            return {"extra_body": {"reasoning_effort": "none"}}
        return {}
    if not caps.thinking:
        return {}
    effective_effort = effort or caps.default_effort
    model_map = _EFFORT_TO_PARAMS.get(model, {})
    params = dict(model_map.get(effective_effort, model_map.get("high", {})))
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
