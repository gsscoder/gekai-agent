from __future__ import annotations

import pytest

from agent.llm.model_caps import MODEL_CAPS, ModelCaps, resolve_thinking_params, thinking_realizable


@pytest.mark.parametrize(
    "model_name, can_disable_thinking, expected_disabled, expected_enabled",
    [
        ("reasoning-only-model", False, False, True),
        ("flexible-model", True, True, True),
    ],
)
def test_thinking_realizable_respects_can_disable_thinking(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    can_disable_thinking: bool,
    expected_disabled: bool,
    expected_enabled: bool,
) -> None:
    monkeypatch.setitem(MODEL_CAPS, model_name, ModelCaps(thinking=True, can_disable_thinking=can_disable_thinking))
    assert thinking_realizable(model_name, thinking=False) is expected_disabled
    assert thinking_realizable(model_name, thinking=True) is expected_enabled


def test_unknown_model_only_realizes_non_thinking() -> None:
    assert thinking_realizable("some-unlisted-model", thinking=False) is True
    assert thinking_realizable("some-unlisted-model", thinking=True) is False


def test_deepseek_v4_pro_can_disable_thinking() -> None:
    # Live probe (real credentials, real sequencer prompt, 3 runs/variant)
    # confirmed deepseek-v4-pro returns reasoning_content of length 0 under
    # `thinking: {"type": "disabled"}` and still produces valid output — it
    # is not a reasoning-only model like o-series.
    assert thinking_realizable("deepseek-v4-pro", thinking=False) is True


def test_resolve_thinking_params_disabled_deepseek_style_is_explicit_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # plan 34 phase 1: `enabled=False` must resolve to an explicit disable
    # payload, not `{}` — a bare `{}` is "unspecified" to DeepSeek, which
    # then defaults to reasoning ON.
    monkeypatch.setitem(MODEL_CAPS, "some-deepseek-model", ModelCaps(thinking=False, thinking_style="deepseek"))
    assert resolve_thinking_params("some-deepseek-model", enabled=False) == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }


def test_resolve_thinking_params_disabled_unknown_style_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # No known provider-specific disable realization for other styles yet —
    # falls back to `{}` rather than inventing an untested payload.
    monkeypatch.setitem(MODEL_CAPS, "other-style-model", ModelCaps(thinking=False, thinking_style="openai"))
    assert resolve_thinking_params("other-style-model", enabled=False) == {}


def test_resolve_thinking_params_enabled_unchanged_for_thinking_model() -> None:
    # Regression guard: a thinking=True model's enable resolution is
    # byte-identical to before the `enabled` param existed.
    assert resolve_thinking_params("deepseek-v4-pro", "high") == {
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    assert resolve_thinking_params("deepseek-v4-pro", "high", enabled=True) == resolve_thinking_params(
        "deepseek-v4-pro", "high"
    )


def test_resolve_thinking_params_unknown_model_is_empty_regardless_of_enabled() -> None:
    assert resolve_thinking_params("unlisted-model") == {}
    assert resolve_thinking_params("unlisted-model", enabled=False) == {}
