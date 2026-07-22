from __future__ import annotations

import pytest

from agent.llm.model_caps import MODEL_CAPS, ModelCaps, thinking_realizable


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
