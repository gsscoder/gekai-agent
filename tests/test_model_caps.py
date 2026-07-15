from __future__ import annotations

import pytest

from agent.llm.model_caps import MODEL_CAPS, ModelCaps, thinking_realizable


def test_reasoning_only_model_cannot_disable_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(MODEL_CAPS, "reasoning-only-model", ModelCaps(thinking=True, can_disable_thinking=False))
    assert thinking_realizable("reasoning-only-model", thinking=False) is False
    assert thinking_realizable("reasoning-only-model", thinking=True) is True


def test_flexible_thinking_model_can_disable_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(MODEL_CAPS, "flexible-model", ModelCaps(thinking=True, can_disable_thinking=True))
    assert thinking_realizable("flexible-model", thinking=False) is True
    assert thinking_realizable("flexible-model", thinking=True) is True


def test_unknown_model_only_realizes_non_thinking() -> None:
    assert thinking_realizable("some-unlisted-model", thinking=False) is True
    assert thinking_realizable("some-unlisted-model", thinking=True) is False
