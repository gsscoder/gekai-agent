"""Coverage for the `/models` grid's pure logic helpers in `agent/tui/app.py`:
`_mask_key` (key-display masking) and `_models_display_key`/
`_models_key_present` (precedence between an in-progress edit and the real
keyring credential).

There is no Pilot-based UI test in this repo; the interactive key-event
cascade (`_handle_models_enter`, `action_navigate_*`,
`action_confirm_or_submit`, `action_cancel_stream`) requires a live `App` to
drive and is verified by trace-through only, mirroring how
tests/test_models_panel.py covers `ModelsPanel` itself without mounting.
"""

from __future__ import annotations

import pytest

from agent.tui.app import _mask_key, _models_display_key, _models_key_present


# ---- _mask_key -------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("sk-1234567890abcdef", "sk" + "*" * 6 + "def"),
        ("sk123", "*" * 6),
        ("ab", "*" * 6),
        ("sk-123", "sk" + "*" * 6 + "123"),
    ],
)
def test_mask_key(raw: str, expected: str) -> None:
    assert _mask_key(raw) == expected


# ---- _models_display_key / _models_key_present ------------------------


def test_display_key_no_key_when_nothing_staged_or_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: False)
    assert _models_display_key("openai:m", {}) == "no key"
    assert _models_key_present("openai:m", {}) is False


def test_display_key_masks_real_stored_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: True)
    monkeypatch.setattr("agent.tui.app.credentials.get_api_key", lambda name: "sk-1234567890abcdef")
    assert _models_display_key("openai:m", {}) == "sk" + "*" * 6 + "def"
    assert _models_key_present("openai:m", {}) is True


def test_display_key_staged_edit_wins_over_real_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: True)
    monkeypatch.setattr("agent.tui.app.credentials.get_api_key", lambda name: "sk-old-real-key-value")
    staged = {"openai:m": "sk-new-staged-key"}
    assert _models_display_key("openai:m", staged) == _mask_key("sk-new-staged-key")
    assert _models_key_present("openai:m", staged) is True


def test_display_key_explicit_clear_wins_over_real_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real keyring credential exists, but the key cell was emptied and
    # confirmed — the row must report "no key" until committed.
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: True)
    assert _models_display_key("openai:m", {"openai:m": ""}) == "no key"
    assert _models_key_present("openai:m", {"openai:m": ""}) is False


def test_staged_key_is_scoped_to_its_own_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent.tui.app.credentials.has_api_key", lambda name: False)
    staged = {"openai:flash": "sk-flash-key"}
    assert _models_key_present("openai:flash", staged) is True
    assert _models_key_present("openai:pro", staged) is False
