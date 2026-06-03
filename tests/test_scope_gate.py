import json
import pytest
from pathlib import Path

from agent.router import evaluate_single_order_gate, Intent
from agent.settings import validate_gate_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_project(tmp_path: Path, data: dict) -> None:
    p = tmp_path / ".gekai" / "settings.local.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def _write_global(home: Path, data: dict) -> None:
    p = home / ".gekai" / "settings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# evaluate_single_order_gate
# ---------------------------------------------------------------------------

class TestEvaluateSingleOrderGate:
    def test_single_action_passes(self):
        segments = [(Intent.ACTION, "refactor auth")]
        rejected, reason = evaluate_single_order_gate(segments)
        assert not rejected
        assert reason is None

    def test_single_chat_passes(self):
        segments = [(Intent.CHAT, "explain auth")]
        rejected, reason = evaluate_single_order_gate(segments)
        assert not rejected
        assert reason is None

    def test_two_actions_rejected(self):
        segments = [(Intent.ACTION, "refactor auth"), (Intent.ACTION, "fix tests")]
        rejected, reason = evaluate_single_order_gate(segments)
        assert rejected
        assert "2" in reason

    def test_two_chats_rejected(self):
        segments = [(Intent.CHAT, "explain auth"), (Intent.CHAT, "explain db")]
        rejected, reason = evaluate_single_order_gate(segments)
        assert rejected
        assert "2" in reason

    def test_chat_plus_action_rejected(self):
        segments = [(Intent.CHAT, "explain auth"), (Intent.ACTION, "refactor auth")]
        rejected, reason = evaluate_single_order_gate(segments)
        assert rejected
        assert "2" in reason

    def test_three_actions_rejected(self):
        segments = [
            (Intent.ACTION, "refactor auth"),
            (Intent.ACTION, "fix tests"),
            (Intent.ACTION, "update docs"),
        ]
        rejected, reason = evaluate_single_order_gate(segments)
        assert rejected
        assert "3" in reason

    def test_empty_segments_passes(self):
        rejected, reason = evaluate_single_order_gate([])
        assert not rejected
        assert reason is None


# ---------------------------------------------------------------------------
# validate_gate_config
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("agent.settings.Path.home", staticmethod(lambda: home))
    return home


class TestValidateGateConfig:
    def test_no_files_no_errors(self, tmp_path, fake_home):
        assert validate_gate_config(tmp_path) == []

    def test_valid_scope_gate_no_errors(self, tmp_path, fake_home):
        _write_project(tmp_path, {"scope_gate": True})
        assert validate_gate_config(tmp_path) == []

    def test_scope_gate_false_is_valid(self, tmp_path, fake_home):
        _write_project(tmp_path, {"scope_gate": False})
        assert validate_gate_config(tmp_path) == []

    def test_scope_gate_string_is_error(self, tmp_path, fake_home):
        _write_project(tmp_path, {"scope_gate": "true"})
        errors = validate_gate_config(tmp_path)
        assert len(errors) == 1
        assert "scope_gate" in errors[0]

    def test_scope_gate_int_is_error(self, tmp_path, fake_home):
        _write_project(tmp_path, {"scope_gate": 1})
        errors = validate_gate_config(tmp_path)
        assert len(errors) == 1
        assert "scope_gate" in errors[0]

    def test_unknown_fields_ignored(self, tmp_path, fake_home):
        _write_project(tmp_path, {"scope_gate": True, "max_prompt_segments": 5})
        assert validate_gate_config(tmp_path) == []

    def test_malformed_json_skipped_no_crash(self, tmp_path, fake_home):
        p = tmp_path / ".gekai" / "settings.local.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not valid json")
        assert validate_gate_config(tmp_path) == []

    def test_malformed_global_json_skipped_no_crash(self, tmp_path, fake_home):
        _write_global(fake_home, {})
        gp = fake_home / ".gekai" / "settings.json"
        gp.write_text("{bad")
        assert validate_gate_config(tmp_path) == []

    def test_invalid_in_both_files_both_collected(self, tmp_path, fake_home):
        _write_project(tmp_path, {"scope_gate": "yes"})
        _write_global(fake_home, {"scope_gate": 0})
        errors = validate_gate_config(tmp_path)
        assert len(errors) == 2
