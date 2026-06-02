import json
import pytest
from pathlib import Path

from agent.router import _word_count, evaluate_structural_gate, Intent
from agent.settings import (
    validate_gate_config,
    load_max_prompt_segments,
    load_big_prompt_min_size_words,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(text: str) -> tuple[Intent, str]:
    return (Intent.CHAT, text)


def _segs(*texts: str) -> list[tuple[Intent, str]]:
    return [_seg(t) for t in texts]


def _words(n: int) -> str:
    return " ".join(["word"] * n)


def _write_project(tmp_path: Path, data: dict) -> None:
    p = tmp_path / ".gekai" / "settings.local.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def _write_global(home: Path, data: dict) -> None:
    p = home / ".gekai" / "settings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# _word_count
# ---------------------------------------------------------------------------

class TestWordCount:
    def test_empty_string(self):
        assert _word_count("") == 0

    def test_entirely_code_fence(self):
        assert _word_count("```hello world foo```") == 0

    def test_prose_before_fence(self):
        assert _word_count("hello world\n```code here```") == 2

    def test_prose_after_fence(self):
        assert _word_count("```code here``` hello world") == 2

    def test_prose_both_sides_of_fence(self):
        assert _word_count("one two ```code``` three four") == 4

    def test_operators_inside_fence_not_counted(self):
        assert _word_count("```x -> y // z === w```") == 0

    def test_operators_outside_fence_not_counted(self):
        # -> // === are not \w+, so they contribute 0 words
        assert _word_count("x -> y") == 2

    def test_underscored_identifier_counts_as_one_word(self):
        assert _word_count("my_func") == 1

    def test_multiple_fences_all_stripped(self):
        assert _word_count("```a b``` middle ```c d```") == 1

    def test_prose_only_no_fence(self):
        assert _word_count("the quick brown fox") == 4

    def test_fence_with_newlines_stripped(self):
        assert _word_count("before\n```\nsome code\nmore code\n```\nafter") == 2


# ---------------------------------------------------------------------------
# evaluate_structural_gate
# ---------------------------------------------------------------------------

class TestEvaluateStructuralGate:
    def test_single_segment_tiny_always_passes(self):
        rejected, reason = evaluate_structural_gate(_segs("hi"))
        assert not rejected
        assert reason is None

    def test_single_segment_large_always_passes(self):
        rejected, reason = evaluate_structural_gate(_segs(_words(200)))
        assert not rejected
        assert reason is None

    def test_exactly_max_segments_passes(self):
        rejected, _ = evaluate_structural_gate(_segs("a", "b", "c"), max_segments=3)
        assert not rejected

    def test_max_segments_plus_one_rejects(self):
        rejected, reason = evaluate_structural_gate(_segs("a", "b", "c", "d"), max_segments=3)
        assert rejected
        assert reason == "Too many tasks (4)"

    def test_rejection_reason_contains_count(self):
        rejected, reason = evaluate_structural_gate(_segs("a", "b", "c", "d", "e"), max_segments=3)
        assert rejected
        assert reason == "Too many tasks (5)"

    def test_two_segments_both_large_still_passes(self):
        big = _words(200)
        rejected, reason = evaluate_structural_gate(_segs(big, big), big_prompt_min_words=50)
        assert not rejected
        assert reason is None

    def test_three_segments_two_big_rejects(self):
        big = _words(51)
        rejected, reason = evaluate_structural_gate(
            _segs(big, big, "small"), big_prompt_min_words=50
        )
        assert rejected
        assert reason == "Multiple complex tasks"

    def test_three_segments_one_big_passes(self):
        big = _words(51)
        rejected, _ = evaluate_structural_gate(
            _segs(big, "small", "also small"), big_prompt_min_words=50
        )
        assert not rejected

    def test_three_segments_zero_big_passes(self):
        rejected, _ = evaluate_structural_gate(
            _segs("a", "b", "c"), big_prompt_min_words=50
        )
        assert not rejected

    def test_exactly_min_words_not_big(self):
        exactly_min = _words(50)
        rejected, _ = evaluate_structural_gate(
            _segs(exactly_min, exactly_min, exactly_min), big_prompt_min_words=50
        )
        assert not rejected

    def test_min_words_plus_one_is_big(self):
        just_over = _words(51)
        rejected, reason = evaluate_structural_gate(
            _segs(just_over, just_over, "small"), big_prompt_min_words=50
        )
        assert rejected
        assert reason == "Multiple complex tasks"

    def test_max_segments_one_single_segment_passes(self):
        rejected, _ = evaluate_structural_gate(_segs("x"), max_segments=1)
        assert not rejected

    def test_max_segments_one_two_segments_rejects(self):
        rejected, reason = evaluate_structural_gate(_segs("x", "y"), max_segments=1)
        assert rejected
        assert reason == "Too many tasks (2)"

    def test_custom_big_prompt_threshold(self):
        words_100 = _words(100)
        rejected, reason = evaluate_structural_gate(
            _segs(words_100, words_100, "small"), big_prompt_min_words=99
        )
        assert rejected
        assert reason == "Multiple complex tasks"

    def test_custom_big_prompt_threshold_not_exceeded(self):
        words_99 = _words(99)
        rejected, _ = evaluate_structural_gate(
            _segs(words_99, words_99, "small"), big_prompt_min_words=100
        )
        assert not rejected


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

    def test_valid_fields_no_errors(self, tmp_path, fake_home):
        _write_project(tmp_path, {
            "max_prompt_segments": 3,
            "big_prompt_min_size_words": 100,
            "scope_gate": True,
        })
        assert validate_gate_config(tmp_path) == []

    def test_max_prompt_segments_zero_is_error(self, tmp_path, fake_home):
        _write_project(tmp_path, {"max_prompt_segments": 0})
        errors = validate_gate_config(tmp_path)
        assert len(errors) == 1
        assert "max_prompt_segments" in errors[0]

    def test_max_prompt_segments_ten_is_error(self, tmp_path, fake_home):
        _write_project(tmp_path, {"max_prompt_segments": 10})
        errors = validate_gate_config(tmp_path)
        assert len(errors) == 1
        assert "max_prompt_segments" in errors[0]

    def test_max_prompt_segments_one_is_valid(self, tmp_path, fake_home):
        _write_project(tmp_path, {"max_prompt_segments": 1})
        assert validate_gate_config(tmp_path) == []

    def test_max_prompt_segments_nine_is_valid(self, tmp_path, fake_home):
        _write_project(tmp_path, {"max_prompt_segments": 9})
        assert validate_gate_config(tmp_path) == []

    def test_big_prompt_min_size_words_49_is_error(self, tmp_path, fake_home):
        _write_project(tmp_path, {"big_prompt_min_size_words": 49})
        errors = validate_gate_config(tmp_path)
        assert len(errors) == 1
        assert "big_prompt_min_size_words" in errors[0]

    def test_big_prompt_min_size_words_201_is_error(self, tmp_path, fake_home):
        _write_project(tmp_path, {"big_prompt_min_size_words": 201})
        errors = validate_gate_config(tmp_path)
        assert len(errors) == 1
        assert "big_prompt_min_size_words" in errors[0]

    def test_big_prompt_min_size_words_50_is_valid(self, tmp_path, fake_home):
        _write_project(tmp_path, {"big_prompt_min_size_words": 50})
        assert validate_gate_config(tmp_path) == []

    def test_big_prompt_min_size_words_200_is_valid(self, tmp_path, fake_home):
        _write_project(tmp_path, {"big_prompt_min_size_words": 200})
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

    def test_scope_gate_bool_is_valid(self, tmp_path, fake_home):
        _write_project(tmp_path, {"scope_gate": True})
        assert validate_gate_config(tmp_path) == []

    def test_scope_gate_false_is_valid(self, tmp_path, fake_home):
        _write_project(tmp_path, {"scope_gate": False})
        assert validate_gate_config(tmp_path) == []

    def test_two_invalid_fields_in_one_file_both_collected(self, tmp_path, fake_home):
        _write_project(tmp_path, {"max_prompt_segments": 0, "big_prompt_min_size_words": 49})
        errors = validate_gate_config(tmp_path)
        assert len(errors) == 2
        assert any("max_prompt_segments" in e for e in errors)
        assert any("big_prompt_min_size_words" in e for e in errors)

    def test_invalid_fields_in_both_files_all_collected(self, tmp_path, fake_home):
        _write_project(tmp_path, {"max_prompt_segments": 0})
        _write_global(fake_home, {"big_prompt_min_size_words": 49})
        errors = validate_gate_config(tmp_path)
        assert len(errors) == 2
        assert any("max_prompt_segments" in e for e in errors)
        assert any("big_prompt_min_size_words" in e for e in errors)

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


# ---------------------------------------------------------------------------
# load_max_prompt_segments
# ---------------------------------------------------------------------------

class TestLoadMaxPromptSegments:
    def test_no_config_returns_default(self, tmp_path, fake_home):
        assert load_max_prompt_segments(tmp_path) == 3

    def test_valid_project_config_returned(self, tmp_path, fake_home):
        _write_project(tmp_path, {"max_prompt_segments": 5})
        assert load_max_prompt_segments(tmp_path) == 5

    def test_valid_global_config_returned_when_no_project(self, tmp_path, fake_home):
        _write_global(fake_home, {"max_prompt_segments": 7})
        assert load_max_prompt_segments(tmp_path) == 7

    def test_project_takes_priority_over_global(self, tmp_path, fake_home):
        _write_project(tmp_path, {"max_prompt_segments": 2})
        _write_global(fake_home, {"max_prompt_segments": 8})
        assert load_max_prompt_segments(tmp_path) == 2

    def test_invalid_project_falls_through_to_global(self, tmp_path, fake_home):
        _write_project(tmp_path, {"max_prompt_segments": 0})
        _write_global(fake_home, {"max_prompt_segments": 6})
        assert load_max_prompt_segments(tmp_path) == 6

    def test_invalid_in_both_returns_default(self, tmp_path, fake_home):
        _write_project(tmp_path, {"max_prompt_segments": 0})
        _write_global(fake_home, {"max_prompt_segments": 10})
        assert load_max_prompt_segments(tmp_path) == 3

    def test_edge_value_one(self, tmp_path, fake_home):
        _write_project(tmp_path, {"max_prompt_segments": 1})
        assert load_max_prompt_segments(tmp_path) == 1

    def test_edge_value_nine(self, tmp_path, fake_home):
        _write_project(tmp_path, {"max_prompt_segments": 9})
        assert load_max_prompt_segments(tmp_path) == 9


# ---------------------------------------------------------------------------
# load_big_prompt_min_size_words
# ---------------------------------------------------------------------------

class TestLoadBigPromptMinSizeWords:
    def test_no_config_returns_default(self, tmp_path, fake_home):
        assert load_big_prompt_min_size_words(tmp_path) == 50

    def test_valid_project_config_returned(self, tmp_path, fake_home):
        _write_project(tmp_path, {"big_prompt_min_size_words": 100})
        assert load_big_prompt_min_size_words(tmp_path) == 100

    def test_valid_global_config_returned_when_no_project(self, tmp_path, fake_home):
        _write_global(fake_home, {"big_prompt_min_size_words": 150})
        assert load_big_prompt_min_size_words(tmp_path) == 150

    def test_project_takes_priority_over_global(self, tmp_path, fake_home):
        _write_project(tmp_path, {"big_prompt_min_size_words": 75})
        _write_global(fake_home, {"big_prompt_min_size_words": 180})
        assert load_big_prompt_min_size_words(tmp_path) == 75

    def test_invalid_project_falls_through_to_global(self, tmp_path, fake_home):
        _write_project(tmp_path, {"big_prompt_min_size_words": 49})
        _write_global(fake_home, {"big_prompt_min_size_words": 120})
        assert load_big_prompt_min_size_words(tmp_path) == 120

    def test_invalid_in_both_returns_default(self, tmp_path, fake_home):
        _write_project(tmp_path, {"big_prompt_min_size_words": 49})
        _write_global(fake_home, {"big_prompt_min_size_words": 201})
        assert load_big_prompt_min_size_words(tmp_path) == 50

    def test_edge_value_50(self, tmp_path, fake_home):
        _write_project(tmp_path, {"big_prompt_min_size_words": 50})
        assert load_big_prompt_min_size_words(tmp_path) == 50

    def test_edge_value_200(self, tmp_path, fake_home):
        _write_project(tmp_path, {"big_prompt_min_size_words": 200})
        assert load_big_prompt_min_size_words(tmp_path) == 200
