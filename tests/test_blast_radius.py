from __future__ import annotations

from agent.blast_radius import _parse_blast_radius_output, count_blast_areas, evaluate_blast_radius_gate


def test_parse_well_formed_line() -> None:
    text = "src/main.py | calculator, arithmetic, compute"
    result = _parse_blast_radius_output(text)
    assert len(result) == 1
    path, keywords = result[0]
    assert path == "src/main.py"
    assert keywords == ["calculator", "arithmetic", "compute"]


def test_parse_multiple_lines() -> None:
    text = (
        "src/main.py | calculator, arithmetic\n"
        "frontend/PasscodeBox.tsx | passcode, pin, otp, dialog"
    )
    result = _parse_blast_radius_output(text)
    assert len(result) == 2
    assert result[0][0] == "src/main.py"
    assert result[1][0] == "frontend/PasscodeBox.tsx"
    assert "pin" in result[1][1]


def test_parse_skips_blank_lines() -> None:
    text = "\nsrc/a.py | alpha\n\nsrc/b.py | beta\n"
    result = _parse_blast_radius_output(text)
    assert len(result) == 2


def test_parse_skips_lines_without_pipe() -> None:
    text = (
        "some prose without a pipe\n"
        "src/main.py | valid, keywords\n"
        "another bad line"
    )
    result = _parse_blast_radius_output(text)
    assert len(result) == 1
    assert result[0][0] == "src/main.py"


def test_parse_empty_text() -> None:
    assert _parse_blast_radius_output("") == []


def test_parse_lowercases_keywords() -> None:
    text = "src/Auth.py | Login, TOKEN, Session"
    result = _parse_blast_radius_output(text)
    _, keywords = result[0]
    assert keywords == ["login", "token", "session"]


def test_parse_deduplicates_keywords() -> None:
    text = "src/a.py | alpha, alpha, beta, alpha"
    _, keywords = _parse_blast_radius_output(text)[0]
    assert keywords.count("alpha") == 1
    assert "beta" in keywords


def test_parse_strips_whitespace_from_keywords() -> None:
    text = "src/a.py |  login ,  token , session "
    _, keywords = _parse_blast_radius_output(text)[0]
    assert keywords == ["login", "token", "session"]


def test_parse_skips_line_with_empty_path() -> None:
    text = " | keyword1, keyword2"
    result = _parse_blast_radius_output(text)
    assert result == []


def test_parse_handles_only_pipe_no_keywords() -> None:
    text = "src/a.py | "
    result = _parse_blast_radius_output(text)
    assert len(result) == 1
    path, keywords = result[0]
    assert path == "src/a.py"
    assert keywords == []


def test_parse_extra_pipes_in_keywords_ignored() -> None:
    # only first | is the separator; rest stays in keywords part
    text = "src/a.py | alpha | beta"
    result = _parse_blast_radius_output(text)
    assert len(result) == 1
    path, keywords = result[0]
    assert path == "src/a.py"
    # "alpha | beta" becomes one keyword "alpha | beta" after split on ","
    # "alpha " and " beta" are individual comma-split parts — but there's no comma here
    # so keywords = ["alpha | beta"] lowercased → ["alpha | beta"]
    assert len(keywords) == 1


def test_parse_preserves_keyword_order_with_dedup() -> None:
    text = "src/a.py | zebra, alpha, beta, alpha, zebra"
    _, keywords = _parse_blast_radius_output(text)[0]
    assert keywords == ["zebra", "alpha", "beta"]


# ---------------------------------------------------------------------------
# count_blast_areas
# ---------------------------------------------------------------------------

def test_count_single_dir() -> None:
    paths = ["agent/tools/shell.py", "agent/tools/list.py"]
    assert count_blast_areas(paths) == 1


def test_count_nested_collapses_to_parent() -> None:
    # agent/tools is ancestor of agent/tools/helpers — collapses to 1
    paths = ["agent/tools/shell.py", "agent/tools/helpers/fs.py"]
    assert count_blast_areas(paths) == 1


def test_count_sibling_dirs() -> None:
    paths = ["agent/tools/shell.py", "agent/helpers/sanitizer.py"]
    assert count_blast_areas(paths) == 2


def test_count_four_files_two_dirs() -> None:
    paths = [
        "agent/tools/read.py",
        "agent/tools/list.py",
        "agent/helpers/sanitizer.py",
        "agent/helpers/compress.py",
    ]
    assert count_blast_areas(paths) == 2


def test_count_root_file_pulls_in_subdirs() -> None:
    # agent is ancestor of agent/tools — collapses to 1
    paths = ["agent/foo.py", "agent/tools/bar.py"]
    assert count_blast_areas(paths) == 1


def test_count_empty() -> None:
    assert count_blast_areas([]) == 0


def test_count_single_file() -> None:
    assert count_blast_areas(["agent/tools/shell.py"]) == 1


# ---------------------------------------------------------------------------
# evaluate_blast_radius_gate
# ---------------------------------------------------------------------------

def _entries(paths: list[str]) -> list[tuple[str, list[str]]]:
    return [(p, []) for p in paths]


def test_gate_passes_at_limit() -> None:
    paths = [f"pkg{i}/file.py" for i in range(5)]
    rejected, reason = evaluate_blast_radius_gate(_entries(paths), limit=5)
    assert not rejected
    assert reason is None


def test_gate_rejects_over_limit() -> None:
    paths = [f"pkg{i}/file.py" for i in range(6)]
    rejected, reason = evaluate_blast_radius_gate(_entries(paths), limit=5)
    assert rejected
    assert reason is not None
    assert "6" in reason
    assert "5" in reason


def test_gate_passes_empty_entries() -> None:
    rejected, reason = evaluate_blast_radius_gate([], limit=5)
    assert not rejected


def test_gate_reason_lists_areas() -> None:
    paths = ["agent/tools/a.py", "agent/helpers/b.py", "frontend/c.tsx"]
    rejected, reason = evaluate_blast_radius_gate(_entries(paths), limit=2)
    assert rejected
    assert reason is not None
    # all three survivor dirs named in reason
    assert "agent/tools" in reason or "agent\\tools" in reason
    assert "agent/helpers" in reason or "agent\\helpers" in reason
    assert "frontend" in reason
