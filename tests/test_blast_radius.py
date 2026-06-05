from __future__ import annotations

from agent.blast_radius import _parse_blast_radius_output


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
