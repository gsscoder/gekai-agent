from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agent.harness.file_locator import FileLocator, _format_hint_section, _parse_locator_output
from agent.pipeline._directives import PIPELINE_DIRECTIVES
from agent.llm.types import Message


def _make_locator() -> FileLocator:
    with patch("agent.harness.file_locator.OpenAIAdapter"):
        return FileLocator(model="test-model", api_key="key", api_base="http://localhost")


def test_locator_system_prompt_contains_pipeline_directives() -> None:
    locator = _make_locator()
    with patch("agent.harness.file_locator.Agent") as MockAgent:
        instance = MagicMock()
        instance.tools = MagicMock()
        instance.run = AsyncMock(return_value=[Message(role="assistant", content="")])
        MockAgent.return_value = instance

        import asyncio
        asyncio.run(locator.locate(Path("."), "add a new feature"))

        _, kwargs = MockAgent.call_args
        assert kwargs["system"].startswith(PIPELINE_DIRECTIVES)


def test_locate_without_hints_omits_hint_section() -> None:
    locator = _make_locator()
    with patch("agent.harness.file_locator.Agent") as MockAgent:
        instance = MagicMock()
        instance.tools = MagicMock()
        instance.run = AsyncMock(return_value=[Message(role="assistant", content="")])
        MockAgent.return_value = instance

        import asyncio
        asyncio.run(locator.locate(Path("."), "add a new feature"))

        _, kwargs = MockAgent.call_args
        assert "<hints>" not in kwargs["system"]


def test_locate_with_hints_injects_hint_section() -> None:
    locator = _make_locator()
    with patch("agent.harness.file_locator.Agent") as MockAgent:
        instance = MagicMock()
        instance.tools = MagicMock()
        instance.run = AsyncMock(return_value=[Message(role="assistant", content="")])
        MockAgent.return_value = instance

        import asyncio
        asyncio.run(locator.locate(Path("."), "add a new feature", hint_paths=["src/main.py", "src/util.py"]))

        _, kwargs = MockAgent.call_args
        system = kwargs["system"]
        assert "<hints>" in system
        assert "src/main.py" in system
        assert "src/util.py" in system


def test_format_hint_section_empty_for_none() -> None:
    assert _format_hint_section(None) == ""


def test_format_hint_section_empty_for_empty_list() -> None:
    assert _format_hint_section([]) == ""


def test_format_hint_section_includes_paths() -> None:
    section = _format_hint_section(["a.py", "b.py"])
    assert section.startswith("<hints>")
    assert "a.py" in section
    assert "b.py" in section


def test_parse_well_formed_line() -> None:
    text = "src/main.py | calculator, arithmetic, compute"
    result = _parse_locator_output(text)
    assert len(result) == 1
    path, keywords = result[0]
    assert path == "src/main.py"
    assert keywords == ["calculator", "arithmetic", "compute"]


def test_parse_multiple_lines() -> None:
    text = (
        "src/main.py | calculator, arithmetic\n"
        "frontend/PasscodeBox.tsx | passcode, pin, otp, dialog"
    )
    result = _parse_locator_output(text)
    assert len(result) == 2
    assert result[0][0] == "src/main.py"
    assert result[1][0] == "frontend/PasscodeBox.tsx"
    assert "pin" in result[1][1]


def test_parse_skips_blank_lines() -> None:
    text = "\nsrc/a.py | alpha\n\nsrc/b.py | beta\n"
    result = _parse_locator_output(text)
    assert len(result) == 2


def test_parse_skips_lines_without_pipe() -> None:
    text = (
        "some prose without a pipe\n"
        "src/main.py | valid, keywords\n"
        "another bad line"
    )
    result = _parse_locator_output(text)
    assert len(result) == 1
    assert result[0][0] == "src/main.py"


def test_parse_empty_text() -> None:
    assert _parse_locator_output("") == []


def test_parse_lowercases_keywords() -> None:
    text = "src/Auth.py | Login, TOKEN, Session"
    result = _parse_locator_output(text)
    _, keywords = result[0]
    assert keywords == ["login", "token", "session"]


def test_parse_deduplicates_keywords() -> None:
    text = "src/a.py | alpha, alpha, beta, alpha"
    _, keywords = _parse_locator_output(text)[0]
    assert keywords.count("alpha") == 1
    assert "beta" in keywords


def test_parse_strips_whitespace_from_keywords() -> None:
    text = "src/a.py |  login ,  token , session "
    _, keywords = _parse_locator_output(text)[0]
    assert keywords == ["login", "token", "session"]


def test_parse_skips_line_with_empty_path() -> None:
    text = " | keyword1, keyword2"
    result = _parse_locator_output(text)
    assert result == []


def test_parse_handles_only_pipe_no_keywords() -> None:
    text = "src/a.py | "
    result = _parse_locator_output(text)
    assert len(result) == 1
    path, keywords = result[0]
    assert path == "src/a.py"
    assert keywords == []


def test_parse_extra_pipes_in_keywords_ignored() -> None:
    text = "src/a.py | alpha | beta"
    result = _parse_locator_output(text)
    assert len(result) == 1
    path, keywords = result[0]
    assert path == "src/a.py"
    assert len(keywords) == 1


def test_parse_preserves_keyword_order_with_dedup() -> None:
    text = "src/a.py | zebra, alpha, beta, alpha, zebra"
    _, keywords = _parse_locator_output(text)[0]
    assert keywords == ["zebra", "alpha", "beta"]