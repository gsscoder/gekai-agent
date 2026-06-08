from __future__ import annotations

from agent.persona import render_tool_instruction
from agent.tools.catalog import ALL_TOOLS, EDIT_TOOLS, FS_TOOLS, READ_TOOLS, SHELL_TOOLS

# The pre-refactor TOOL_INSTRUCTION constant, kept here only as a behavior
# baseline — render_tool_instruction(ALL_TOOLS) must reproduce it verbatim.
_LEGACY_TOOL_INSTRUCTION = (
    "if the question requires file contents, implementation details, logic, or architecture depth, "
    "you MUST use tools to read actual files — do not guess or rely on training knowledge; "
    "when multiple targets are nearby, prefer one wider ranged read_file call over many individual reads; "
    "use run_command for build, test, and git operations; "
    "run_command is stateless — cd does not persist across calls, each call starts in repo root; "
    "prefer read_file/grep/list_files over shell equivalents for reading files"
)


def test_full_toolset_reproduces_legacy_instruction_verbatim():
    assert render_tool_instruction(ALL_TOOLS) == _LEGACY_TOOL_INSTRUCTION


def test_read_only_set_omits_shell_guidance():
    text = render_tool_instruction(READ_TOOLS)
    assert "run_command" not in text
    assert "you MUST use tools to read actual files" in text
    assert "wider ranged read_file call" in text


def test_read_only_set_omits_cross_tool_preference_line():
    # the "prefer read_file over shell" line only fires when both groups are present
    text = render_tool_instruction(READ_TOOLS)
    assert "prefer read_file/grep/list_files over shell equivalents" not in text


def test_shell_only_set_omits_read_guidance():
    text = render_tool_instruction(SHELL_TOOLS)
    assert "you MUST use tools to read actual files" not in text
    assert "wider ranged read_file call" not in text
    assert "use run_command for build, test, and git operations" in text


def test_read_and_shell_together_include_cross_tool_line():
    text = render_tool_instruction(READ_TOOLS + SHELL_TOOLS)
    assert "prefer read_file/grep/list_files over shell equivalents for reading files" in text


def test_edit_and_fs_only_set_yields_no_guidance():
    # neither read nor shell present — no fragment should fire
    assert render_tool_instruction(EDIT_TOOLS + FS_TOOLS) == ""


def test_empty_toolset_yields_empty_instruction():
    assert render_tool_instruction([]) == ""
