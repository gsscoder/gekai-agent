from __future__ import annotations

import pytest

from agent.persona import ROOT_SYSTEM_PROMPT, _IDENTITY_ROOT, _ROOT_DIRECTIVES, _SHARED_BODY, render_tool_instruction
from agent.tools.catalog import ALL_TOOLS, EDIT_TOOLS, FS_TOOLS, READ_TOOLS, SHELL_TOOLS


def test_system_prompt_is_identity_plus_shared_body_plus_root_directives():
    # locks root's prompt composition — subagents reuse _SHARED_BODY
    # but get _IDENTITY_SUB + their own <directives> instead of _IDENTITY_ROOT
    # + _ROOT_DIRECTIVES
    assert ROOT_SYSTEM_PROMPT == _IDENTITY_ROOT + _SHARED_BODY + "\n<directives>\n" + _ROOT_DIRECTIVES

# The pre-refactor TOOL_INSTRUCTION constant, kept here only as a behavior
# baseline — render_tool_instruction(ALL_TOOLS) must reproduce it verbatim.
_LEGACY_TOOL_INSTRUCTION = (
    "if the question requires file contents, implementation details, logic, or architecture depth, "
    "you MUST use tools to read actual files — do not guess or rely on training knowledge; "
    "when multiple targets are nearby, prefer one wider ranged read_file call over many individual reads; "
    "before reading multiple files, use list_files to check their sizes first; "
    "use run_command for build, test, and git operations; "
    "run_command is stateless — cd does not persist across calls, each call starts in workspace root; "
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


@pytest.mark.parametrize(
    "shell_kind,expected,other",
    [
        ("powershell", "the shell is PowerShell", "the shell is bash"),
        ("bash", "the shell is bash", "the shell is PowerShell"),
    ],
)
def test_shell_kind_adds_matching_fact_only(shell_kind, expected, other):
    text = render_tool_instruction(SHELL_TOOLS, shell_kind=shell_kind)
    assert expected in text
    assert other not in text


def test_shell_kind_fact_does_not_leak_without_shell_trigger():
    text = render_tool_instruction(READ_TOOLS, shell_kind="powershell")
    assert "the shell is PowerShell" not in text


def test_shell_kind_omitted_adds_no_fact():
    text = render_tool_instruction(SHELL_TOOLS)
    assert "the shell is PowerShell" not in text
    assert "the shell is bash" not in text
    assert "the shell is sh" not in text


_DELEGATE_FRAGMENT = (
    "judge the investigation radius before you search: when the target is already named and tight — a specific file, symbol, or directory — use your own read tools directly; "
    "delegate to a read-only lookup specialist when the radius is wide or unknown, when the answer could live anywhere, or when the question crosses layers you have not mapped yet; "
    "delegate work-shaped tasks only when they fall outside your own mandate and a listed specialist covers them, never as a substitute for doing your own task; "
    "you stay accountable for the final report, so fold the delegated result into your own output instead of treating it as fire-and-forget"
)


def test_delegate_absent_leaves_instruction_unchanged():
    assert render_tool_instruction(READ_TOOLS + SHELL_TOOLS) == (
        _LEGACY_TOOL_INSTRUCTION
    )


def test_delegate_present_appends_only_its_own_fragment():
    baseline = render_tool_instruction(READ_TOOLS + SHELL_TOOLS)
    text = render_tool_instruction(READ_TOOLS + SHELL_TOOLS + ("delegate",))
    assert text == baseline + "; " + _DELEGATE_FRAGMENT
