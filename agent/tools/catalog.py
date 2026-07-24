from __future__ import annotations

# Tool-name groups, mirroring the @tool functions registered by make_tools().
# Single source of truth for subagent allowlists and prompt generation —
# any rename/addition in files.py / shell.py must be mirrored here.
READ_TOOLS = ("read_file", "list_files", "grep", "file_info", "symbols")
EDIT_TOOLS = ("edit_file", "write_file")
FS_TOOLS = ("move_file", "copy_file", "delete_file", "make_dir")
SHELL_TOOLS = ("run_command",)

ALL_TOOLS = READ_TOOLS + EDIT_TOOLS + FS_TOOLS + SHELL_TOOLS

# Tighten-only ladder for assignment-time tool scoping (harness/tool_scope.py):
# each rung is a superset of the previous one. Index 0="read", 1="edit", 2="fs"/full.
FLOOR: tuple[str, ...] = READ_TOOLS + SHELL_TOOLS
RUNGS: tuple[tuple[str, ...], ...] = (FLOOR, FLOOR + EDIT_TOOLS, ALL_TOOLS)
