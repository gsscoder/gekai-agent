from __future__ import annotations

from pathlib import Path

from .files import make_file_tools, _move_file, _copy_file, _delete_file, _make_dir
from .shell import make_shell_tools, _run_command


def make_tools(working_dir: Path) -> list:
    return make_file_tools(working_dir) + make_shell_tools(working_dir)
