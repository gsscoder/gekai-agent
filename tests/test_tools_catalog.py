from __future__ import annotations

from pathlib import Path

from agent.tools import make_tools
from agent.tools.catalog import ALL_TOOLS, EDIT_TOOLS, FS_TOOLS, READ_TOOLS, SHELL_TOOLS


def test_groups_are_disjoint():
    groups = [READ_TOOLS, EDIT_TOOLS, FS_TOOLS, SHELL_TOOLS]
    seen: set[str] = set()
    for group in groups:
        for name in group:
            assert name not in seen, f"{name!r} appears in more than one tool group"
            seen.add(name)


def test_all_tools_matches_registered_tools(tmp_path: Path):
    # guards against catalog drift if a tool is renamed/added/removed in tools/files.py
    # or tools/shell.py without updating agent/tools/catalog.py
    registered = {t.name for t in make_tools(tmp_path)}
    assert set(ALL_TOOLS) == registered
