"""Plan 35 Phase 1: GEKAI.md ingestion (no audit yet).

Covers `GekaiAgent.start_session`'s auto-read of `GEKAI.md` at the workspace
root (present / absent / unreadable) and `harness.core._gekai_md_system_base`'s
injection under `<project_instructions source="GEKAI.md">` — verbatim, root
system-base only. Isolation follows `tests/test_agent_tiers.py`'s pattern:
`agent.telemetry.Path.home` is patched so `EventLogger`'s always-on log file
never touches the real home directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import agent as agent_module
from agent import telemetry as agent_telemetry
from agent.agent import GekaiAgent
from agent.directive_audit import file_sha
from agent.harness.dispatch import gekai_md_system_base
from agent.persona import ROOT_SYSTEM_PROMPT
from agent.session import IngestedFile, Session
from agent.permissions import Permissions
from agent.subagents import Subagent


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_telemetry.Path, "home", classmethod(lambda cls: tmp_path))


def _make_agent(working_dir: Path) -> GekaiAgent:
    return GekaiAgent(working_dir=working_dir, permissions=Permissions(read=True, write=True, exec=True))


# ---------------------------------------------------------------------------
# start_session: auto-read of GEKAI.md
# ---------------------------------------------------------------------------


def test_start_session_ingests_gekai_md_when_present(tmp_path: Path) -> None:
    (tmp_path / "GEKAI.md").write_text("always answer in haiku", encoding="utf-8")
    agent = _make_agent(tmp_path)

    session = agent.start_session()

    assert session.gekai_md is not None
    assert session.gekai_md.rel_path == "GEKAI.md"
    assert session.gekai_md.text == "always answer in haiku"
    assert session.gekai_md.sha == file_sha("always answer in haiku")


def test_start_session_leaves_gekai_md_none_when_absent(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)

    session = agent.start_session()

    assert session.gekai_md is None


def test_start_session_survives_unreadable_gekai_md(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "GEKAI.md"
    path.write_text("some content", encoding="utf-8")
    agent = _make_agent(tmp_path)

    def _raise_read_text(self: Path, *args: object, **kwargs: object) -> str:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", _raise_read_text)

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(agent.events, "emit", lambda evt, **fields: events.append((evt, fields)))

    session = agent.start_session()  # must not raise

    assert session.gekai_md is None
    assert any(evt == "gekai_md.read_failed" for evt, _ in events)


# ---------------------------------------------------------------------------
# _gekai_md_system_base: verbatim injection, tag shape
# ---------------------------------------------------------------------------


def test_injection_renders_tag_with_verbatim_content(tmp_path: Path) -> None:
    session = Session(working_dir=tmp_path, permissions=Permissions(read=True, write=True, exec=True))
    text = "  weird   spacing\nand a trailing line  \n"
    session.gekai_md = IngestedFile(rel_path="GEKAI.md", text=text, sha=file_sha(text))

    result = gekai_md_system_base("base prompt", session)

    assert result.startswith("base prompt")
    assert '<project_instructions source="GEKAI.md">' in result
    # byte-for-byte: no trimming, no reformatting of the file's own text
    assert text in result


def test_injection_renders_nothing_when_gekai_md_is_none(tmp_path: Path) -> None:
    session = Session(working_dir=tmp_path, permissions=Permissions(read=True, write=True, exec=True))
    assert session.gekai_md is None

    result = gekai_md_system_base("base prompt", session)

    assert result == "base prompt"
    assert "project_instructions" not in result


# ---------------------------------------------------------------------------
# subagent isolation: a cold specialist's system base never carries the tag,
# even when root's does for the very same session (plan 35 decision 10 /
# plan 28 decision 13). This mirrors `Harness.stream()`'s own branch
# selection: root calls `_gekai_md_system_base`, a subagent instead calls
# `subagent.build_system_base()` — the two are structurally disjoint calls,
# not a flag on a shared one.
# ---------------------------------------------------------------------------


def test_subagent_system_base_never_carries_project_instructions(tmp_path: Path) -> None:
    session = Session(working_dir=tmp_path, permissions=Permissions(read=True, write=True, exec=True))
    session.gekai_md = IngestedFile(rel_path="GEKAI.md", text="secret project rule", sha="deadbeef")

    root_base = gekai_md_system_base(ROOT_SYSTEM_PROMPT, session)
    assert '<project_instructions source="GEKAI.md">' in root_base
    assert "secret project rule" in root_base

    subagent = Subagent(name="t", namespace="coding", description="d")
    subagent_base = subagent.build_system_base()
    assert '<project_instructions source="GEKAI.md">' not in subagent_base
    assert "secret project rule" not in subagent_base
