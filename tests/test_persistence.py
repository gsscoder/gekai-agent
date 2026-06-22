from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from agent.diff import DiffLine
from agent.persistence import (
    append_command,
    append_diff,
    append_event,
    append_message,
    append_operation,
    append_subagent_done,
    append_subagent_start,
    load_session,
    load_timeline,
    session_file,
)
from agent.session import Session


def _make_session(tmp_path: Path) -> Session:
    s = Session(id=str(uuid.uuid4()), working_dir=tmp_path)
    return s


# ---------------------------------------------------------------------------
# append_message injects kind="turn"
# ---------------------------------------------------------------------------

def test_append_message_has_kind_turn(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_message(s, {"role": "user", "content": "hello"})
    lines = session_file(s).read_text().splitlines()
    entry = json.loads(lines[0])
    assert entry["kind"] == "turn"
    assert entry["role"] == "user"
    assert entry["content"] == "hello"


# ---------------------------------------------------------------------------
# append_command / append_event round-trip
# ---------------------------------------------------------------------------

def test_append_command_round_trip(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_command(s, "/config:gate off")
    entry = json.loads(session_file(s).read_text().splitlines()[0])
    assert entry["kind"] == "command"
    assert entry["content"] == "/config:gate off"
    assert "timestamp" in entry


def test_append_event_round_trip(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_event(s, "gate blocked: 6 areas", source="gate")
    entry = json.loads(session_file(s).read_text().splitlines()[0])
    assert entry["kind"] == "event"
    assert entry["source"] == "gate"
    assert entry["content"] == "gate blocked: 6 areas"


# ---------------------------------------------------------------------------
# append_diff / append_subagent_start / append_subagent_done / append_operation
# round-trip (Phase A/B/C resume-fidelity timeline writers)
# ---------------------------------------------------------------------------

def test_append_diff_round_trip(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_diff(
        s,
        "src/foo.py",
        [DiffLine(kind="add", text="+ x"), DiffLine(kind="del", text="- y")],
    )
    entry = json.loads(session_file(s).read_text().splitlines()[0])
    assert entry["kind"] == "diff"
    assert entry["path"] == "src/foo.py"
    assert entry["lines"] == [{"k": "add", "t": "+ x"}, {"k": "del", "t": "- y"}]
    # no turn kwarg passed -> key absent (matches append_message's `if turn:` pattern)
    assert "turn" not in entry


def test_append_diff_with_turn_id(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_diff(s, "src/foo.py", [DiffLine(kind="context", text=" unchanged")], turn="abc123")
    entry = json.loads(session_file(s).read_text().splitlines()[0])
    assert entry["turn"] == "abc123"


def test_append_subagent_start_round_trip(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_subagent_start(
        s,
        namespace="code",
        name="code-fixer",
        bg_color="#3a6ea5",
        ui_label="fix the bug",
    )
    entry = json.loads(session_file(s).read_text().splitlines()[0])
    assert entry["kind"] == "subagent_start"
    assert entry["namespace"] == "code"
    assert entry["name"] == "code-fixer"
    assert entry["bg_color"] == "#3a6ea5"
    assert entry["ui_label"] == "fix the bug"
    assert "turn" not in entry


def test_append_subagent_start_with_turn_id(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_subagent_start(
        s,
        namespace="code",
        name="code-fixer",
        bg_color="#3a6ea5",
        ui_label="fix the bug",
        turn="abc123",
    )
    entry = json.loads(session_file(s).read_text().splitlines()[0])
    assert entry["turn"] == "abc123"


def test_append_subagent_done_round_trip(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_subagent_done(s, "3 tools · 1.2k tokens · 4.1s", bg_color="#3a6ea5")
    entry = json.loads(session_file(s).read_text().splitlines()[0])
    assert entry["kind"] == "subagent_done"
    assert entry["summary"] == "3 tools · 1.2k tokens · 4.1s"
    assert entry["bg_color"] == "#3a6ea5"
    assert "turn" not in entry


def test_append_subagent_done_with_turn_id(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_subagent_done(s, "done", bg_color="#3a6ea5", turn="abc123")
    entry = json.loads(session_file(s).read_text().splitlines()[0])
    assert entry["turn"] == "abc123"


def test_append_operation_round_trip(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_operation(s, "* Crafted for 2.3s (3 tools)", "#00ff88")
    entry = json.loads(session_file(s).read_text().splitlines()[0])
    assert entry["kind"] == "operation"
    assert entry["content"] == "* Crafted for 2.3s (3 tools)"
    assert entry["color"] == "#00ff88"
    assert "turn" not in entry


def test_append_operation_with_turn_id(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_operation(s, "* Crafted for 2.3s (3 tools)", "#00ff88", turn="abc123")
    entry = json.loads(session_file(s).read_text().splitlines()[0])
    assert entry["turn"] == "abc123"


# ---------------------------------------------------------------------------
# load_session — only kind=="turn" returned
# ---------------------------------------------------------------------------

def test_load_session_excludes_command(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_message(s, {"role": "user", "content": "hi"})
    append_command(s, "/exit")
    append_message(s, {"role": "assistant", "content": "bye"})

    result = load_session(s.id)
    assert result is not None
    _, _, messages = result
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant"]


def test_load_session_excludes_event(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_message(s, {"role": "user", "content": "hi"})
    append_event(s, "iteration limit", source="max_iterations")

    result = load_session(s.id)
    assert result is not None
    _, _, messages = result
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_load_session_missing_kind_defaults_to_turn(tmp_path: Path) -> None:
    """Old-format entries without kind field are treated as turns."""
    s = _make_session(tmp_path)
    path = session_file(s)
    # write a legacy entry without kind
    path.write_text(
        json.dumps({"timestamp": "2025-01-01T00:00:00.000Z", "role": "user", "content": "legacy"}) + "\n"
    )
    result = load_session(s.id)
    assert result is not None
    _, _, messages = result
    assert len(messages) == 1
    assert messages[0]["content"] == "legacy"


# ---------------------------------------------------------------------------
# load_timeline — all entries ordered, system turns excluded
# ---------------------------------------------------------------------------

def test_load_timeline_includes_all_kinds(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_message(s, {"role": "user", "content": "do x"})
    append_command(s, "/config:gate off")
    append_event(s, "Gate disabled", source="command")
    append_message(s, {"role": "assistant", "content": "done"})
    append_event(s, "iteration limit", source="max_iterations")

    result = load_timeline(s.id)
    assert result is not None
    _, entries = result
    kinds = [e["kind"] for e in entries]
    assert kinds == ["turn", "command", "event", "turn", "event"]


def test_load_timeline_includes_new_kinds(tmp_path: Path) -> None:
    """Extends test_load_timeline_includes_all_kinds with the Phase A/B/C kinds."""
    s = _make_session(tmp_path)
    append_message(s, {"role": "user", "content": "fix the diff"})
    append_diff(s, "src/foo.py", [DiffLine(kind="add", text="+ x")])
    append_subagent_start(
        s, namespace="code", name="code-fixer", bg_color="#3a6ea5", ui_label="fix the bug"
    )
    append_command(s, "/config:gate off")
    append_subagent_done(s, "3 tools · 1.2k tokens · 4.1s", bg_color="#3a6ea5")
    append_operation(s, "* Crafted for 2.3s (3 tools)", "#00ff88")
    append_message(s, {"role": "assistant", "content": "done"})

    result = load_timeline(s.id)
    assert result is not None
    _, entries = result
    kinds = [e["kind"] for e in entries]
    assert kinds == [
        "turn",
        "diff",
        "subagent_start",
        "command",
        "subagent_done",
        "operation",
        "turn",
    ]


def test_load_timeline_excludes_non_persistent_system(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    # write a system message (non-persistent, like SYSTEM_PROMPT)
    path = session_file(s)
    path.write_text(
        json.dumps({"timestamp": "2025-01-01T00:00:00.000Z", "kind": "turn", "role": "system", "content": "you are gekai"}) + "\n" +
        json.dumps({"timestamp": "2025-01-01T00:00:00.001Z", "kind": "turn", "role": "user", "content": "hi"}) + "\n"
    )
    result = load_timeline(s.id)
    assert result is not None
    _, entries = result
    # system prompt (non-persistent) should be excluded
    assert len(entries) == 1
    assert entries[0]["role"] == "user"


def test_load_timeline_ordering(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_message(s, {"role": "user", "content": "first"})
    append_command(s, "/clear")
    append_message(s, {"role": "assistant", "content": "second"})

    result = load_timeline(s.id)
    assert result is not None
    _, entries = result
    contents = [e.get("content") for e in entries]
    assert contents == ["first", "/clear", "second"]


def test_load_timeline_returns_none_for_missing(tmp_path: Path) -> None:
    assert load_timeline("nonexistent-id") is None


# ---------------------------------------------------------------------------
# max_iterations: no empty assistant turn in session
# ---------------------------------------------------------------------------

def test_no_empty_assistant_turn_on_max_iter(tmp_path: Path) -> None:
    """Simulates what process_stream does: event instead of empty turn."""
    s = _make_session(tmp_path)
    append_message(s, {"role": "user", "content": "do something"})
    # process_stream path: max_iter_hit=True, no chunks → append_event, no append_message
    append_event(s, "agent hit iteration limit without producing a response", source="max_iterations")

    result = load_session(s.id)
    assert result is not None
    _, _, messages = result
    # only the user turn; no empty assistant
    assert len(messages) == 1
    assert messages[0]["role"] == "user"

    timeline_result = load_timeline(s.id)
    assert timeline_result is not None
    _, entries = timeline_result
    assert len(entries) == 2
    assert entries[1]["kind"] == "event"
    assert entries[1]["source"] == "max_iterations"
