from __future__ import annotations

import uuid
from pathlib import Path

from agent.persistence import append_compact, append_message, load_session
from agent.session import Session


def _make_session(tmp_path: Path) -> Session:
    s = Session(id=str(uuid.uuid4()), working_dir=tmp_path)
    return s


# ---------------------------------------------------------------------------
# load_session — compact boundary drops pre-compact turns
# ---------------------------------------------------------------------------

def test_load_session_drops_turns_before_compact(tmp_path: Path) -> None:
    s = _make_session(tmp_path)
    append_message(s, {"role": "user", "content": "first"})
    append_message(s, {"role": "assistant", "content": "second"})
    append_message(s, {"role": "user", "content": "third"})
    append_compact(s, "the summary")
    append_message(s, {"role": "user", "content": "fourth"})
    append_message(s, {"role": "assistant", "content": "fifth"})

    result = load_session(s.id)
    assert result is not None
    _, _, messages = result
    assert [{"role": m["role"], "content": m["content"]} for m in messages] == [
        {"role": "user", "content": "the summary"},
        {"role": "user", "content": "fourth"},
        {"role": "assistant", "content": "fifth"},
    ]


def test_load_session_no_compact_returns_full_history(tmp_path: Path) -> None:
    """Regression check: no compact entry -> unchanged behavior."""
    s = _make_session(tmp_path)
    append_message(s, {"role": "user", "content": "first"})
    append_message(s, {"role": "assistant", "content": "second"})
    append_message(s, {"role": "user", "content": "third"})

    result = load_session(s.id)
    assert result is not None
    _, _, messages = result
    assert [{"role": m["role"], "content": m["content"]} for m in messages] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
