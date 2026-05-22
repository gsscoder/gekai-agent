from __future__ import annotations
import json
import re
from pathlib import Path
from .router import Session


def _normalize_path(p: Path) -> str:
    """C:\MyCompany\My-Projects\Super_notepad -> cmycompanymyprojectssupernotepad"""
    return re.sub(r"[^a-z0-9]", "", str(p).lower())


def _short_id(session_id: str) -> str:
    """c2f27e2b-eb5c-48aa-9078-2f6e0df1b7b4 -> 2f6e0df1b7b4"""
    return session_id.split("-")[-1]


def session_file(session: Session) -> Path:
    base = Path.home() / ".gekai" / "sessions" / _normalize_path(session.working_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_short_id(session.id)}.json"


def save_session(session: Session) -> None:
    data = {
        "session_id": session.id,
        "working_dir": str(session.working_dir),
        "messages": session.messages,
    }
    session_file(session).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_session(session_id: str, working_dir: Path) -> tuple[str, list[dict]] | None:
    """
    Returns (session_id, user/assistant messages). System messages are re-injected
    fresh on startup. Returns None if not found.
    """
    short = session_id.split("-")[-1]
    path = Path.home() / ".gekai" / "sessions" / _normalize_path(working_dir) / f"{short}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    messages = [m for m in data.get("messages", []) if m.get("role") in ("user", "assistant")]
    return data["session_id"], messages
