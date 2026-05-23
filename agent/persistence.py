from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from .router import Session


def _serialize(data: dict) -> str:
    header = {k: v for k, v in data.items() if k != "messages"}
    base = json.dumps(header, indent=2).rstrip()[:-1]  # strip closing }
    messages_str = ",\n    ".join(
        json.dumps(m, separators=(",", ":")) for m in data.get("messages", [])
    )
    return f'{base},\n  "messages": [\n    {messages_str}\n  ]\n}}'


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_path(p: Path) -> str:
    """C:\MyCompany\My-Projects\Super_notepad -> cmycompanymyprojectssupernotepad"""
    return re.sub(r"[^a-z0-9]", "", str(p).lower())


def session_file(session: Session) -> Path:
    base = Path.home() / ".gekai" / "workspaces" / _normalize_path(session.working_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{session.id}.json"


def save_session(session: Session) -> None:
    path = session_file(session)
    now = _now()
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        created_at = existing.get("created_at", now)
    except (FileNotFoundError, json.JSONDecodeError):
        created_at = now
    data = {
        "session_id": session.id,
        "working_dir": str(session.working_dir),
        "created_at": created_at,
        "last_accessed_at": now,
        "messages": session.messages,
    }
    path.write_text(_serialize(data), encoding="utf-8")


def load_session(session_id: str, working_dir: Path) -> tuple[str, list[dict]] | None:
    """
    Returns (session_id, user/assistant messages). System messages are re-injected
    fresh on startup. Returns None if not found.
    """
    path = Path.home() / ".gekai" / "workspaces" / _normalize_path(working_dir) / f"{session_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    messages = [m for m in data.get("messages", []) if m.get("role") in ("user", "assistant")]
    return data["session_id"], messages
