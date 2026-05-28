from __future__ import annotations
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from .router import Session


def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _normalize_path(p: Path) -> str:
    """C:\MyCompany\My-Projects\Super_notepad -> cmycompanymyprojectssupernotepad"""
    return re.sub(r"[^a-z0-9]", "", str(p).lower())


def session_file(session: Session) -> Path:
    base = Path.home() / ".gekai" / "workspaces" / _normalize_path(session.working_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{session.id}.jsonl"


def append_message(session: Session, message: dict) -> None:
    path = session_file(session)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"timestamp": now_utc_str(), **message}, separators=(",", ":")) + "\n")


def append_debug(session: Session, message: dict) -> None:
    base = Path.home() / ".gekai" / "workspaces" / _normalize_path(session.working_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{session.id}.debug.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"timestamp": now_utc_str(), "content": message["content"]}, separators=(",", ":")) + "\n")


def load_session(session_id: str, working_dir: Path) -> tuple[str, list[dict]] | None:
    """
    Returns (session_id, user/assistant messages). System messages are re-injected
    fresh on startup. Returns None if not found.
    """
    path = Path.home() / ".gekai" / "workspaces" / _normalize_path(working_dir) / f"{session_id}.jsonl"
    if not path.exists():
        return None
    messages: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("role") in ("user", "assistant"):
                messages.append(m)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return session_id, messages
