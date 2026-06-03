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


def _meta_path(workspace_folder: Path) -> Path:
    return workspace_folder / "meta.json"


def _write_meta(workspace_folder: Path, working_dir: Path) -> None:
    meta = _meta_path(workspace_folder)
    if not meta.exists():
        meta.write_text(json.dumps({"working_dir": str(working_dir)}), encoding="utf-8")


def session_file(session: Session) -> Path:
    base = Path.home() / ".gekai" / "workspaces" / _normalize_path(session.working_dir)
    base.mkdir(parents=True, exist_ok=True)
    _write_meta(base, session.working_dir)
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


_PERSISTENT_SYSTEM_PREFIXES = ("[artifact]", "<lang>")


def _is_persistent_system_message(m: dict) -> bool:
    if m.get("role") != "system":
        return False
    content = m.get("content") or ""
    return content.startswith(_PERSISTENT_SYSTEM_PREFIXES)


def load_session(session_id: str) -> tuple[str, Path, list[dict]] | None:
    """
    Returns (session_id, working_dir, user/assistant + persistent system messages).
    Always-fresh system messages (SYSTEM_PROMPT, workspace) are re-injected on startup.
    Returns None if not found or meta is missing/malformed.
    """
    matches = list((Path.home() / ".gekai" / "workspaces").glob(f"*/{session_id}.jsonl"))
    if not matches:
        return None
    path = matches[0]
    workspace_folder = path.parent
    try:
        meta_raw = _meta_path(workspace_folder).read_text(encoding="utf-8")
        working_dir = Path(json.loads(meta_raw)["working_dir"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None
    messages: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("role") in ("user", "assistant") or _is_persistent_system_message(m):
                messages.append(m)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return session_id, working_dir, messages


