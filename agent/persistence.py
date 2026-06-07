from __future__ import annotations
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from .session import Session


def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _normalize_path(p: Path) -> str:
    r"""C:\MyCompany\My-Projects\Super_notepad -> cmycompanymyprojectssupernotepad"""
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


def _append(session: Session, entry: dict) -> None:
    path = session_file(session)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"timestamp": now_utc_str(), **entry}, separators=(",", ":")) + "\n")


def append_message(session: Session, message: dict) -> None:
    _append(session, {"kind": "turn", **message})


def append_command(session: Session, text: str) -> None:
    _append(session, {"kind": "command", "content": text})


def append_event(session: Session, content: str, source: str) -> None:
    _append(session, {"kind": "event", "source": source, "content": content})


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


def _session_path(session_id: str) -> tuple[Path, Path] | None:
    """Return (jsonl_path, workspace_folder) or None if not found."""
    matches = list((Path.home() / ".gekai" / "workspaces").glob(f"*/{session_id}.jsonl"))
    if not matches:
        return None
    path = matches[0]
    return path, path.parent


def _read_working_dir(workspace_folder: Path) -> Path | None:
    try:
        meta_raw = _meta_path(workspace_folder).read_text(encoding="utf-8")
        return Path(json.loads(meta_raw)["working_dir"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def load_session(session_id: str) -> tuple[str, Path, list[dict]] | None:
    """Return (session_id, working_dir, model-turn messages only).

    Only kind=="turn" entries (or entries with no kind field, for backward compat)
    are returned. Always-fresh system messages are re-injected on startup.
    """
    result = _session_path(session_id)
    if result is None:
        return None
    path, workspace_folder = result
    working_dir = _read_working_dir(workspace_folder)
    if working_dir is None:
        return None
    messages: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            kind = m.get("kind", "turn")
            if kind != "turn":
                continue
            if m.get("role") in ("user", "assistant") or _is_persistent_system_message(m):
                messages.append(m)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return session_id, working_dir, messages


def load_timeline(session_id: str) -> tuple[Path, list[dict]] | None:
    """Return (working_dir, all entries ordered) for chat rebuild.

    Includes turns, commands, and events — everything the user saw on screen.
    Always-fresh system turns (SYSTEM_PROMPT, workspace) are excluded.
    """
    result = _session_path(session_id)
    if result is None:
        return None
    path, workspace_folder = result
    working_dir = _read_working_dir(workspace_folder)
    if working_dir is None:
        return None
    entries: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            kind = m.get("kind", "turn")
            if kind == "turn":
                role = m.get("role")
                # skip always-fresh system messages
                if role == "system" and not _is_persistent_system_message(m):
                    continue
            entries.append(m)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return working_dir, entries
