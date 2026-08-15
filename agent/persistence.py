from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
from .session import Session


def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _normalize_path(p: Path) -> str:
    r"""C:\MyCompany\My-Projects\Super_notepad -> cmycompanymyprojectssupernotepad-a1b2c3d4

    A short hash of the resolved absolute path is appended so that distinct
    directories differing only by punctuation/case (which collapse to the same
    stripped slug) don't collide on the same storage bucket.
    """
    resolved = str(p.resolve())
    key = os.path.normcase(resolved)
    slug = re.sub(r"[^a-z0-9]", "", resolved.lower())
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


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


def append_message(session: Session, message: dict, *, turn: str | None = None) -> None:
    entry: dict = {"kind": "turn", **message}
    if turn:
        entry["turn"] = turn
    _append(session, entry)


def append_command(session: Session, text: str) -> None:
    _append(session, {"kind": "command", "content": text})


def append_event(session: Session, content: str, source: str) -> None:
    _append(session, {"kind": "event", "source": source, "content": content})


def append_diff(session: Session, path: str, diff_lines: list, *, turn: str | None = None) -> None:
    entry: dict = {
        "kind": "diff",
        "path": path,
        "lines": [{"k": dl.kind, "t": dl.text} for dl in diff_lines],
    }
    if turn:
        entry["turn"] = turn
    _append(session, entry)


def append_subagent_start(
    session: Session, *, namespace: str, name: str, bg_color: str, ui_label: str, turn: str | None = None,
) -> None:
    entry: dict = {
        "kind": "subagent_start",
        "namespace": namespace,
        "name": name,
        "bg_color": bg_color,
        "ui_label": ui_label,
    }
    if turn:
        entry["turn"] = turn
    _append(session, entry)


def append_subagent_done(session: Session, summary: str, *, bg_color: str, turn: str | None = None) -> None:
    entry: dict = {"kind": "subagent_done", "summary": summary, "bg_color": bg_color}
    if turn:
        entry["turn"] = turn
    _append(session, entry)


def append_operation(session: Session, content: str, color: str, *, turn: str | None = None) -> None:
    entry: dict = {"kind": "operation", "content": content, "color": color}
    if turn:
        entry["turn"] = turn
    _append(session, entry)


def append_compact(session: Session, summary: str) -> None:
    _append(session, {"kind": "compact", "content": summary})


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

    If a kind=="compact" entry exists, everything before the last one is dropped:
    the compact entry's content is surfaced as a synthetic leading user message,
    followed by any turn entries that came after the boundary.
    """
    result = _session_path(session_id)
    if result is None:
        return None
    path, workspace_folder = result
    working_dir = _read_working_dir(workspace_folder)
    if working_dir is None:
        return None
    try:
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    compact_index = None
    for i, m in enumerate(entries):
        if m.get("kind") == "compact":
            compact_index = i
    messages: list[dict] = []
    if compact_index is not None:
        messages.append({"role": "user", "content": entries[compact_index]["content"]})
        entries = entries[compact_index + 1:]
    for m in entries:
        kind = m.get("kind", "turn")
        if kind != "turn":
            continue
        if m.get("role") in ("user", "assistant") or _is_persistent_system_message(m):
            messages.append(m)
    return session_id, working_dir, messages


def load_timeline(session_id: str) -> tuple[Path, list[dict]] | None:
    """Return (working_dir, all entries ordered) for chat rebuild.

    Includes turns, commands, and events — everything the user saw on screen.
    Always-fresh system turns (ROOT_SYSTEM_PROMPT, workspace) are excluded.
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
