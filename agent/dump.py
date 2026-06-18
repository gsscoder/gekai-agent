from __future__ import annotations

from .persistence import load_timeline

SEPARATOR = "─" * 60


def dump(scope: str, session_id: str) -> int:
    """Print session data for `scope` to console. Returns process exit code."""
    if scope == "prompts":
        return _dump_prompts(session_id)
    print(f"unknown dump scope {scope!r}; supported: prompts")
    return 1


def _dump_prompts(session_id: str) -> int:
    result = load_timeline(session_id)
    if result is None:
        print(f"session {session_id} not found")
        return 1
    _, entries = result
    prompts = [
        e.get("content", "")
        for e in entries
        if e.get("kind", "turn") == "turn" and e.get("role") == "user"
    ]
    if not prompts:
        # degenerate case: a hand-edited/corrupted session file with zero user turns.
        # Not a real state in normal operation (Gekai never surfaces a session id before
        # the first turn completes), but must not crash if someone pokes at the files.
        return 0
    print((f"\n{SEPARATOR}\n").join(prompts))
    return 0
