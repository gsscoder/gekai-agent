"""Shared per-path locking and atomic file writes: every on-disk settings/
cache file in this codebase (`agent/settings.py`'s `settings.local.json` and
`settings.json`, `agent/directive_audit.py`'s `directive-audit.json`) is
read-modified-written by potentially concurrent callers — overlapping
permission-grant callbacks, background audit tasks — racing to update the
same file. A lock keyed by the resolved path serializes that critical
section per file without serializing unrelated files against each other, and
the write itself goes through a tempfile-then-`os.replace` so a crash or
concurrent reader never observes a partially-written file.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

_locks: dict[Path, threading.Lock] = {}
_locks_guard = threading.Lock()


def lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _locks[path] = lock
        return lock


def atomic_write(path: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


__all__ = ["lock_for", "atomic_write"]
