from __future__ import annotations

import json
import time
from pathlib import Path


class PromptHistory:
    def __init__(self, path: Path, max_entries: int = 100) -> None:
        self._path = path
        self._max_entries = max_entries
        self._entries: list[dict] = []
        self._index: int | None = None
        self._draft: str = ""

    def append(self, text: str) -> None:
        entries = self.load()
        # Dedup: remove existing entry with same text
        entries = [e for e in entries if e["text"] != text]
        entries.append({"timestamp": time.time(), "text": text})
        # Trim oldest from front if over limit
        if len(entries) > self._max_entries:
            entries = entries[-self._max_entries:]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def load(self) -> list[dict]:
        if not self._path.exists():
            return []
        entries: list[dict] = []
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            return []
        # Sort oldest to newest by timestamp
        entries.sort(key=lambda e: e.get("timestamp", 0))
        return entries

    def start_browse(self, current_input: str) -> None:
        if not self._entries:
            self._entries = self.load()
        self._draft = current_input
        self._index = len(self._entries)

    def browse_up(self) -> str | None:
        if not self._entries:
            self._entries = self.load()
        if not self._entries:
            return None
        if self._index is None:
            self._index = len(self._entries)
        self._index -= 1
        if self._index < 0:
            self._index = 0
        return self._entries[self._index]["text"]

    def browse_down(self) -> str | None:
        if not self._entries:
            self._entries = self.load()
        if self._index is None:
            return None
        self._index += 1
        if self._index >= len(self._entries):
            self._index = len(self._entries)
            return self._draft
        return self._entries[self._index]["text"]

    def reset(self) -> None:
        self._index = None
        self._draft = ""
        self._entries = []

    @property
    def is_browsing(self) -> bool:
        return self._index is not None
