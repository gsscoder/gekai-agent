from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .persistence import now_utc_str


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": now_utc_str(),
            "run": getattr(record, "run_id", ""),
            "evt": record.msg,
            "level": record.levelname.lower(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, separators=(",", ":"), default=str)


class EventLogger:
    """Always-on JSONL event log at ~/.gekai/logs/events-{date}-{time}-{run}.jsonl.

    One file per process execution. emit() is fire-and-forget and never raises.
    """

    def __init__(self) -> None:
        self.run_id = uuid.uuid4().hex[:8]
        self.turn_count = 0
        self._start = time.monotonic()
        self._logger = logging.getLogger(f"gekai.events.{self.run_id}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        try:
            log_dir = Path.home() / ".gekai" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            handler: logging.Handler = logging.FileHandler(
                log_dir / f"events-{ts}-{self.run_id}.jsonl", encoding="utf-8"
            )
            handler.setFormatter(_JsonFormatter())
        except OSError:
            handler = logging.NullHandler()
        self._logger.addHandler(handler)

    def emit(self, evt: str, *, level: str = "info", **fields: object) -> None:
        try:
            self._logger.log(
                getattr(logging, level.upper()), evt,
                extra={"run_id": self.run_id, "fields": fields},
            )
        except Exception:
            pass

    def new_turn(self) -> str:
        self.turn_count += 1
        return uuid.uuid4().hex[:8]

    def runtime_s(self) -> float:
        return round(time.monotonic() - self._start, 3)

    def close(self) -> None:
        for h in list(self._logger.handlers):
            try:
                h.close()
            except Exception:
                pass
            self._logger.removeHandler(h)
