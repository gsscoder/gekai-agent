from __future__ import annotations

import sqlite3
from pathlib import Path

from agent import __version__

SCHEMA_VERSION = __version__

_DDL = """
CREATE TABLE IF NOT EXISTS _meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id   INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS file_keywords (
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    keyword TEXT NOT NULL,
    UNIQUE(file_id, keyword)
);
"""


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)
    conn.execute(
        "INSERT OR REPLACE INTO _meta(key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    conn.commit()


def ensure(working_dir: Path) -> sqlite3.Connection:
    gekai_dir = working_dir / ".gekai"
    gekai_dir.mkdir(exist_ok=True)
    db_path = gekai_dir / "workspace.db"

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")

    _apply_schema(conn)
    return conn


def save_blast_radius(
    conn: sqlite3.Connection,
    entries: list[tuple[str, list[str]]],
) -> None:
    for path, keywords in entries:
        conn.execute("INSERT OR IGNORE INTO files(path) VALUES (?)", (path,))
        row = conn.execute(
            "SELECT id FROM files WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            continue
        file_id = row[0]
        for kw in keywords:
            kw_norm = kw.strip().lower()
            if kw_norm:
                conn.execute(
                    "INSERT OR IGNORE INTO file_keywords(file_id, keyword) VALUES (?, ?)",
                    (file_id, kw_norm),
                )
    conn.commit()


def files_for_keywords(
    conn: sqlite3.Connection,
    keywords: list[str],
) -> list[str]:
    raise NotImplementedError
