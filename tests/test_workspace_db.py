from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent import __version__
from agent.workspace_db import SCHEMA_VERSION, ensure, save_blast_radius, files_for_keywords


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def test_ensure_creates_schema(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    tables = _tables(conn)
    conn.close()
    assert "_meta" in tables
    assert "files" in tables
    assert "file_keywords" in tables


def test_ensure_creates_gekai_dir(tmp_path: Path) -> None:
    ensure(tmp_path).close()
    assert (tmp_path / ".gekai" / "workspace.db").exists()


def test_ensure_sets_schema_version(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    row = conn.execute(
        "SELECT value FROM _meta WHERE key = 'schema_version'"
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == SCHEMA_VERSION
    assert SCHEMA_VERSION == __version__


def test_ensure_idempotent(tmp_path: Path) -> None:
    conn1 = ensure(tmp_path)
    conn1.close()
    conn2 = ensure(tmp_path)
    tables = _tables(conn2)
    row = conn2.execute(
        "SELECT value FROM _meta WHERE key = 'schema_version'"
    ).fetchone()
    conn2.close()
    assert "_meta" in tables
    assert row[0] == SCHEMA_VERSION


def test_ensure_updates_version_on_older_db(tmp_path: Path) -> None:
    # DB exists with an older version — ensure() updates version in-place, keeps data
    db_path = tmp_path / ".gekai" / "workspace.db"
    (tmp_path / ".gekai").mkdir()
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE)")
    conn.execute(
        "CREATE TABLE file_keywords ("
        "file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE, "
        "keyword TEXT NOT NULL, UNIQUE(file_id, keyword))"
    )
    conn.execute("INSERT INTO _meta VALUES ('schema_version', 'old-version')")
    conn.execute("INSERT INTO files(path) VALUES ('existing/file.py')")
    conn.commit()
    conn.close()

    conn = ensure(tmp_path)
    row = conn.execute(
        "SELECT value FROM _meta WHERE key = 'schema_version'"
    ).fetchone()
    file_row = conn.execute(
        "SELECT path FROM files WHERE path = 'existing/file.py'"
    ).fetchone()
    conn.close()

    assert row[0] == SCHEMA_VERSION
    assert file_row is not None  # existing data preserved


def test_save_blast_radius_upserts_files(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    entries = [("src/main.py", ["main", "entry", "cli"])]
    save_blast_radius(conn, entries)
    row = conn.execute("SELECT path FROM files WHERE path = 'src/main.py'").fetchone()
    conn.close()
    assert row is not None


def test_save_blast_radius_upserts_keywords(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    entries = [("src/auth.py", ["login", "token", "session"])]
    save_blast_radius(conn, entries)
    rows = conn.execute(
        "SELECT keyword FROM file_keywords WHERE file_id = "
        "(SELECT id FROM files WHERE path = 'src/auth.py')"
    ).fetchall()
    conn.close()
    keywords = {r[0] for r in rows}
    assert keywords == {"login", "token", "session"}


def test_save_blast_radius_deduplicates_keywords(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    entries = [("src/auth.py", ["login", "login", "token"])]
    save_blast_radius(conn, entries)
    rows = conn.execute(
        "SELECT keyword FROM file_keywords WHERE file_id = "
        "(SELECT id FROM files WHERE path = 'src/auth.py')"
    ).fetchall()
    conn.close()
    assert len(rows) == 2


def test_save_blast_radius_multiple_entries(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    entries = [
        ("src/a.py", ["alpha"]),
        ("src/b.py", ["beta"]),
    ]
    save_blast_radius(conn, entries)
    count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    conn.close()
    assert count == 2


def test_save_blast_radius_idempotent(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    entries = [("src/a.py", ["alpha", "bravo"])]
    save_blast_radius(conn, entries)
    save_blast_radius(conn, entries)  # save twice
    count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    conn.close()
    assert count == 1


def test_save_blast_radius_skips_empty_keywords(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    entries = [("src/a.py", ["", "  ", "valid"])]
    save_blast_radius(conn, entries)
    rows = conn.execute(
        "SELECT keyword FROM file_keywords WHERE file_id = "
        "(SELECT id FROM files WHERE path = 'src/a.py')"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "valid"


def test_files_for_keywords_not_implemented(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    with pytest.raises(NotImplementedError):
        files_for_keywords(conn, ["test"])
    conn.close()
