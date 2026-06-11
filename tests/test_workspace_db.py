from __future__ import annotations

import sqlite3
from pathlib import Path

from agent import __version__
from agent.workspace.db import (
    SCHEMA_VERSION,
    _content_hash,
    ensure,
    find_candidates,
    handle_db_upgrade,
    is_fresh,
    mine_keywords,
    save_findings,
)


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def _insert_file(
    conn: sqlite3.Connection,
    path: str,
    keywords: list[str],
    size: int | None = None,
    mtime_ns: int | None = None,
    content_hash: str | None = None,
) -> int:
    conn.execute(
        "INSERT INTO files(path, size, mtime_ns, content_hash) VALUES (?, ?, ?, ?)",
        (path, size, mtime_ns, content_hash),
    )
    file_id = conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()[0]
    for kw in keywords:
        conn.execute("INSERT INTO file_keywords(file_id, keyword) VALUES (?, ?)", (file_id, kw))
    conn.commit()
    return file_id


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


def test_ensure_rebuilds_on_version_mismatch(tmp_path: Path) -> None:
    # DB exists with an older schema_version — ensure() rebuilds files/file_keywords
    # rather than migrating: the cache is disposable, stale rows are dropped
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
    cols = {r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()}
    conn.close()

    assert row[0] == SCHEMA_VERSION
    assert file_row is None  # stale-schema rows dropped on rebuild
    assert {"size", "mtime_ns", "content_hash", "indexed_at"} <= cols


def test_handle_db_upgrade_rebuilds_on_version_mismatch(tmp_path: Path) -> None:
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

    handle_db_upgrade(tmp_path)

    conn = ensure(tmp_path)
    row = conn.execute(
        "SELECT value FROM _meta WHERE key = 'schema_version'"
    ).fetchone()
    file_row = conn.execute(
        "SELECT path FROM files WHERE path = 'existing/file.py'"
    ).fetchone()
    conn.close()

    assert row[0] == SCHEMA_VERSION
    assert file_row is None  # stale-schema rows dropped on rebuild


def test_ensure_preserves_data_when_version_matches(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    save_findings(conn, tmp_path, [("src/a.py", ["alpha"])])
    conn.close()

    conn = ensure(tmp_path)
    row = conn.execute("SELECT path FROM files WHERE path = 'src/a.py'").fetchone()
    conn.close()
    assert row is not None


def test_files_table_has_cache_columns(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()}
    conn.close()
    assert {"size", "mtime_ns", "content_hash", "indexed_at"} <= cols


def test_save_findings_upserts_files(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    entries = [("src/main.py", ["main", "entry", "cli"])]
    save_findings(conn, tmp_path, entries)
    row = conn.execute("SELECT path FROM files WHERE path = 'src/main.py'").fetchone()
    conn.close()
    assert row is not None


def test_save_findings_upserts_keywords(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    entries = [("src/auth.py", ["login", "token", "session"])]
    save_findings(conn, tmp_path, entries)
    rows = conn.execute(
        "SELECT keyword FROM file_keywords WHERE file_id = "
        "(SELECT id FROM files WHERE path = 'src/auth.py')"
    ).fetchall()
    conn.close()
    keywords = {r[0] for r in rows}
    assert keywords == {"login", "token", "session"}


def test_save_findings_deduplicates_keywords(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    entries = [("src/auth.py", ["login", "login", "token"])]
    save_findings(conn, tmp_path, entries)
    rows = conn.execute(
        "SELECT keyword FROM file_keywords WHERE file_id = "
        "(SELECT id FROM files WHERE path = 'src/auth.py')"
    ).fetchall()
    conn.close()
    assert len(rows) == 2


def test_save_findings_multiple_entries(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    entries = [
        ("src/a.py", ["alpha"]),
        ("src/b.py", ["beta"]),
    ]
    save_findings(conn, tmp_path, entries)
    count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    conn.close()
    assert count == 2


def test_save_findings_idempotent(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    entries = [("src/a.py", ["alpha", "bravo"])]
    save_findings(conn, tmp_path, entries)
    save_findings(conn, tmp_path, entries)  # save twice
    count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    conn.close()
    assert count == 1


def test_save_findings_skips_empty_keywords(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    entries = [("src/a.py", ["", "  ", "valid"])]
    save_findings(conn, tmp_path, entries)
    rows = conn.execute(
        "SELECT keyword FROM file_keywords WHERE file_id = "
        "(SELECT id FROM files WHERE path = 'src/a.py')"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "valid"


def test_save_findings_stamps_hash_and_stat(tmp_path: Path) -> None:
    f = tmp_path / "src" / "main.py"
    f.parent.mkdir(parents=True)
    f.write_text("print('hi')")

    conn = ensure(tmp_path)
    save_findings(conn, tmp_path, [("src/main.py", ["main"])])
    row = conn.execute(
        "SELECT size, mtime_ns, content_hash, indexed_at FROM files WHERE path = 'src/main.py'"
    ).fetchone()
    conn.close()

    size, mtime_ns, content_hash, indexed_at = row
    assert size == f.stat().st_size
    assert mtime_ns is not None
    assert content_hash == _content_hash(f)
    assert indexed_at is not None


def test_save_findings_missing_file_stores_null_metadata(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    save_findings(conn, tmp_path, [("does/not/exist.py", ["x"])])
    row = conn.execute(
        "SELECT size, mtime_ns, content_hash FROM files WHERE path = 'does/not/exist.py'"
    ).fetchone()
    conn.close()
    assert row == (None, None, None)


def test_save_findings_then_find_candidates_returns_fresh_row(tmp_path: Path) -> None:
    # end-to-end: save_findings stamps metadata that is_fresh accepts immediately
    f = tmp_path / "src" / "main.py"
    f.parent.mkdir(parents=True)
    f.write_text("print('hi')")

    conn = ensure(tmp_path)
    save_findings(conn, tmp_path, [("src/main.py", ["main", "entry"])])
    result = find_candidates(conn, tmp_path, ["main"])
    conn.close()
    assert result == [("src/main.py", 1)]


def test_mine_keywords_lowercases_single_token() -> None:
    assert mine_keywords("PromptBuilder") == ["promptbuilder"]


def test_mine_keywords_splits_on_non_alnum() -> None:
    result = mine_keywords("agent/workspace/db.py")
    for kw in ("agent", "workspace", "db", "py"):
        assert kw in result


def test_mine_keywords_adds_adjacent_joins() -> None:
    result = mine_keywords("prompt builder")
    assert result == ["prompt", "builder", "promptbuilder", "prompt_builder"]


def test_mine_keywords_dedupes_preserving_order() -> None:
    result = mine_keywords("alpha alpha beta")
    assert result == [
        "alpha", "beta", "alphaalpha", "alpha_alpha", "alphabeta", "alpha_beta",
    ]


def test_mine_keywords_empty_string_returns_empty() -> None:
    assert mine_keywords("") == []


def test_mine_keywords_single_token_no_joins() -> None:
    assert mine_keywords("calculator") == ["calculator"]


def test_mine_keywords_output_matches_find_candidates_normalization(tmp_path: Path) -> None:
    # mined keywords must match the lowercase form stored by save_findings
    f = tmp_path / "src" / "prompt_builder.py"
    f.parent.mkdir(parents=True)
    f.write_text("class PromptBuilder: ...")

    conn = ensure(tmp_path)
    save_findings(conn, tmp_path, [("src/prompt_builder.py", ["PromptBuilder", "prompt_builder"])])
    mined = mine_keywords("update the prompt builder")
    result = find_candidates(conn, tmp_path, mined)
    conn.close()
    # both stored keywords ("promptbuilder", "prompt_builder") overlap with mined
    assert result == [("src/prompt_builder.py", 2)]


def test_content_hash_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("hello world")
    assert _content_hash(f) == _content_hash(f)


def test_content_hash_differs_for_different_content(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello")
    b.write_text("world")
    assert _content_hash(a) != _content_hash(b)


def test_is_fresh_missing_file_is_stale(tmp_path: Path) -> None:
    assert is_fresh(tmp_path, "does/not/exist.py", None, None, None) is False


def test_is_fresh_matching_stat_is_fresh(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("content")
    st = f.stat()
    assert is_fresh(tmp_path, "a.py", st.st_size, st.st_mtime_ns, None) is True


def test_is_fresh_stat_mismatch_with_matching_hash_is_fresh(tmp_path: Path) -> None:
    # git-checkout case: mtime/size on record no longer match, but content is identical
    f = tmp_path / "a.py"
    f.write_text("content")
    digest = _content_hash(f)
    assert is_fresh(tmp_path, "a.py", -1, -1, digest) is True


def test_is_fresh_stat_and_hash_mismatch_is_stale(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("content")
    assert is_fresh(tmp_path, "a.py", -1, -1, "deadbeef") is False


def test_is_fresh_no_metadata_is_stale(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("content")
    assert is_fresh(tmp_path, "a.py", None, None, None) is False


def test_find_candidates_empty_keywords_returns_empty(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    f = tmp_path / "a.py"
    f.write_text("x")
    st = f.stat()
    _insert_file(conn, "a.py", ["alpha"], st.st_size, st.st_mtime_ns)
    result = find_candidates(conn, tmp_path, [])
    conn.close()
    assert result == []


def test_find_candidates_no_overlap_returns_empty(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    f = tmp_path / "a.py"
    f.write_text("x")
    st = f.stat()
    _insert_file(conn, "a.py", ["alpha"], st.st_size, st.st_mtime_ns)
    result = find_candidates(conn, tmp_path, ["nomatch"])
    conn.close()
    assert result == []


def test_find_candidates_scores_by_keyword_overlap(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("a")
    b.write_text("b")
    sa, sb = a.stat(), b.stat()
    _insert_file(conn, "a.py", ["alpha", "beta"], sa.st_size, sa.st_mtime_ns)
    _insert_file(conn, "b.py", ["alpha"], sb.st_size, sb.st_mtime_ns)
    result = find_candidates(conn, tmp_path, ["alpha", "beta"])
    conn.close()
    assert result[0] == ("a.py", 2)
    assert result[1] == ("b.py", 1)


def test_find_candidates_respects_limit(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    for i in range(5):
        p = tmp_path / f"f{i}.py"
        p.write_text("x")
        st = p.stat()
        _insert_file(conn, f"f{i}.py", ["alpha"], st.st_size, st.st_mtime_ns)
    result = find_candidates(conn, tmp_path, ["alpha"], limit=2)
    conn.close()
    assert len(result) == 2


def test_find_candidates_keeps_fresh_via_content_hash(tmp_path: Path) -> None:
    # git-checkout case: stat no longer matches, content hash does
    f = tmp_path / "a.py"
    f.write_text("content")
    digest = _content_hash(f)
    conn = ensure(tmp_path)
    _insert_file(conn, "a.py", ["alpha"], size=-1, mtime_ns=-1, content_hash=digest)
    result = find_candidates(conn, tmp_path, ["alpha"])
    conn.close()
    assert result == [("a.py", 1)]


def test_find_candidates_evicts_stale_rows(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("original")
    conn = ensure(tmp_path)
    _insert_file(conn, "a.py", ["alpha"], size=999, mtime_ns=123, content_hash="deadbeef")
    result = find_candidates(conn, tmp_path, ["alpha"])
    files_left = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    keywords_left = conn.execute("SELECT COUNT(*) FROM file_keywords").fetchone()[0]
    conn.close()
    assert result == []
    assert files_left == 0
    assert keywords_left == 0  # cascade delete


def test_find_candidates_evicts_missing_file(tmp_path: Path) -> None:
    conn = ensure(tmp_path)
    _insert_file(conn, "gone.py", ["alpha"], size=1, mtime_ns=1)
    result = find_candidates(conn, tmp_path, ["alpha"])
    files_left = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    conn.close()
    assert result == []
    assert files_left == 0
