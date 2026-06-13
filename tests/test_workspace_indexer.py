from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

from agent.workspace.db import ensure, find_candidates, mine_keywords
from agent.workspace.indexer import build_index
from agent.workspace.scanner import list_files


def _keywords_for(conn: sqlite3.Connection, path: str) -> set[str]:
    rows = conn.execute(
        "SELECT fk.keyword FROM file_keywords fk "
        "JOIN files f ON f.id = fk.file_id WHERE f.path = ?",
        (path,),
    ).fetchall()
    return {r[0] for r in rows}


def _file_row(conn: sqlite3.Connection, path: str) -> tuple | None:
    return conn.execute(
        "SELECT size, mtime_ns, content_hash FROM files WHERE path = ?", (path,)
    ).fetchone()


def test_build_index_populates_all_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.js").write_text("const x = 1;\n")
    (tmp_path / "README.md").write_text("# hi\n")

    conn = ensure(tmp_path)
    stats = build_index(tmp_path, conn)

    assert stats.file_count == 3
    assert stats.indexed_count == 3
    assert stats.skipped_fresh == 0

    for path in ("a.py", "b.js", "README.md"):
        row = _file_row(conn, path)
        assert row is not None
        size, mtime_ns, content_hash = row
        assert size is not None
        assert mtime_ns is not None
        assert content_hash is not None

    conn.close()


def test_build_index_extracts_path_tokens(tmp_path: Path) -> None:
    nested = tmp_path / "src" / "utils"
    nested.mkdir(parents=True)
    (nested / "helper.py").write_text("x = 1\n")

    conn = ensure(tmp_path)
    build_index(tmp_path, conn)

    keywords = _keywords_for(conn, "src/utils/helper.py")
    conn.close()

    assert {"src", "utils", "helper", "py"} <= keywords


def test_build_index_extracts_symbols(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("def foo():\n    pass\n\n\nclass Bar:\n    pass\n")

    conn = ensure(tmp_path)
    stats = build_index(tmp_path, conn)

    keywords = _keywords_for(conn, "module.py")
    conn.close()

    assert "foo" in keywords
    assert "bar" in keywords  # save_findings lowercases keywords
    assert stats.symbols_extracted >= 2
    assert stats.symbol_files == 1


def test_build_index_no_symbols_for_unsupported_ext(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# heading\n\nsome content here\n")

    conn = ensure(tmp_path)
    stats = build_index(tmp_path, conn)

    keywords = _keywords_for(conn, "notes.md")
    conn.close()

    assert keywords == {"notes", "md"}
    assert stats.symbol_files == 0
    assert stats.symbols_extracted == 0


def test_build_index_skips_fresh_on_second_run(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")

    conn = ensure(tmp_path)
    first = build_index(tmp_path, conn)
    assert first.indexed_count == 2
    assert first.skipped_fresh == 0

    second = build_index(tmp_path, conn)
    conn.close()

    assert second.file_count == 2
    assert second.indexed_count == 0
    assert second.skipped_fresh == 2


def test_build_index_reindexes_modified_file(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("x = 1\n")

    conn = ensure(tmp_path)
    build_index(tmp_path, conn)

    # modify content and bump mtime so the stat-freshness check misses
    target.write_text("def newly_added():\n    pass\n")
    new_mtime = time.time() + 5
    os.utime(target, (new_mtime, new_mtime))

    stats = build_index(tmp_path, conn)
    keywords = _keywords_for(conn, "a.py")
    conn.close()

    assert stats.indexed_count == 1
    assert stats.skipped_fresh == 0
    assert "newly_added" in keywords


def test_build_index_findable_by_find_candidates(tmp_path: Path) -> None:
    nested = tmp_path / "agent" / "workspace"
    nested.mkdir(parents=True)
    (nested / "indexer.py").write_text("def build_index():\n    pass\n")

    conn = ensure(tmp_path)
    build_index(tmp_path, conn)

    keywords = mine_keywords("how does the indexer work")
    candidates = find_candidates(conn, tmp_path, keywords)
    conn.close()

    paths = [path for path, _ in candidates]
    assert "agent/workspace/indexer.py" in paths


def test_build_index_stats_match_scanner(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.txt").write_text("hello\n")

    conn = ensure(tmp_path)
    stats = build_index(tmp_path, conn)
    conn.close()

    assert stats.file_count == len(list_files(tmp_path))
    assert stats.duration_ms >= 0
