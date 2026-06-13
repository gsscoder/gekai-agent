from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from . import db, scanner
from .symbols import _EXT_TO_LANG, extract_symbol_names


@dataclass
class IndexStats:
    file_count: int
    indexed_count: int
    skipped_fresh: int
    symbols_extracted: int
    symbol_files: int
    duration_ms: int


def _stat_fresh(
    working_dir: Path,
    rel_path: str,
    existing: tuple[int | None, int | None] | None,
) -> bool:
    """Stat-only freshness check (no content-hash fallback).

    Used during bulk indexing where reading every file twice (once to hash,
    once to extract keywords) would be wasteful; a stat mismatch simply
    triggers re-indexing instead.
    """
    if existing is None:
        return False
    size, mtime_ns = existing
    if size is None or mtime_ns is None:
        return False
    try:
        st = (working_dir / rel_path).stat()
    except OSError:
        return False
    return st.st_size == size and st.st_mtime_ns == mtime_ns


def build_index(
    working_dir: Path,
    conn: sqlite3.Connection,
    *,
    batch_size: int = 200,
) -> IndexStats:
    """Eagerly populate `files`/`file_keywords` for every file in the repo.

    Flips the locator cache from reactive (populated only after `FileLocator`
    visits a file) to proactive: path tokens are extracted for every file,
    and tree-sitter symbol names for languages in `_EXT_TO_LANG`. Files whose
    stat (size + mtime_ns) matches the existing row are skipped.
    """
    start = time.monotonic()

    all_files = scanner.list_files(working_dir)
    existing_rows = conn.execute("SELECT path, size, mtime_ns FROM files").fetchall()
    existing: dict[str, tuple[int | None, int | None]] = {
        path: (size, mtime_ns) for path, size, mtime_ns in existing_rows
    }

    indexed_count = 0
    skipped_fresh = 0
    symbols_extracted = 0
    symbol_files = 0
    batch: list[tuple[str, list[str]]] = []

    for rel_path in all_files:
        if _stat_fresh(working_dir, rel_path, existing.get(rel_path)):
            skipped_fresh += 1
            continue

        keywords = db._TOKEN_RE.findall(rel_path)

        lang_name = _EXT_TO_LANG.get(Path(rel_path).suffix.lower())
        if lang_name is not None:
            symbol_files += 1
            names = extract_symbol_names(working_dir / rel_path, lang_name)
            if names:
                keywords.extend(names)
                symbols_extracted += len(names)

        batch.append((rel_path, keywords))
        indexed_count += 1

        if len(batch) >= batch_size:
            db.save_findings(conn, working_dir, batch)
            batch = []

    if batch:
        db.save_findings(conn, working_dir, batch)

    return IndexStats(
        file_count=len(all_files),
        indexed_count=indexed_count,
        skipped_fresh=skipped_fresh,
        symbols_extracted=symbols_extracted,
        symbol_files=symbol_files,
        duration_ms=int((time.monotonic() - start) * 1000),
    )
