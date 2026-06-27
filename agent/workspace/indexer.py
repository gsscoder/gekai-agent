from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from . import chunker, db, embed, scanner
from .symbols import _EXT_TO_LANG, extract_symbol_names


@dataclass
class IndexStats:
    file_count: int
    indexed_count: int
    skipped_fresh: int
    symbols_extracted: int
    symbol_files: int
    duration_ms: int
    chunk_count: int = 0


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


def _flush_chunk_batch(
    conn: sqlite3.Connection,
    working_dir: Path,
    chunk_batch: list[tuple[str, list[str]]],
) -> int:
    if not chunk_batch:
        return 0
    all_texts: list[str] = []
    mapping: list[tuple[str, int]] = []
    for rel_path, chunks in chunk_batch:
        mapping.append((rel_path, len(chunks)))
        all_texts.extend(chunks)
    vectors = embed.embed_texts(all_texts)
    vec_idx = 0
    total = 0
    for rel_path, count in mapping:
        db.save_chunk_vectors(
            conn, rel_path,
            all_texts[vec_idx : vec_idx + count],
            vectors[vec_idx : vec_idx + count],
        )
        total += count
        vec_idx += count
    return total


def build_index(
    working_dir: Path,
    conn: sqlite3.Connection,
    *,
    batch_size: int = 200,
) -> IndexStats:
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
    chunk_count = 0
    keyword_batch: list[tuple[str, list[str]]] = []
    chunk_batch: list[tuple[str, list[str]]] = []

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

        keyword_batch.append((rel_path, keywords))
        indexed_count += 1

        try:
            full_path = working_dir / rel_path
            if full_path.stat().st_size <= chunker.MAX_FILE_BYTES:
                text = full_path.read_text(encoding="utf-8", errors="replace")
                chunks = chunker.chunk_text(text)
                if chunks:
                    chunk_batch.append((rel_path, chunks))
        except OSError:
            pass

        if len(keyword_batch) >= batch_size:
            db.save_findings(conn, working_dir, keyword_batch)
            chunk_count += _flush_chunk_batch(conn, working_dir, chunk_batch)
            keyword_batch = []
            chunk_batch = []

    if keyword_batch:
        db.save_findings(conn, working_dir, keyword_batch)
        chunk_count += _flush_chunk_batch(conn, working_dir, chunk_batch)

    return IndexStats(
        file_count=len(all_files),
        indexed_count=indexed_count,
        skipped_fresh=skipped_fresh,
        symbols_extracted=symbols_extracted,
        symbol_files=symbol_files,
        chunk_count=chunk_count,
        duration_ms=int((time.monotonic() - start) * 1000),
    )
