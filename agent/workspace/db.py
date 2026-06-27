from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from pathlib import Path

import sqlite_vec

from agent import __version__

_HASH_CHUNK_SIZE = 65536
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

SCHEMA_VERSION = __version__
_EMBED_DIM = 384

_META_DDL = """
CREATE TABLE IF NOT EXISTS _meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_DDL = f"""
CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    size         INTEGER,
    mtime_ns     INTEGER,
    content_hash TEXT,
    indexed_at   INTEGER
);

CREATE TABLE IF NOT EXISTS file_keywords (
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    keyword TEXT NOT NULL,
    UNIQUE(file_id, keyword)
);

CREATE INDEX IF NOT EXISTS idx_file_keywords_keyword ON file_keywords(keyword);

CREATE VIRTUAL TABLE IF NOT EXISTS file_chunks USING vec0(
    +file_id INTEGER,
    +chunk_idx INTEGER,
    vector FLOAT[{_EMBED_DIM}]
);
"""

_REBUILD_DDL = """
DROP TABLE IF EXISTS file_chunks;
DROP TABLE IF EXISTS file_keywords;
DROP TABLE IF EXISTS files;
"""


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_META_DDL)
    version_row = conn.execute(
        "SELECT value FROM _meta WHERE key = 'schema_version'"
    ).fetchone()
    dim_row = conn.execute(
        "SELECT value FROM _meta WHERE key = 'embed_dim'"
    ).fetchone()
    schema_stale = version_row is not None and version_row[0] != SCHEMA_VERSION
    dim_stale = dim_row is not None and dim_row[0] != str(_EMBED_DIM)
    if schema_stale or dim_stale:
        conn.executescript(_REBUILD_DDL)
    conn.executescript(_DDL)
    conn.execute(
        "INSERT OR REPLACE INTO _meta(key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    conn.execute(
        "INSERT OR REPLACE INTO _meta(key, value) VALUES ('embed_dim', ?)",
        (str(_EMBED_DIM),),
    )
    conn.commit()


def ensure(working_dir: Path) -> sqlite3.Connection:
    gekai_dir = working_dir / ".gekai"
    gekai_dir.mkdir(exist_ok=True)
    db_path = gekai_dir / "workspace.db"

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")

    _apply_schema(conn)
    return conn


def handle_db_upgrade(working_dir: Path) -> None:
    """Startup hook: bring an existing workspace.db up to the current schema.

    For now this delegates to `ensure`, which drops and rebuilds `files`/
    `file_keywords` when `schema_version` doesn't match (alpha, no
    migrations — cache is disposable). Expand here as upgrade needs grow
    post-release.
    """
    ensure(working_dir).close()


def save_findings(
    conn: sqlite3.Connection,
    working_dir: Path,
    entries: list[tuple[str, list[str]]],
) -> None:
    now = int(time.time())
    for path, keywords in entries:
        size: int | None = None
        mtime_ns: int | None = None
        content_hash: str | None = None
        try:
            full_path = working_dir / path
            st = full_path.stat()
            size = st.st_size
            mtime_ns = st.st_mtime_ns
            content_hash = _content_hash(full_path)
        except OSError:
            pass  # non-fatal: row stays usable as a hint, just stale until re-stamped

        conn.execute("INSERT OR IGNORE INTO files(path) VALUES (?)", (path,))
        conn.execute(
            "UPDATE files SET size = ?, mtime_ns = ?, content_hash = ?, indexed_at = ? "
            "WHERE path = ?",
            (size, mtime_ns, content_hash, now, path),
        )
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


def save_chunk_vectors(
    conn: sqlite3.Connection,
    rel_path: str,
    chunks: list[str],
    vectors: list[list[float]],
) -> None:
    row = conn.execute("SELECT id FROM files WHERE path = ?", (rel_path,)).fetchone()
    if row is None:
        return
    file_id = row[0]
    old_rowids = conn.execute(
        "SELECT rowid FROM file_chunks WHERE file_id = ?", (file_id,)
    ).fetchall()
    if old_rowids:
        conn.executemany("DELETE FROM file_chunks WHERE rowid = ?", old_rowids)
    for chunk_idx, (_, vec) in enumerate(zip(chunks, vectors)):
        conn.execute(
            "INSERT INTO file_chunks(file_id, chunk_idx, vector) VALUES (?, ?, ?)",
            (file_id, chunk_idx, sqlite_vec.serialize_float32(vec)),
        )
    conn.commit()


def _content_hash(path: Path) -> str:
    h = hashlib.blake2b()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def is_fresh(
    working_dir: Path,
    path: str,
    size: int | None,
    mtime_ns: int | None,
    content_hash: str | None,
) -> bool:
    """Two-tier staleness check for a cached file row.

    Fast path: stat size + mtime_ns match -> fresh, no read.
    Fallback: blake2b content hash matches -> fresh (covers the
    git-checkout case where mtime is rewritten but content is unchanged).
    """
    try:
        st = (working_dir / path).stat()
    except OSError:
        return False
    if size is not None and mtime_ns is not None \
            and st.st_size == size and st.st_mtime_ns == mtime_ns:
        return True
    if content_hash is None:
        return False
    return _content_hash(working_dir / path) == content_hash


def find_candidates(
    conn: sqlite3.Connection,
    working_dir: Path,
    keywords: list[str],
    limit: int = 10,
) -> list[tuple[str, int]]:
    """Look up cached candidate paths by keyword overlap.

    Score = number of `keywords` that match a file's stored keywords.
    The SQL query itself is bounded (ORDER BY score DESC LIMIT) so a
    generic keyword can't pull thousands of rows into Python. Rows that
    fail `is_fresh` are evicted (cascades to file_keywords) and excluded
    from the result; stale rows ranked below the SQL limit are not
    evicted by this call.
    """
    normalized = [kw.strip().lower() for kw in keywords if kw.strip()]
    if not normalized:
        return []

    # Bound the SQL result so a generic keyword can't return thousands of rows;
    # the extra headroom over `limit` absorbs rows that fail is_fresh below.
    sql_limit = limit * 3
    placeholders = ", ".join("?" for _ in normalized)
    rows = conn.execute(
        "SELECT f.id, f.path, f.size, f.mtime_ns, f.content_hash, COUNT(*) AS score "
        "FROM files f JOIN file_keywords fk ON fk.file_id = f.id "
        f"WHERE fk.keyword IN ({placeholders}) "
        "GROUP BY f.id ORDER BY score DESC LIMIT ?",
        [*normalized, sql_limit],
    ).fetchall()

    candidates: list[tuple[str, int]] = []
    stale_ids: list[int] = []
    for file_id, path, size, mtime_ns, content_hash, score in rows:
        if is_fresh(working_dir, path, size, mtime_ns, content_hash):
            candidates.append((path, score))
        else:
            stale_ids.append(file_id)

    if stale_ids:
        conn.executemany("DELETE FROM files WHERE id = ?", [(i,) for i in stale_ids])
        conn.commit()

    return candidates[:limit]


def find_semantic(
    conn: sqlite3.Connection,
    working_dir: Path,
    query: str,
    k: int = 10,
) -> list[tuple[str, float]]:
    from .embed import embed_texts  # lazy: keeps fastembed out of module-level import
    vec_bytes = sqlite_vec.serialize_float32(embed_texts([query])[0])
    rows = conn.execute(
        "SELECT file_id, distance FROM file_chunks"
        " WHERE vector MATCH ? AND k = ? ORDER BY distance",
        (vec_bytes, k * 3),
    ).fetchall()

    results: list[tuple[str, float]] = []
    stale_ids: list[int] = []
    seen: set[int] = set()
    for file_id, dist in rows:
        if file_id in seen:
            continue
        seen.add(file_id)
        row = conn.execute(
            "SELECT path, size, mtime_ns, content_hash FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
        if row is None:
            stale_ids.append(file_id)
            continue
        path, size, mtime_ns, content_hash = row
        if is_fresh(working_dir, path, size, mtime_ns, content_hash):
            results.append((path, dist))
        else:
            stale_ids.append(file_id)

    if stale_ids:
        conn.executemany("DELETE FROM files WHERE id = ?", [(i,) for i in stale_ids])
        conn.commit()

    return results[:k]


_RRF_C = 60


def find_hybrid(
    conn: sqlite3.Connection,
    working_dir: Path,
    keywords: list[str],
    query: str,
    k: int = 10,
) -> list[str]:
    lexical = find_candidates(conn, working_dir, keywords, limit=k * 2)
    semantic = find_semantic(conn, working_dir, query, k=k * 2)

    scores: dict[str, float] = {}
    for rank, (path, _) in enumerate(lexical):
        scores[path] = scores.get(path, 0.0) + 1.0 / (_RRF_C + rank)
    for rank, (path, _) in enumerate(semantic):
        scores[path] = scores.get(path, 0.0) + 1.0 / (_RRF_C + rank)

    return sorted(scores, key=lambda p: scores[p], reverse=True)[:k]


def mine_keywords(request: str) -> list[str]:
    """Extract lookup keywords from a user request for `find_candidates`.

    Tokenizes on alphanumeric runs and lowercases (matching how
    `file_keywords` are normalized), then adds adjacent-token joins so a
    multi-word signal like "prompt builder" also matches keywords mined as
    "promptbuilder" or "prompt_builder". Deduplicates, preserving order.
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(request)]

    keywords: list[str] = []
    seen: set[str] = set()

    def _add(kw: str) -> None:
        if kw not in seen:
            seen.add(kw)
            keywords.append(kw)

    for tok in tokens:
        _add(tok)
    for a, b in zip(tokens, tokens[1:]):
        _add(a + b)
        _add(f"{a}_{b}")

    return keywords
