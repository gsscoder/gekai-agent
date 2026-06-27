# Workspace caching

Gekai caches per-workspace metadata in `.gekai/workspace.db` (SQLite). The cache is **disposable** — a schema or embed-dim mismatch drops and rebuilds it on next startup; no migration shims.

## Schema version

`_meta.schema_version` = current `__version__`. On mismatch all tables are dropped and rebuilt.

## Embed dim

`_meta.embed_dim` = current `EMBED_DIM` (384 for bge-small-en-v1.5). On mismatch `file_chunks` is rebuilt. Change the model → dim → triggers rebuild automatically.

## Tables

| Table | Purpose |
|---|---|
| `files` | Path + stat (size, mtime_ns, content_hash) for freshness |
| `file_keywords` | Token keywords per file for lexical retrieval |
| `file_chunks` | vec0 virtual table — chunk embeddings for semantic retrieval |
| `_meta` | Key/value store for schema_version and embed_dim |

## Freshness

Two-tier check: stat (size + mtime_ns) first, then blake2b hash fallback (handles git checkout mtime resets). Stale rows are evicted on read.
