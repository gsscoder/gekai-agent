# Caching
Catalog of caching mechanisms in gekai-agent. Each cache gets its own subsection below, documenting storage, schema, write/read paths, staleness handling, and wiring into the agent. Currently one cache exists (locator/workspace), but new caches should be added here as their own `##` sections.

## Locator cache (workspace.db)

**Storage**: SQLite at `<working_dir>/.gekai/workspace.db`, opened via `agent.workspace.db.ensure(working_dir)`. WAL mode, `busy_timeout=5000`, `foreign_keys=ON`.

**Schema** (`_DDL` in `agent/workspace/db.py`):
| Table | Columns |
|---|---|
| `_meta` | `key TEXT PRIMARY KEY`, `value TEXT NOT NULL` — currently holds `schema_version` |
| `files` | `id INTEGER PRIMARY KEY`, `path TEXT NOT NULL UNIQUE`, `size INTEGER`, `mtime_ns INTEGER`, `content_hash TEXT`, `indexed_at INTEGER` |
| `file_keywords` | `file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE`, `keyword TEXT NOT NULL`, `UNIQUE(file_id, keyword)` — inverted index, one row per (file, normalized lowercase keyword) |

**Versioning / upgrades**: `SCHEMA_VERSION = agent.__version__`. On `ensure()`, `_apply_schema` compares the stored `_meta.schema_version` to `SCHEMA_VERSION`; on mismatch it runs `_REBUILD_DDL` (drops `files`/`file_keywords`, cascade-drops `file_keywords` rows) and recreates from `_DDL` — alpha project, no migrations, cache is fully disposable.

`handle_db_upgrade(working_dir)` is a startup hook (called once from `GekaiAgent.__init__`, right after `working_dir` is set) that delegates to `ensure(working_dir).close()`, so an existing `workspace.db` from an older schema version gets rebuilt before the first per-turn cache access. It's intentionally a thin placeholder — the place to grow more elaborate upgrade/migration logic post-release if ever needed.

**Write side — `save_findings(conn, working_dir, entries)`**: called from `ws_manager`'s onboarding walk (`agent/workspace/indexer.py`) with `entries: list[tuple[path, keywords]]` (see Wiring below for how this call site changed). For each `(path, keywords)`:
- Stats the file and computes `_content_hash` (blake2b of file bytes, chunked via `_HASH_CHUNK_SIZE`), upserts the `files` row stamping `size`, `mtime_ns`, `content_hash`, `indexed_at` (epoch seconds).
- `OSError` during stat/hash is non-fatal — stored as NULLs (row still usable as a hint, just immediately stale).
- Keywords are lowercased/stripped and inserted into `file_keywords` (`INSERT OR IGNORE`, deduped via the UNIQUE constraint).

**Staleness — `is_fresh(working_dir, path, size, mtime_ns, content_hash)`**: two-tier check.
- Fast path: `stat()` and compare `size` + `mtime_ns` (no file read) — if both match, fresh.
- Fallback: if stat differs but `content_hash` matches `_content_hash` of current bytes, still fresh — covers the git-checkout case (mtime rewritten, content unchanged).
- Missing file or no stored metadata → stale.

**Read side — `find_candidates(conn, working_dir, keywords, limit=10) -> list[tuple[path, score]]`**: normalizes `keywords` (lowercase/strip), `SELECT`s `files` joined to `file_keywords` where `keyword IN (...)`, `GROUP BY file.id`, `score = COUNT(*)` (number of overlapping keywords), `ORDER BY score DESC`. For each result row, runs `is_fresh`; fresh rows become candidates, stale/missing-file rows are deleted from `files` (cascades to `file_keywords`) — lazy, self-healing eviction with no background GC. Returns top `limit` `(path, score)` pairs.

**`mine_keywords(request: str) -> list[str]`**: local, no-LLM keyword extractor for turning a user request into lookup keywords for `find_candidates`. Tokenizes on `[A-Za-z0-9]+` runs, lowercases, then for each adjacent token pair adds both the concatenation (`promptbuilder`) and underscore-join (`prompt_builder`) — so "prompt builder" matches keywords mined from identifiers like `PromptBuilder`/`prompt_builder`. Deduplicates preserving order.

**Wiring — write side alive, per-turn read side orphaned**: the dissolve-planner refactor (see
`docs/architecture.md`) deleted `FileLocator` and `GekaiAgent.locate`/`.rewrite` along with the
whole per-turn locate→rewrite pipeline stage, and with it the only call site for `find_hybrid`
(the per-turn hint lookup). `save_findings` is still called — now from `ws_manager`'s onboarding
walk (`agent/workspace/indexer.py`) rather than from a per-turn locate step — so the `files`/
`file_keywords` tables still get populated on workspace onboarding/reindex. `find_candidates`,
`find_hybrid`, and `mine_keywords` have no remaining callers: the main agent now decides which
files to touch itself, via its own read tools, with no locate/hint stage in front of it. The
hint-mode read path described below is currently dead code, pending either revival (e.g. as a hint
fed into a `delegate` step) or removal.

**Instrumentation (historical)**: the `locate` event (`agent.events`, see `logging.md`) used to carry `hints` (`len(hint_paths)`) and `overlap` (`|hint_paths ∩ final_entries| / |final_entries|`, rounded to 3 decimals, `0.0` if `final_entries` empty), in addition to `files` and `duration_ms`. That event source no longer fires — `locate` was emitted from the now-deleted TUI locate stage.

**Non-goals (v1)**:
- "Bypass mode" (skipping the locator entirely when cache coverage is high) — deferred pending overlap data.
- Cross-machine hash portability beyond blake2b-of-bytes.
- TTL/age-based eviction (`indexed_at` is stored but currently unused).

See `memory-bank/detailed-spec/locator-cache.md` for the original v1 design plan/build-order.
