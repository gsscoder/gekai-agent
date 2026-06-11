# Locator Cache — v1 Plan

Status: **planned, not implemented**. This document is the build spec for v1.

## Goal

Turn the write-only `workspace.db` into a working cache for `FileLocator`.
Persisted locator findings (file paths + mined keywords) become an inverted
index that, on a new request, supplies **candidate paths** to bias the locator
— cutting blind-grep iterations without skipping the LLM call.

v1 is **hint mode only**: the locator still runs every turn and verifies every
file. The cache biases it; it never replaces it. Bypassing the locator entirely
is explicitly out of scope (see Non-goals).

## Naming change (do this first)

"Blast radius" is the **guard's** concept (`evaluate_blast_radius_gate`,
`agent/pipeline/blast_radius.py`) — the area metric that decides whether a turn
is too broad. Persisting locator output is a **general file_locator** concern,
not a guard concern. The current name `save_blast_radius` conflates the two.

Rename, no compatibility shims (alpha):

| Old                      | New                  | Direction | Notes                                        |
|--------------------------|----------------------|-----------|----------------------------------------------|
| `save_blast_radius`      | `save_findings`      | write     | persists what the locator found              |
| — (new)                  | `find_candidates`    | read      | keyword lookup → ranked, freshness-filtered  |
| — (new)                  | `is_fresh`           | helper    | per-file staleness check                     |
| — (new)                  | `_content_hash`      | helper    | blake2b of file bytes                        |

- Vocabulary: locator produces **findings** (write); a new request reads
  **candidates** (read). Coherent narrative, no "blast radius" leakage.
- Table names `files` / `file_keywords` are already generic — keep them.
- Guard code (`blast_radius.py`, `evaluate_blast_radius_gate`) is untouched.
- Call site `agent/agent.py:164` updates to `save_findings`.

## Schema evolution

Alpha, no-backward-compat → bump `_DDL` + `SCHEMA_VERSION` (= `__version__`),
rebuild table. Add to `files`:

```sql
CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    size         INTEGER,        -- bytes, from stat
    mtime_ns     INTEGER,        -- st_mtime_ns, from stat
    content_hash TEXT,           -- blake2b(file bytes), hex
    indexed_at   INTEGER         -- epoch seconds, debug / future TTL
);
```

`file_keywords` unchanged.

## Staleness — two-tier, lazy

```python
def is_fresh(working_dir: Path, path: str,
             size: int | None, mtime_ns: int | None,
             content_hash: str | None) -> bool:
    p = working_dir / path
    try:
        st = p.stat()
    except OSError:
        return False                                  # gone → stale
    if size is not None and mtime_ns is not None \
       and st.st_size == size and st.st_mtime_ns == mtime_ns:
        return True                                   # fast path, no read
    return content_hash is not None and _content_hash(p) == content_hash
                                                      # git-checkout case: touched, identical
```

- Cheap path: `stat` only (size + mtime_ns), no file read.
- Authoritative fallback: `blake2b` of bytes — rescues the git-checkout case
  (mtime rewritten, content identical) and confirms real edits.
- Eviction is **lazy**: `find_candidates` deletes any row that fails `is_fresh`
  (`DELETE FROM files` → CASCADE drops its keywords). Self-healing index, no
  background GC pass.

## Read side — `find_candidates`

```python
def find_candidates(conn, working_dir: Path, keywords: list[str],
                    limit: int = 10) -> list[tuple[str, int]]:
    # 1. SELECT path, COUNT(matching keywords) AS score
    #    FROM files JOIN file_keywords ... WHERE keyword IN (?)
    #    GROUP BY path ORDER BY score DESC
    # 2. for each row: is_fresh? keep : evict (DELETE)
    # 3. return top-`limit` fresh (path, score)
```

Score = count of overlapping keywords (inverted-index overlap).

## Write side — extend `save_findings`

Same INSERT logic as today, plus stamp `size`, `mtime_ns`, `content_hash`,
`indexed_at` per file (one `stat` + one `blake2b` per inserted path).
`stat`/hash failures are non-fatal — store NULLs, row still usable as a hint
(it'll just be treated as stale on next read until re-stamped).

## Keyword miner (local, no LLM)

New helper — shared between the cache lookup and (later) the locator's own
signal-mining step:

```python
def mine_keywords(request: str) -> list[str]:
    # tokenize; expand each token across casings/joins:
    # "prompt builder" -> prompt, builder, promptbuilder,
    #   PromptBuilder, prompt_builder, build_prompt
    # lowercase, dedup, preserve order
```

Decision: **local extractor, not an LLM pre-call.** A hint tolerates noisy
keywords (locator verifies anyway); an LLM pre-call would add latency/cost to
every turn and partly defeat the speedup.

## Wiring

- `FileLocator.locate(working_dir, request, hint_paths=None)` gains the optional
  arg. When present, system prompt gets a line:
  `"previously-relevant for these signals: <hint_paths> — verify before listing"`.
- `agent/agent.py` `locate()` pre-step:
  1. `kw = mine_keywords(user_input)`
  2. `conn = workspace_db.ensure(working_dir)`
  3. `cands = workspace_db.find_candidates(conn, working_dir, kw)`
  4. pass `[p for p, _ in cands]` as `hint_paths`.
- Write path (`agent.py:164`) → renamed `save_findings`, now also stamps hashes.

## Instrumentation (do in v1, pays off for v2)

Log per turn (existing `EventLogger`): `hint_paths` count, final `entries`
count, and overlap = `|hint ∩ final| / |final|`. This is the hit-rate data
needed to decide whether Bypass mode is ever safe — collect it from day one
instead of guessing later.

## Non-goals (v1)

- **Bypass mode** (skip the LLM when cached candidates fully cover + fresh).
  Deferred — keyword overlap ≠ semantic match; enable only if logged hit rate
  proves precision is high.
- Cross-machine hash portability beyond blake2b-of-bytes.
- TTL / age-based eviction (`indexed_at` stored now, unused until needed).

## Build order

1. Rename `save_blast_radius` → `save_findings`; update call site + tests.
2. Schema: add columns, bump `SCHEMA_VERSION`; update `test_workspace_db.py`.
3. `_content_hash` + `is_fresh` helpers (+ tests, incl. git-checkout case:
   same content, bumped mtime → fresh).
4. `find_candidates` with lazy eviction (+ tests: scoring, stale drop, missing
   file drop).
5. Stamp hashes inside `save_findings` (+ test columns populated).
6. `mine_keywords` helper (+ tests for casing/join expansion).
7. `locate(..., hint_paths=)` + prompt injection; wire `agent.py` pre-step.
8. Overlap instrumentation in the locate path.

Steps 1–6 are pure `agent/workspace/db.py` + tests, no behavior change to the
running agent — safe to land incrementally. Behavior changes only at step 7.
