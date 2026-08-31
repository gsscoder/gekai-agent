# Caching
Catalog of caching mechanisms in gekai-agent. Each cache gets its own subsection below, documenting storage, schema, write/read paths, staleness handling, and wiring into the agent. Currently one cache exists (locator/workspace), but new caches should be added here as their own `##` sections.

## Locator cache (workspace.db) — removed

**Status**: the module this section used to document no longer exists in the codebase. No
`agent/workspace/db.py`, no `agent/workspace/indexer.py`, no `sqlite3` import anywhere under
`agent/`, and no `SCHEMA_VERSION` constant (verified via `git grep -rn "SCHEMA_VERSION" agent/` —
no hits). `save_findings`, `find_candidates`, `mine_keywords`, and `.gekai/workspace.db` have zero
remaining references in code; the previously-documented "write side alive, read side orphaned"
state (write side populated by `ws_manager`'s onboarding walk) no longer holds either — that
caller is also gone. There is currently no locator-cache/keyword-hint mechanism in the agent at
all; file discovery is done entirely by the main agent's own read tools.

See `memory-bank/locator-cache.md` for the historical v1 design plan/build-order.