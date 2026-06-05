# Architecture
Precision-scoped AI coding agent with checkpoint-oriented design and LLM-backed intent routing

## Package Layout
`agent/` root: `agent.py` (orchestration), `router.py` (intents + session), `tools.py` (read/search/grep),
`settings.py` (permissions), `permissions.py` (permission gate + callback), `persistence.py` (JSONL append), `normalizer.py` + `subagent.py` (support infrastructure)
Subpackages: `handlers/` (chat, action), `profiles/` (`__init__.py` + one file per profile + `_coding.py` shared directives), `ws_manager/` (workspace enrichment + SubAgent — dead code),
`tui/` (Textual app — see tui-layout.md), `commands/` (slash command registry)

## Session
`Session` in `router.py`; holds a GUID, `messages: list[dict]`, `working_dir`, `permissions`, `scope_gate: bool = True`, `blast_radius_limit: int = 5`

`messages` starts with two system entries: `SYSTEM_PROMPT` at `[0]` + workspace context at `[1]` (TOON-encoded)
Workspace context `<workspace>` block begins with `"verified repository metadata — treat as authoritative for high-level questions:"` preamble line
Injected subset: `workspace_name`, `workspace_type`, `branch`, `primary_languages`, `projects`; `extensions` when `projects` empty; `domain_map` when present

`GekaiAgent.process_stream(session, user_input, route, entries=None, permission_callback=None)` is the sole owner of session writes:
`route: Route`; `entries: list[tuple[str, list[str]]] | None` — pre-computed locate results (None for chat/generic); `permission_callback: PermissionCallback | None`
- appends `{"role": "user"}` once per turn before dispatching
- appends `{"role": "assistant"}` once per turn after handler completes

## Router
`Router.route(user_input, history=None)` — single SUPP-model LLM call, temperature 0. Returns `Route`.

`Route` dataclass: `intent: Intent`, `profile: AgentProfile | None`; property `namespace` (derived from `profile.namespace`, or `"generic"` for action/generic, or `None` for chat/rejected).

History: last 6 user/assistant turns from session messages prepended before user message.

Router output token → Route mapping:
- `chat` → `Route(CHAT)`
- `action/generic` → `Route(ACTION, profile=None)`
- `REJECTED` → `Route(REJECTED)` — non-English input
- `<profile-name>` → `Route(ACTION, profile=<matched>)`
- unknown token → warning log + fallback `Route(ACTION, profile=None)`

Intents: `CHAT`, `ACTION`, `REJECTED`
`GekaiAgent._handlers` maps `Intent → Handler`; keys are `Intent.CHAT` and `Intent.ACTION`
Verbatim file output is handled by the `<file_handling>` rule in `SYSTEM_PROMPT`, not a dedicated intent

## Blast-Radius Gate
Applies only to profiled action routes (`route.profile is not None`). `chat` and `action/generic` bypass gate.

Pipeline (in TUI `_stream`):
1. `Router.route()` → `Route`
2. If profiled: `BlastRadiusLocator.locate(working_dir, user_input)` → `entries: list[tuple[path, keywords]]`
3. `evaluate_blast_radius_gate(entries, session.blast_radius_limit)` → `(rejected, reason)`
4. If rejected and `session.scope_gate`: display rejection, return

`BlastRadiusLocator` — agentic SUPP-model call (up to 5 iterations, read-only tools). Roams freely — reads any file type. Any exception propagates (fail-hard).

Area metric — **ancestor-collapsed directory count, code files only:**
1. Filter `entries` to `_CODE_EXTENSIONS` paths only
2. Collect parent dir of each surviving file
3. Drop any dir that has an ancestor also in the set
4. Count survivors

`_CODE_EXTENSIONS`: all popular languages — `.py .pyi .ipynb` · `.js .jsx .mjs .cjs` · `.ts .tsx` · `.vue .svelte` · `.go` · `.java` · `.cs` · `.kt .kts` · `.swift` · `.rs` · `.c .h .cpp .cc .cxx .hpp` · `.rb` · `.php` · `.scala` · `.dart` · `.ex .exs` · `.lua` · `.hs` · `.r`
Broader than `_EXT_TO_LANG` (AST support) — gate coverage ≠ symbol-parse coverage.
Manifests/configs/docs (`pyproject.toml`, `package.json`, `.yaml`, `.md`, etc.) are inspected but never counted.
All-config change → 0 areas → always passes.

`blast_radius_limit` loaded via `load_blast_radius_limit(working_dir)`: project `.gekai/settings.local.json` overrides user `~/.gekai/settings.json`; absent → `5`. Manual JSON edit only — no slash command. Gate on/off reuses `/config:gate on|off`.

## LLM Integration
`openai` SDK (`AsyncOpenAI`) for chat and classification; `llmstitch` for tool-calling loop in ActionHandler
Env vars (CORE — used by ChatHandler, ActionHandler):
- `GEKAI_CORE_MODEL_NAME` — model id, e.g. `deepseek-chat`
- `GEKAI_CORE_MODEL_KEY`
- `GEKAI_CORE_MODEL_URL` — e.g. `https://api.deepseek.com/v1`

Env vars (SUPP — used by `Router`, `BlastRadiusLocator`; `WsManager`/`enrich_workspace` also reference these but are dead code; `GEKAI_SUPPORT_MODEL_NAME` and `GEKAI_SUPPORT_MODEL_KEY` are required; `GEKAI_SUPPORT_MODEL_URL` defaults to CORE equivalent if unset):
- `GEKAI_SUPPORT_MODEL_NAME`
- `GEKAI_SUPPORT_MODEL_KEY`
- `GEKAI_SUPPORT_MODEL_URL`

`ActionHandler._build_system(profile)` composes the transient system prompt: `SYSTEM_PROMPT` + optional `<directives>` block (profile.directives when non-empty) + `<tools>` block (`_TOOL_INSTRUCTION`); tags are non-closing
`_TOOL_INSTRUCTION` — if the question requires file contents, implementation details, logic, or architecture depth, use tools to read actual files; do not guess or rely on training knowledge; when multiple targets are nearby, prefer one wider ranged read_file call over many individual reads; no `<workspace>` reference
`ActionHandler.stream()` is a transient incarnation: passes `_recency_turns(session.messages, _RECENCY_N=2)` (last 2 user/assistant pairs, system messages skipped, trailing user input excluded) + current user input; full session history is NOT passed to the action agent
`--debug` active: `stream()` calls `append_debug(session, {"content": system})` before dispatch — transient system string written to `.debug.jsonl`

## Workspace Scan / WsManager [dead code]
`WsManager` (`ws_manager/subagent.py`), `scan_workspace`, `enrich_workspace`, and all activation logic are dead code — present in source but not called from any live path; disabled pending redesign

`GekaiAgent.create_ws_manager()` and `GekaiAgent.update_workspace_context()` exist but are only reachable from dead methods

`_run_ws_manager()`, `_maybe_rescan_workspace()`, and `_rebuild_workspace()` in `tui/app.py` carry `# [dead code]` markers; none are reachable from the live startup or stream path

`/workspace:rebuild` command (`commands/workspace.py`) carries a `# [dead code]` marker; the command class is not registered and `execute()` returns an empty `CommandResult`

Staleness logic (`_maybe_rescan_workspace` + `load_ws_scan_staleness_min`) is dead for the same reason

**What remains active:** `_init_session()` still reads `.gekai/workspace.json` unconditionally if it exists and passes the dict to `agent.start_session()`, which injects the `<workspace>` block into `session.messages[1]` via `_format_workspace_context()`. The cache file can exist from a prior run; if absent, an empty dict is used and the block is injected with default/unknown values. No live code writes `workspace.json`.

`scan_workspace(working_dir)` (in `workspace.py`) — phases below are inactive:
1. repo name (git remote) + branch
2. manifest scan (recursive, root + subdirs, skip hidden/vendor) → `projects` list with lang + path
3. extension frequency (top 15) → `extensions` map; used as primary signal only when no manifests
4. AI instruction file detection (root + one level deep): `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, `.windsurfrules`, `.clinerules`, `.github/copilot-instructions.md`, etc.

`enrich_workspace` (`ws_manager/enrichment.py`) — inactive: two parallel LLM calls (proj_brief + domain_map); fires async callbacks `on_file` and `on_infer_end` for TUI progress

## Session Persistence
Sessions stored as JSONL at `~/.gekai/workspaces/{normalized-repo-path}/{session-id}.jsonl`; each line is a timestamped message appended via `append_message()`
Debug messages written to `{session-id}.debug.jsonl` via `append_debug()`
`load_session()` returns `(session_id, working_dir, user/assistant + persistent system messages)`; always-fresh system messages (SYSTEM_PROMPT, workspace) are excluded and re-injected on startup

On resume (`--resume <session-id>`): restores session ID and user/assistant messages; re-injects fresh system messages; prints conversation history to terminal; reads `workspace.json` cache if present (workspace scan is dead code — no rescan occurs)

## Streaming UX
Textual exclusive worker per turn; see `tui-layout.md → Streaming Worker`.
- spinner `· • ● •` + random operative verb + elapsed time in `#status-line` (accent color via `styles.color`)
- ESC cancels worker; `Ctrl+C` quits app
- on completion: ASSISTANT widget (Markdown) + OPERATION widget (`* {PastVerb} for {duration}`)

## Commands
Slash-prefixed input intercepted by `CommandPalette` then dispatched via `CommandRegistry`.
- `/exit` — exit to terminal (with farewell message + delay)
- `/clear` — clears chat and starts a new session (resets session ID)
- `/config:gate on|off` — enable or disable the scope gate for the current project; persists to `.gekai/settings.local.json`; updates `session.scope_gate` immediately
- `/workspace:rebuild` — [dead code] command class exists in `commands/workspace.py` but is not registered; `_rebuild_workspace()` in `app.py` carries a `# [dead code]` marker and is unreachable

## CLI Flags
- `--debug` — prints `[router: INTENT/namespace/profile]` in color `#BA55D3` (medium_orchid) as an OPERATION widget in the TUI chat, 1 line below the user prompt
- `--resume` / `-r` — resume a previous session by ID
- `--working-dir` / `-d` — override working directory (default: cwd)