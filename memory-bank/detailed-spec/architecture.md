# Architecture
Precision-scoped AI coding agent with checkpoint-oriented design and LLM-backed intent routing

## Package Layout
`agent/` root: `agent.py` (orchestration), `router.py` (intents + session), `tools.py` (read/search/grep),
`settings.py` (permissions), `permissions.py` (permission gate + callback), `persistence.py` (JSONL append), `normalizer.py` + `subagent.py` (support infrastructure)
Subpackages: `handlers/` (chat, action), `ws_explorer/` (workspace enrichment + SubAgent — dead code),
`tui/` (Textual app — see tui-layout.md), `commands/` (slash command registry)

## Session
`Session` in `router.py`; holds a GUID, `messages: list[dict]`, `working_dir`, `permissions`, `scope_gate: bool = True`

`messages` starts with two system entries: `SYSTEM_PROMPT` at `[0]` + workspace context at `[1]` (TOON-encoded)
Workspace context `<workspace>` block begins with `"verified repository metadata — treat as authoritative for high-level questions:"` preamble line
Injected subset: `workspace_name`, `workspace_type`, `branch`, `primary_languages`, `projects`; `extensions` when `projects` empty; `domain_map` when present

`GekaiAgent.process_stream(session, user_input, segments, original_input=None, permission_callback=None)` is the sole owner of session writes:
`segments: list[tuple[Intent, str]]`; `permission_callback: PermissionCallback | None = None` passed to the `PermissionGate` for runtime grant prompts
- appends `{"role": "user"}` once per turn before dispatching
- appends `{"role": "assistant"}` once per turn after all segments complete
- `memorize` segments append an additional `{"role": "system", "content": "[preference] ..."}` inline

## Intent Routing
`IntentClassifier.classify(user_input, history=None)` decomposes user input into `list[tuple[Intent, str]]` via one LLM call
`history` — optional list of session messages; last 6 user/assistant turns prepended as context before the user message
Returns `[(Intent.CHAT, user_input)]` on unparseable output

Before classification, `PromptNormalizer` normalizes/translates user input using the support model
`GekaiAgent.normalize()` returns `(normalized_prompt, source_language_or_None)`; original input stored as the session user message

Intents:
- `chat` — general coding Q&A; answer from model knowledge + session history
- `action` — needs to inspect or modify the repository; permission resolved at tool-call time by PermissionGate
- `memorize` — user states a rule/preference; stored as system message in session

`GekaiAgent._handlers` maps `Intent → Handler`; keys are `Intent.CHAT` and `Intent.ACTION`
Verbatim file output is handled by the `<file_handling>` rule in `SYSTEM_PROMPT`, not a dedicated intent

## LLM Integration
`openai` SDK (`AsyncOpenAI`) for chat and classification; `llmstitch` for tool-calling loop in ActionHandler
Env vars (CORE — used by ChatHandler, ActionHandler):
- `GEKAI_CORE_MODEL_NAME` — model id, e.g. `deepseek-chat`
- `GEKAI_CORE_MODEL_KEY`
- `GEKAI_CORE_MODEL_URL` — e.g. `https://api.deepseek.com/v1`

Env vars (SUPP — used by `IntentClassifier` and `PromptNormalizer`; `WsExplorer`/`enrich_workspace` also reference these but are dead code; `GEKAI_SUPPORT_MODEL_NAME` and `GEKAI_SUPPORT_MODEL_KEY` are required; `GEKAI_SUPPORT_MODEL_URL` defaults to CORE equivalent if unset):
- `GEKAI_SUPPORT_MODEL_NAME`
- `GEKAI_SUPPORT_MODEL_KEY`
- `GEKAI_SUPPORT_MODEL_URL`

`ActionHandler` appends `_TOOL_INSTRUCTION` to `SYSTEM_PROMPT` at agent construction — balanced rule: `<workspace>` block is authoritative for high-level questions (proj_brief, tech_stack, primary_languages, branch, domain_map); tools are mandatory for file contents, implementation details, logic, or architecture depth

## Workspace Scan / WsExplorer [dead code]
`WsExplorer` (`ws_explorer/subagent.py`), `scan_workspace`, `enrich_workspace`, and all activation logic are dead code — present in source but not called from any live path; disabled pending redesign

`GekaiAgent.create_ws_explorer()` and `GekaiAgent.update_workspace_context()` exist but are only reachable from dead methods

`_run_ws_explorer()`, `_maybe_rescan_workspace()`, and `_rebuild_workspace()` in `tui/app.py` carry `# [dead code]` markers; none are reachable from the live startup or stream path

`/workspace:rebuild` command (`commands/workspace.py`) carries a `# [dead code]` marker; the command class is not registered and `execute()` returns an empty `CommandResult`

Staleness logic (`_maybe_rescan_workspace` + `load_ws_scan_staleness_min`) is dead for the same reason

**What remains active:** `_init_session()` still reads `.gekai/workspace.json` unconditionally if it exists and passes the dict to `agent.start_session()`, which injects the `<workspace>` block into `session.messages[1]` via `_format_workspace_context()`. The cache file can exist from a prior run; if absent, an empty dict is used and the block is injected with default/unknown values. No live code writes `workspace.json`.

`scan_workspace(working_dir)` (in `workspace.py`) — phases below are inactive:
1. repo name (git remote) + branch
2. manifest scan (recursive, root + subdirs, skip hidden/vendor) → `projects` list with lang + path
3. extension frequency (top 15) → `extensions` map; used as primary signal only when no manifests
4. AI instruction file detection (root + one level deep): `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, `.windsurfrules`, `.clinerules`, `.github/copilot-instructions.md`, etc.

`enrich_workspace` (`ws_explorer/enrichment.py`) — inactive: two parallel LLM calls (proj_brief + domain_map); fires async callbacks `on_file` and `on_infer_end` for TUI progress

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
- `--debug` — prints `[classifier: INTENT, ...]` in color `#BA55D3` (medium_orchid) as an OPERATION widget in the TUI chat, 1 line below the user prompt
- `--resume` / `-r` — resume a previous session by ID
- `--working-dir` / `-d` — override working directory (default: cwd)