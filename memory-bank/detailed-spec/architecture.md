# Architecture
Precision-scoped AI coding agent with checkpoint-oriented design and LLM-backed intent routing

## Package Layout
`agent/` root: `agent.py` (orchestration), `router.py` (intents + session), `tools.py` (read/search/grep),
`settings.py` (permissions), `persistence.py` (JSONL append), `normalizer.py` + `subagent.py` (support infrastructure)
Subpackages: `handlers/` (chat, query, action, display), `ws_explorer/` (workspace enrichment + SubAgent),
`tui/` (Textual app — see tui-layout.md), `commands/` (slash command registry)

## Session
`Session` in `router.py`; holds a GUID, `messages: list[dict]`, `working_dir`, `permissions`

`messages` starts with two system entries: `SYSTEM_PROMPT` at `[0]` + workspace context at `[1]` (TOON-encoded)
Workspace context `<workspace>` block begins with `"verified repository metadata — treat as authoritative for high-level questions:"` preamble line
Injected subset: `workspace_name`, `workspace_type`, `branch`, `primary_languages`, `projects`; `extensions` when `projects` empty; `domain_map` when present

`GekaiAgent.process_stream()` is the sole owner of session writes:
- appends `{"role": "user"}` once per turn before dispatching
- appends `{"role": "assistant"}` once per turn after all segments complete
- `memorize` segments append an additional `{"role": "system", "content": "[preference] ..."}` inline

## Intent Routing
`IntentClassifier.classify(user_input, history=None)` decomposes user input into `list[tuple[Intent, str, bool]]` via one LLM call
`history` — optional list of session messages; last 6 user/assistant turns prepended as context before the user message
Third element is a `plan` boolean flag (whether enrichment planning is needed)
Returns `[(Intent.CHAT, user_input, False)]` on unparseable output

Before classification, `PromptNormalizer` normalizes/translates user input using the support model
`GekaiAgent.normalize()` returns `(normalized_prompt, source_language_or_None)`; original input stored as the session user message

Intents:
- `chat` — general coding Q&A; answer from model knowledge + session history
- `query` — needs repo inspection: read files, search code, understand structure
- `display` — user wants raw file contents printed verbatim; no summarization or analysis
- `action` — modifies repository files (create, edit, delete, refactor)
- `memorize` — user states a rule/preference; stored as system message in session
- `clarify` — classifier thinks message is ambiguous; routes to CHAT handler anyway (model decides with full session context)

`GekaiAgent._handlers` maps `Intent → Handler`; `CLARIFY` routes to `Intent.CHAT` handler; `DISPLAY` routes to `DisplayHandler`

## LLM Integration
`openai` SDK (`AsyncOpenAI`) for chat and classification; `llmstitch` for tool-calling loop in QueryHandler
Env vars (CORE — used by ChatHandler, QueryHandler, ActionHandler):
- `GEKAI_CORE_MODEL_NAME` — model id, e.g. `deepseek-chat`
- `GEKAI_CORE_MODEL_KEY`
- `GEKAI_CORE_MODEL_URL` — e.g. `https://api.deepseek.com/v1`

Env vars (SUPP — used by `IntentClassifier`, `PromptNormalizer`, and `WsExplorer`/`enrich_workspace`; `GEKAI_SUPPORT_MODEL_NAME` and `GEKAI_SUPPORT_MODEL_KEY` are required; `GEKAI_SUPPORT_MODEL_URL` defaults to CORE equivalent if unset):
- `GEKAI_SUPPORT_MODEL_NAME`
- `GEKAI_SUPPORT_MODEL_KEY`
- `GEKAI_SUPPORT_MODEL_URL`

`QueryHandler` appends `_TOOL_INSTRUCTION` to `SYSTEM_PROMPT` at agent construction — balanced rule: `<workspace>` block is authoritative for high-level questions (proj_brief, tech_stack, primary_languages, branch, domain_map); tools are mandatory for file contents, implementation details, logic, or architecture depth

## Workspace Scan / WsExplorer
`WsExplorer` (`ws_explorer/subagent.py`) is a `SubAgent` that runs scan + enrichment and streams `SubAgentEvent` instances to the TUI
Activation policy: runs at startup and on `/workspace:rebuild`; when `query+plan` intent is detected and workspace may be stale, a Y/N confirmation is shown before re-running
`.gekai/` directory is excluded from workspace scan activation triggers

`scan_workspace(working_dir)` (in `workspace.py`) writes `.gekai/workspace.json`
Phases:
1. repo name (git remote) + branch
2. manifest scan (recursive, root + subdirs, skip hidden/vendor) → `projects` list with lang + path
3. extension frequency (top 15) → `extensions` map; used as primary signal only when no manifests
4. AI instruction file detection (root + one level deep): `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, `.windsurfrules`, `.clinerules`, `.github/copilot-instructions.md`, etc.

`enrich_workspace` (`ws_explorer/enrichment.py`) runs two parallel LLM calls (proj_brief + domain_map); fires async callbacks `on_file`, `on_infer_start`, `on_infer_delta`, `on_infer_end` for TUI progress

Cache: `workspace.json` reused if mtime < 15 min (fresh session) or < 30 min (resume). `created_at` preserved across re-scans.

## Session Persistence
Sessions stored as JSONL at `~/.gekai/workspaces/{normalized-repo-path}/{session-id}.jsonl`; each line is a timestamped message appended via `append_message()`
Debug messages written to `{session-id}.debug.jsonl` via `append_debug()`
`load_session()` returns `(session_id, user/assistant messages)`; system messages are excluded (re-injected fresh on startup)

On resume (`--resume <session-id>`): restores session ID and user/assistant messages; re-injects fresh system messages; prints conversation history to terminal; skips workspace scan if cache valid

## Streaming UX
Textual exclusive worker per turn; see `tui-layout.md → Streaming Worker`.
- spinner `| / - \` + random operative verb + token count in `#status-line` (accent color via `styles.color`)
- ESC cancels worker; `Ctrl+C` quits app
- on completion: ASSISTANT widget (Markdown) + OPERATION widget (`* {PastVerb} for {duration}`)

## Commands
Slash-prefixed input intercepted by `CommandPalette` then dispatched via `CommandRegistry`.
- `/exit` — exit to terminal (with farewell message + delay)
- `/clear` — clears chat and starts a new session (resets session ID)
- `/workspace:rebuild` — re-scan + AI-enrich workspace; handled directly in `GekaiApp._rebuild_workspace()`, not via registry dispatch

## CLI Flags
- `--debug` — prints `[classifier: INTENT+plan, ...]` in color `#BA55D3` (medium_orchid) as an OPERATION widget in the TUI chat, 1 line below the user prompt
- `--resume` / `-r` — resume a previous session by ID
- `--working-dir` / `-d` — override working directory (default: cwd)