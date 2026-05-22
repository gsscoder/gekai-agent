# Architecture
Precision-scoped AI coding agent with checkpoint-oriented design and LLM-backed intent routing

## Package Layout
```
agent/
  main.py          — REPL entry point, CLI flags, streaming loop, ESC cancellation
  agent.py         — GekaiAgent: classifies intent, routes to handler, owns session writes
  router.py        — Intent enum, Session dataclass, IntentClassifier (LLM call)
  router.py        — SYSTEM_PROMPT, CLASSIFIER_PROMPT constants
  persistence.py   — save_session / load_session (JSON, ~/.gekai/sessions/)
  workspace.py     — scan_workspace, get_git_branch, AI instruction file detection
  tools.py         — make_tools() factory: read_file, list_files, grep (llmstitch @tool)
  ui.py            — console, render_response (Markdown), spinner, banner, operative verbs
  settings.py      — Permissions dataclass, load/prompt helpers
  handlers/
    base.py        — Handler protocol
    chat.py        — ChatHandler: LLM streaming, yields chunks only (no session writes)
    query.py       — QueryHandler: llmstitch Agent tool loop
    action.py      — ActionHandler: stub
  commands/
    base.py        — Command protocol, CommandResult dataclass
    registry.py    — CommandRegistry: /name dispatch
    exit.py        — ExitCommand: /exit
```

## Session
`Session` in `router.py`; holds a GUID, `messages: list[dict]`, `working_dir`, `permissions`

`messages` starts with two system entries: `SYSTEM_PROMPT` + workspace context (TOON-encoded)

`GekaiAgent.process_stream()` is the sole owner of session writes:
- appends `{"role": "user"}` once per turn before dispatching
- appends `{"role": "assistant"}` once per turn after all segments complete
- `memorize` segments append an additional `{"role": "system", "content": "[preference] ..."}` inline

## Intent Routing
`IntentClassifier` decomposes user input into `list[tuple[Intent, str]]` via one LLM call
Returns `[(Intent.CHAT, user_input)]` on unparseable output

Intents:
- `chat` — general coding Q&A; answer from model knowledge + session history
- `query` — needs repo inspection: read files, search code, understand structure
- `action` — modifies repository files (create, edit, delete, refactor)
- `memorize` — user states a rule/preference; stored as system message in session
- `clarify` — classifier thinks message is ambiguous; routes to CHAT handler anyway (model decides with full session context)

`GekaiAgent._handlers` maps `Intent → Handler`; `CLARIFY` routes to `Intent.CHAT` handler
Classifier is stateless (no session history passed); context continuity is the handlers' responsibility

## LLM Integration
`openai` SDK (`AsyncOpenAI`) for chat and classification; `llmstitch` for tool-calling loop in QueryHandler
Env vars:
- `GEKAI_DEFAULT_MODEL` — model id, e.g. `deepseek-chat`
- `GEKAI_API_KEY`
- `GEKAI_BASE_URL` — e.g. `https://api.deepseek.com/v1`

## Workspace Scan
`scan_workspace(working_dir)` runs at startup; writes `.gekai/workspace.json`

Phases (each calls `on_step` callback for progress display):
1. repo name (git remote) + branch
2. manifest scan (recursive, root + subdirs, skip hidden/vendor) → `projects` list with lang + path
3. extension frequency (top 15) → `extensions` map; used as primary signal only when no manifests
4. AI instruction file detection (root + one level deep): `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, `.windsurfrules`, `.clinerules`, `.github/copilot-instructions.md`, etc.

Cache: `workspace.json` reused if mtime < 15 min (fresh session) or < 30 min (resume). `created_at` preserved across re-scans.

Workspace context injected into session as second system message (TOON-encoded subset: repo_name, branch, primary_languages, projects; extensions only when projects empty). `ai_instructions` stored but not injected yet.

## Session Persistence
Sessions saved to `~/.gekai/sessions/{normalized-repo-path}/{short-id}.json` after each turn
Fields: `session_id`, `working_dir`, `created_at`, `last_accessed_at`, `messages`

On resume (`--resume <session-id>`): restores session ID and user/assistant messages; re-injects fresh system messages; prints conversation history to terminal; skips workspace scan if cache valid

## Streaming UX
`Rich.Live(transient=True)` during each turn:
- spinner `| / - \` + random operative verb (surgical/medical terms) + token count
- ESC cancels in-flight interaction via `threading.Event` + `asyncio.wait(FIRST_COMPLETED)`
- on completion: spinner clears, operation summary (`* {PastVerb} for {duration}`), response rendered as Markdown with `●` prefix

## Commands
Slash-prefixed input dispatched via `CommandRegistry` before agent routing
Current: `/exit`

## CLI Flags
- `--debug` / `-d` — prints `debug: {intent}: {sub_prompt}` per segment in grey50
- `--resume` / `-r` — resume a previous session by ID
- `--working-dir` — override working directory (default: cwd)